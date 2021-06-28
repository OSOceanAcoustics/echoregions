import pandas as pd
import os
import numpy as np
from .ev_parser import EvParserBase
from .utils import parse_time


class Regions2DParser(EvParserBase):
    """Class for parsing EV 2D region (EVR) files.
    Using this class directly is not recommended; use Regions2D instead.
    """
    def __init__(self, input_file=None):
        super().__init__(input_file, 'EVR')
        self.raw_range = None
        self.min_depth = None   # Set to replace -9999.9900000000 depth values which are EVR min range
        self.max_depth = None   # Set to replace 9999.9900000000 depth values which are EVR max range
        self.offset = 0         # Set to apply depth offset (meters)

    def _parse(self, fid):
        """Reads an open file and returns the file metadata and region information"""
        def _region_metadata_to_dict(line):
            """Assigns a name to each value in the metadata line for each region"""
            top_y = self.swap_depth_edge(line[9])
            bottom_y = self.swap_depth_edge(line[12])
            bound_calculated = int(line[6])
            if bound_calculated:
                left_x = parse_time(f'{line[7]} {line[8]}', unix=False)
                right_x = parse_time(f'{line[10]} {line[11]}', unix=False)
            else:
                left_x = f'D{line[7]} {line[8]}'
                right_x = f'D{line[10]} {line[11]}'

            return {
                'region_id': int(line[2]),
                'structure_version': line[0],                               # 13 currently
                'point_count': line[1],                                     # Number of points in the region
                'selected': line[3],                                        # Always 0
                'creation_type': line[4],                                   # How the region was created
                'dummy': line[5],                                           # Always -1
                'bounding_rectangle_calculated': bound_calculated,          # 1 if next 4 fields valid. O otherwise
                # Date encoded as CCYYMMDD and times in HHmmSSssss
                # Where CC=Century, YY=Year, MM=Month, DD=Day, HH=Hour, mm=minute, SS=second, ssss=0.1 milliseconds
                'bounding_rectangle_left_x': left_x,                        # Time and date of bounding box left x
                'bounding_rectangle_right_x': right_x,                      # Time and date of bounding box right x
                'bounding_rectangle_top_y': top_y,                          # Top of bounding box
                'bounding_rectangle_bottom_y': bottom_y,                    # Bottom of bounding box
            }

        def _points_to_list(line):
            """Takes a line with point information and creates a tuple (x, y) for each point"""
            points_x = parse_time([f'{line[idx]} {line[idx + 1]}' for idx in range(0, len(line), 3)]).values
            points_y = np.array([self.swap_depth_edge(line[idx + 2]) for idx in range(0, len(line), 3)])
            return points_x, points_y

        # Read header containing metadata about the EVR file
        file_type, file_format_number, echoview_version = self.read_line(fid, True)
        file_metadata = pd.Series({
            'file_name': os.path.splitext(os.path.basename(self.input_file))[0],
            'file_type': file_type,
            'file_format_number': file_format_number,
            'echoview_version': echoview_version
        })
        df = pd.DataFrame()
        row = {}
        n_regions = int(self.read_line(fid))
        # Loop over all regions in file
        for r in range(n_regions):
            # Unpack region data
            fid.readline()    # blank line separates each region
            r_metadata = _region_metadata_to_dict(self.read_line(fid, True))
            # Add notes to region data
            n_note_lines = int(self.read_line(fid))
            r_notes = [self.read_line(fid) for line in range(n_note_lines)]
            # Add detection settings to region data
            n_detection_setting_lines = int(self.read_line(fid))
            r_detection_settings = [self.read_line(fid) for line in range(n_detection_setting_lines)]
            # Add classification to region data
            r_metadata['region_classification'] = self.read_line(fid)
            # Add point x and y
            points_line = self.read_line(fid, True)
            # For type: 0=bad (No data), 1=analysis, 3=fishtracks, 4=bad (empty water)
            r_metadata['type'] = points_line.pop()
            r_points = _points_to_list(points_line)
            r_metadata['name'] = self.read_line(fid)

            # Store region data into a GeoDataFrame
            row = pd.concat([
                file_metadata,
                pd.Series(r_metadata)[r_metadata.keys()],
                pd.Series({'ping_time': r_points[0]}),
                pd.Series({'depth': r_points[1]}),
                pd.Series({'notes': r_notes}),
                pd.Series({'detection_settings': r_detection_settings})
            ])
            df = df.append(row, ignore_index=True)

        return df[row.keys()].convert_dtypes()

    def set_range_edge_from_raw(self, raw, model='EK60'):
        try:
            import echopype as ep
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("This function requires 'echopype' to be installed") from e

        remove = False
        if raw.endswith('.raw') and os.path.isfile(raw):
            tmp_c = ep.Convert(raw, model=model)
            tmp_c.to_netcdf(save_path='./')
            raw = tmp_c.output_file
            remove = True
        elif not raw.endswith('.nc') and not raw.endswith('.zarr'):
            raise ValueError("Invalid raw file")

        ed = ep.process.EchoData(raw)
        proc = ep.process.Process(model, ed)
        # proc.get_range # Calculate range directly as opposed to with get_Sv
        proc.get_Sv(ed)

        self.raw_range = ed.range.isel(frequency=0, ping_time=0).load()

        self.max_depth = self.raw_range.max().values
        self.min_depth = self.raw_range.min().values

        ed.close()
        if remove:
            os.remove(tmp_c.output_file)

    def swap_depth_edge(self, depth):
        """Replace 9999.99 and -9999.99 edge values with user specified min and max values.
        Applies offset to depth value if depth is not an edge value.

        Parameters
        ----------
        depth : float
            Depth in meters or depth edge

        Returns
        -------
        float
            Depth in meters
        """
        depth = float(depth)
        if depth == 9999.99 and self.max_depth is not None:
            return self.max_depth
        elif depth == -9999.99 and self.min_depth is not None:
            return self.min_depth
        elif depth != -9999.99 and depth != 9999.99:
            return depth + self.offset
        else:
            return depth
