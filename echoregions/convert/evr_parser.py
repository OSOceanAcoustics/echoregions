import pandas as pd
from collections import defaultdict
import os
import copy
from .ev_parser import EvParserBase
from .utils import parse_time, validate_path


class Region2DParser(EvParserBase):
    """Class for parsing EV 2D region (EVR) files.
    Using this class directly is not recommended; use Regions2D instead.
    """
    def __init__(self, input_file=None):
        super().__init__(input_file, 'EVR')
        self._raw_range = None
        self._min_depth = None      # Set to replace -9999.9900000000 range values which are EVR min range
        self._max_depth = None      # Set to replace 9999.9900000000 range values which are EVR max range
        self.raw_range = None
        self.min_depth = None
        self.max_depth = None

    def _parse(self, fid, convert_time=False, convert_range_edges=False, offset=0):
        """Reads an open file and returns the file metadata and region information"""
        def _region_metadata_to_dict(line):
            """Assigns a name to each value in the metadata line for each region"""
            top_y = self.swap_range_edge(line[9]) if convert_range_edges else line[9]
            bottom_y = self.swap_range_edge(line[12]) if convert_range_edges else line[12]
            top_y = float(top_y) + offset
            bottom_y = float(bottom_y) + offset

            left_x = parse_time(f'D{line[7]}T{line[8]}') if convert_time else f'D{line[7]}T{line[8]}'
            right_x = parse_time(f'D{line[10]}T{line[11]}') if convert_time else f'D{line[10]}T{line[11]}'
            return {
                'structure_version': line[0],                               # 13 currently
                'point_count': line[1],                                     # Number of points in the region
                'selected': line[3],                                        # Always 0
                'creation_type': line[4],                                   # How the region was created
                'dummy': line[5],                                           # Always -1
                'bounding_rectangle_calculated': line[6],                   # 1 if next 4 fields valid. O otherwise
                # Date encoded as CCYYMMDD and times in HHmmSSssss
                # Where CC=Century, YY=Year, MM=Month, DD=Day, HH=Hour, mm=minute, SS=second, ssss=0.1 milliseconds
                'bounding_rectangle_left_x': left_x,                        # Time and date of bounding box left x
                'bounding_rectangle_top_y': top_y,                          # Top of bounding box
                'bounding_rectangle_right_x': right_x,                      # Time and date of bounding box right x
                'bounding_rectangle_bottom_y': bottom_y,                    # Bottom of bounding box
            }

        def _points_to_dict(line):
            """Takes a line with point information and creates a tuple (x, y) for each point"""
            points = {}
            for point_num, idx in enumerate(range(0, len(line), 3)):
                x = f'D{line[idx]}T{line[idx + 1]}'
                if convert_time:
                    x = parse_time(x)
                y = line[idx + 2]
                if convert_range_edges:
                    if y == '9999.9900000000' and self.max_depth is not None:
                        y = float(self.max_depth)
                    elif y == '-9999.9900000000' and self.min_depth is not None:
                        y = float(self.min_depth)

                points[point_num] = [x, y]
            return points

        # Read header containing metadata about the EVR file
        file_type, file_format_number, echoview_version = self.read_line(fid, True)
        file_metadata = {
            'file_name': os.path.splitext(os.path.basename(self.input_file))[0],
            'file_type': file_type,
            'file_format_number': file_format_number,
            'echoview_version': echoview_version
        }

        regions = defaultdict(dict)
        n_regions = int(self.read_line(fid))
        # Loop over all regions in file
        for r in range(n_regions):
            fid.readline()    # blank line separates each region
            region_metadata = self.read_line(fid, True)
            rid = region_metadata[2]        # Region ID (unique for each region)

            regions[rid]['metadata'] = _region_metadata_to_dict(region_metadata)
            # Add notes to region data
            n_note_lines = int(self.read_line(fid))
            regions[rid]['notes'] = [self.read_line(fid) for line in range(n_note_lines)]
            # Add detection settings to region data
            n_detection_setting_lines = int(self.read_line(fid))
            regions[rid]['detection_settings'] = [self.read_line(fid) for line in range(n_detection_setting_lines)]
            # Add classification to region data
            regions[rid]['metadata']['region_classification'] = self.read_line(fid)
            # Add point x and y
            points_line = self.read_line(fid, True)
            # For type: 0=bad (No data), 1=analysis, 3=fishtracks, 4=bad (empty water)
            regions[rid]['metadata']['type'] = points_line.pop()
            regions[rid]['points'] = _points_to_dict(points_line)
            regions[rid]['metadata']['name'] = self.read_line(fid)

        return file_metadata, regions

    def to_dataframe(self, **kwargs):
        # Parse EVR file if it hasn't already been done
        if not self.output_data:
            self.parse_file(**kwargs)

        df = pd.DataFrame()
        # Save file metadata for each point
        metadata = pd.Series(self.output_data['metadata'])
        row = []
        # Loop over each region
        for rid, region in self.output_data['regions'].items():
            # Save region information for each point
            region_metadata = pd.Series(region['metadata'])
            region_notes = pd.Series({'notes': region['notes']})
            detection_settings = pd.Series({'detection_settings': region['detection_settings']})
            region_id = pd.Series({'region_id': rid})
            # Loop over each point in each region. One row of the dataframe corresponds to one point
            for p, point in enumerate(region['points'].values()):
                point = pd.Series({
                    'point_idx': str(p),
                    'ping_time': point[0],
                    'depth': point[1],
                })
                row = pd.concat([region_id, point, metadata, region_metadata, region_notes, detection_settings])
                df = df.append(row, ignore_index=True)
        # Reorder columns
        return df[row.keys()]

    def to_csv(self, save_path=None, **kwargs):
        # Parse EVR file if it hasn't already been done
        if not self.output_data:
            self.parse_file(**kwargs)
        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext='.csv')
        # Export to csv
        self.to_dataframe().to_csv(save_path, index=False)
        self._output_file.append(save_path)

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

        self._raw_range = ed.range.isel(frequency=0, ping_time=0).load()

        self.max_depth = self._raw_range.max().values
        self.min_depth = self._raw_range.min().values

        ed.close()
        if remove:
            os.remove(tmp_c.output_file)

    def swap_range_edge(self, y):
        if float(y) == 9999.99 and self.max_depth is not None:
            return self.max_depth
        elif float(y) == -9999.99 and self.min_depth is not None:
            return self.min_depth
        else:
            return float(y)

    def convert_output(self, convert_time=True, convert_range_edges=True):
        for region in self.output_data['regions'].values():
            if convert_time:
                region['metadata']['bounding_rectangle_left_x'] =\
                    parse_time(region['metadata']['bounding_rectangle_left_x'])
                region['metadata']['bounding_rectangle_right_x'] =\
                    parse_time(region['metadata']['bounding_rectangle_left_x'])
            if convert_range_edges:
                region['metadata']['bounding_rectangle_top_y'] =\
                    self.swap_range_edge(region['metadata']['bounding_rectangle_top_y'])
                region['metadata']['bounding_rectangle_bottom_y'] =\
                    self.swap_range_edge(region['metadata']['bounding_rectangle_bottom_y'])
            region['points'] = self.convert_points(region['points'], convert_time, convert_range_edges)

    def convert_points(self, points, convert_time=True, convert_range_edges=True, offset=0, unix=False):
        def convert_single(point):
            if convert_time:
                point[0] = parse_time(point[0], unix=unix)
            if convert_range_edges:
                point[1] = self.swap_range_edge(point[1]) + offset

        singular = True if not isinstance(points[0], list) else False
        if singular:
            points = [points]
        else:
            points = copy.deepcopy(points)

        if isinstance(points, dict):
            for point in points.values():
                convert_single(point)
        else:
            for point in points:
                convert_single(point)
        if singular:
            points = points[0]
        return points
