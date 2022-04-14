import os

import numpy as np
import pandas as pd

from .ev_parser import EvParserBase
from .utils import parse_time


class Regions2DParser(EvParserBase):
    """Class for parsing EV 2D region (EVR) files.
    Using this class directly is not recommended; use Regions2D instead.
    """

    def __init__(self, input_file=None):
        super().__init__(input_file, "EVR")

    def _parse(self, fid):
        """Reads an open file and returns the file metadata and region information"""

        def _region_metadata_to_dict(line):
            """Assigns a name to each value in the metadata line for each region"""
            top = float(line[9])
            bottom = float(line[12])
            bound_calculated = int(line[6])
            if bound_calculated:
                left = parse_time(f"{line[7]} {line[8]}", unix=False)
                right = parse_time(f"{line[10]} {line[11]}", unix=False)
            else:
                left = f"D{line[7]} {line[8]}"
                right = f"D{line[10]} {line[11]}"

            return {
                "region_id": int(line[2]),
                "region_structure_version": line[0],  # 13 currently
                "region_point_count": line[1],  # Number of points in the region
                "region_selected": line[3],  # Always 0
                "region_creation_type": line[4],  # How the region was created
                "dummy": line[5],  # Always -1
                "region_bbox_calculated": bound_calculated,  # 1 if next 4 fields valid.
                # O otherwise
                # Date encoded as CCYYMMDD and times in HHmmSSssss
                # Where CC=Century, YY=Year, MM=Month, DD=Day, HH=Hour,
                # mm=minute, SS=second, ssss=0.1 milliseconds
                "region_bbox_left": left,  # Time and date of bounding box left x
                "region_bbox_right": right,  # Time and date of bounding box right x
                "region_bbox_top": top,  # Top of bounding box
                "region_bbox_bottom": bottom,  # Bottom of bounding box
            }

        def _parse_points(line):
            """Takes a line with point information and creates a tuple (x, y) for each point"""
            points_x = parse_time(
                [f"{line[idx]} {line[idx + 1]}" for idx in range(0, len(line), 3)]
            ).values
            points_y = np.array(
                [float(line[idx + 2]) for idx in range(0, len(line), 3)]
            )
            return points_x, points_y

        # Read header containing metadata about the EVR file
        file_type, file_format_number, echoview_version = self.read_line(fid, True)
        file_metadata = pd.Series(
            {
                # TODO: add back the trailing ".evr" in filename for completeness
                "file_name": os.path.splitext(os.path.basename(self.input_file))[0],
                "file_type": file_type,
                "evr_file_format_number": file_format_number,
                "echoview_version": echoview_version,
            }
        )
        df = pd.DataFrame()
        row = {}
        n_regions = int(self.read_line(fid))
        # Loop over all regions in file
        for r in range(n_regions):
            # Unpack region data
            fid.readline()  # blank line separates each region

            # TODO: consider using fid.readlines() directly for code readability
            r_metadata = _region_metadata_to_dict(self.read_line(fid, True))
            # Add notes to region data
            n_note_lines = int(self.read_line(fid))
            r_notes = [self.read_line(fid) for line in range(n_note_lines)]
            # Add detection settings to region data
            n_detection_setting_lines = int(self.read_line(fid))
            r_detection_settings = [
                self.read_line(fid) for line in range(n_detection_setting_lines)
            ]
            # Add class to region data
            r_metadata["region_class"] = self.read_line(fid)
            # Add point x and y
            points_line = self.read_line(fid, True)
            # For type: 0=bad (No data), 1=analysis, 3=fishtracks, 4=bad (empty water)
            r_metadata["region_type"] = points_line.pop()
            r_points = _parse_points(points_line)
            r_metadata["region_name"] = self.read_line(fid)

            # Store region data into a GeoDataFrame
            row = pd.concat(
                [
                    file_metadata,
                    pd.Series(r_metadata)[r_metadata.keys()],
                    pd.Series({"time": r_points[0]}),
                    pd.Series({"depth": r_points[1]}),
                    pd.Series({"region_notes": r_notes}),
                    pd.Series({"region_detection_settings": r_detection_settings}),
                ]
            )
            df = df.append(row, ignore_index=True)

        return df[row.keys()].convert_dtypes()
