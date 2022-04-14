import os

import pandas as pd

from .ev_parser import EvParserBase
from .utils import parse_time


class LineParser(EvParserBase):
    """Class for parsing EV lines (EVL) files"""

    def __init__(self, input_file=None):
        super().__init__(input_file, "EVL")

        self._data_dict = {}

    def _parse(self, fid):
        # Read header containing metadata about the EVL file
        file_type, file_format_number, ev_version = self.read_line(fid, True)
        file_metadata = {
            # TODO: add back the trailing ".evl" in filename for completeness
            "file_name": os.path.splitext(os.path.basename(self.input_file))[0],
            "file_type": file_type,
            "evl_file_format_version": file_format_number,
            "echoview_version": ev_version,
        }
        # TODO: below is better implemented as reading to EOF
        # and check if the total number of lines read equals to n_points;
        # if the number of lines don't match, return error
        points = []
        n_points = int(self.read_line(fid))
        for i in range(n_points):
            date, time, depth, status = self.read_line(fid, split=True)
            points.append(
                {
                    "time": f"{date} {time}",  # Format: CCYYMMDD HHmmSSssss
                    "depth": float(depth),  # Depth [m]
                    "status": status,  # 0 = none, 1 = unverified, 2 = bad, 3 = good
                }
            )
        # Store JSON serializable data
        self._data_dict = {"metadata": file_metadata, "points": points}

        # Put data into a DataFrame
        df = pd.DataFrame(self._data_dict["points"])
        # Save file metadata for each point
        df = df.assign(**self._data_dict["metadata"])
        df.loc[:, "time"] = df.loc[:, "time"].apply(parse_time)
        order = list(self._data_dict["metadata"].keys()) + list(
            self._data_dict["points"][0].keys()
        )
        return df[order]
