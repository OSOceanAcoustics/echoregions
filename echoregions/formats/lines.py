from ..convert import LineParser, utils
from ..plot.line_plot import LinesPlotter
from pathlib import Path
import numpy as np


class Lines():
    def __init__(self, input_file=None, parse=True, convert_time=False,
                 replace_nan_range_value=None, offset=0):
        self._plotter = None
        self._masker = None
        self._points = None

        self.parser = LineParser(input_file)
        if parse:
            self.parse_file(convert_time=convert_time, replace_nan_range_value=replace_nan_range_value, offset=offset)

    def __iter__(self):
        """Get points as an iterable. Allows looping over Lines object."""
        return iter(self.points)

    def __getitem__(self, idx):
        """Indexing lines object will return the point at that index"""
        return self.points[idx]

    @property
    def output_data(self):
        """Dictionary containing the data parsed from the EVL file"""
        return self.parser.output_data

    @property
    def output_file(self):
        """Path(s) to the list of files saved.
        String if a single file. LIst of strings if multiple.
        """
        return self.parser.output_file

    @property
    def input_file(self):
        """String path to the EVL file"""
        return self.parser.input_file

    @property
    def points(self):
        """List of points in the form `(time, depth)`"""
        if self._points is None:
            if not self.output_data:
                raise ValueError("Input file has not been parsed; call `parse_file` to parse.")
            else:
                self._points = np.array([(point['x'], point['y']) for point in self.output_data['points']])
        return self._points

    def parse_file(self, convert_time=False, replace_nan_range_value=None, offset=0):
        """Parse the EVL file into `Lines.output_data`

        Parameters
        ----------
        convert_time : bool, default False
            Convert EV time to datetime64.
        replace_nan_range_value : float, default ``None``
            Depth in meters to replace -10000.990000 ranges with.
            Don't replace if ``None``.
        offset : float, default 0
            depth offset in meters.
        """
        self.parser.parse_file(
            convert_time=convert_time,
            replace_nan_range_value=replace_nan_range_value,
            offset=offset
        )

    def to_csv(self, save_path=None, convert_time=False, convert_range_edges=True):
        """Convert an EVL file to a CSV

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        convert_time : bool, default False
            Convert times in the EV datetime format to numpy datetime64.
        convert_range_edges : bool, default True
            Convert -9999.99 and -9999.99 depth edges to real values for EVL files.
            Set the values by assigning range values to `min_depth` and `max_depth`
            or by passing a file into `set_range_edge_from_raw`.
        """
        self.parser.to_csv(save_path=save_path, convert_time=convert_time, convert_range_edges=convert_range_edges)

    def to_dataframe(self, convert_time=False, convert_range_edges=True):
        """Organize EVL data into a Pandas DataFrame.
        See `Lines.to_csv` for arguments
        """
        return self.parser.to_dataframe(convert_time=convert_time, convert_range_edges=convert_range_edges)

    def to_json(self, save_path=None, convert_range_edges=True, pretty=False):
        """Convert EVL to JSON

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        convert_range_edges : bool, default True
            Convert -9999.99 and -9999.99 depth edges to real values for EVL files.
            Set the values by assigning range values to `min_depth` and `max_depth`
            or by passing a file into `set_range_edge_from_raw`.
        pretty : bool, default False
            Output more human readable JSON
        """
        self.parser.to_json(save_path=save_path, pretty=pretty, convert_range_edges=convert_range_edges)
