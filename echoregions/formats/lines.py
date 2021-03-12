from ..convert.evl_parser import LineParser
import numpy as np


class Lines():
    def __init__(self, input_file=None, parse=True, convert_time=False,
                 replace_nan_range_value=None, offset=0):
        self._parser = LineParser(input_file)
        self._plotter = None
        self._masker = None
        self._points = None

        if parse:
            self.parse_file(
                convert_time=convert_time,
                replace_nan_range_value=replace_nan_range_value,
                offset=offset
            )

    def __iter__(self):
        """Get points as an iterable. Allows looping over Lines object."""
        return iter(self.points)

    def __getitem__(self, idx):
        """Indexing lines object will return the point at that index"""
        return self.points[idx]

    @property
    def output_data(self):
        """Dictionary containing the data parsed from the EVL file"""
        return self._parser.output_data

    @property
    def output_file(self):
        """Path(s) to the list of files saved.
        String if a single file. LIst of strings if multiple.
        """
        return self._parser.output_file

    @property
    def input_file(self):
        """String path to the EVL file"""
        return self._parser.input_file

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
        self._parser.parse_file(
            convert_time=convert_time,
            replace_nan_range_value=replace_nan_range_value,
            offset=offset
        )

    def to_csv(self, save_path=None, **kwargs):
        """Convert an EVL file to a CSV

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        kwargs : keyword arguments
            Additional arguments passed to `Lines.parse_file`
        """
        self._parser.to_csv(save_path=save_path, **kwargs)

    def to_dataframe(self, **kwargs):
        """Organize EVL data into a Pandas DataFrame.
        See `Lines.to_csv` for arguments
        """
        return self._parser.to_dataframe(**kwargs)

    def to_json(self, save_path=None, pretty=False, **kwargs):
        """Convert EVL to JSON

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        pretty : bool, default False
            Output more human readable JSON
        kwargs : keyword arguments
            Additional arguments passed to `Lines.parse_file`
        """
        self._parser.to_json(save_path=save_path, pretty=pretty, **kwargs)

    def convert_points(self, points, convert_time=True, replace_nan_range_value=None, offset=0):
        """Convert x and y values of points from the EV format.
        Modifies points in-place.

        Parameters
        ----------
        points : list or dict
            List containing EVL points or a single point in dict form
        convert_time : bool, default True
            Convert EV time to datetime64
        replace_nan_range_value : float, default ``None``
            Value in meters to replace -10000.990000 ranges with.
            Don't replace if ``None``.
        offset : float, default 0
            Depth offset in meters.

        Returns
        -------
        list or dict
            Converted points with type depending on input
        """
        return self._parser.convert_points(
            points=points,
            convert_time=convert_time,
            replace_nan_range_value=replace_nan_range_value,
            offset=offset
        )

    def _init_plotter(self):
        """Initialize the object used to plot lines"""
        if self._plotter is None:
            if not self.output_data:
                raise ValueError("Input file has not been parsed; call `parse_file` to parse.")
            from ..plot.line_plot import LinesPlotter
            self._plotter = LinesPlotter(self)

    def plot(
        self,
        calibrated_dataset=None,
        min_ping_time=None,
        max_ping_time=None,
        fill_between=True,
        max_depth=0,
        alpha=0.5,
        **kwargs
    ):
        """
        Plot the points in the EVL file.

        Parameters
        ----------
        calibrated_dataset : Dataset, default ``None``
            Dataset containing range and ping_time that sets the bounds for the points plotted.
        min_ping_time : datetime64, default ``None``
            Lower ping_time bound.
        max_ping_time : datetime64, default ``None``
            Upper ping_time bound.
        fill_between : bool, default True
            Use matplotlib `fill_between` to plot the line.
            The area between the EVL points and `max_depth` will be filled in.
        max_depth : float, default 0
            The `fill_between` function will color in the area betwen the points and
            this depth value given in meters.
        alpha : float, default 0.5
            Opacity of the plot
        kwargs : keyword arguments
            Additional arguments passed to matplotlib `plot` or `fill_between`.
            Useful arguments include `color`, `lw`, and `marker`.
        """
        self._init_plotter()
        self._plotter.plot(
            calibrated_dataset=calibrated_dataset,
            min_ping_time=min_ping_time,
            max_ping_time=max_ping_time,
            fill_between=fill_between,
            max_depth=max_depth,
            alpha=alpha,
            **kwargs
        )
