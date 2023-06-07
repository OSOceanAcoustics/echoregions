from typing import Dict, Iterable, List, Union

from pandas import DataFrame, Series, Timestamp

from ..convert.evl_parser import LineParser
from . import Geometry


class Lines(Geometry):
    def __init__(
        self,
        input_file: str = None,
        parse: bool = True,
        nan_depth_value: float = None,
        offset: float = 0,
    ):
        self._parser = LineParser(input_file)
        self._plotter = None
        self._masker = None

        self.nan_depth_value = nan_depth_value
        self.offset = offset

        self.data = None
        if parse:
            self.parse_file()

    def __iter__(self) -> Iterable:
        """Get points as an iterable. Allows looping over Lines object."""
        return iter(self.points)

    def __getitem__(self, idx: int) -> Union[Dict, List]:
        """Indexing lines object will return the point at that index"""
        return self.points[idx]

    @property
    def output_file(self) -> Union[List[str], str]:
        """Path(s) to the list of files saved.
        String if a single file. List of strings if multiple.
        """
        return self._parser.output_file

    @property
    def input_file(self) -> str:
        """String path to the EVL file"""
        return self._parser.input_file

    @property
    def nan_depth_value(self) -> Union[int, float]:
        return self._nan_depth_value

    @nan_depth_value.setter
    def nan_depth_value(self, val: Union[int, float]) -> None:
        """Set the depth in meters that the -10000.99 depth value will be set to"""
        self._nan_depth_value = float(val) if val is not None else None

    @property
    def offset(self) -> Union[int, float]:
        """Get the depth offset to apply to y values"""
        return self._offset

    @offset.setter
    def offset(self, val: Union[int, float]):
        """Set the depth offset to apply to y values"""
        self._offset = float(val)

    def parse_file(self) -> None:
        """
        Parse the EVL file into `Lines.data`.
        """
        self.data = self._parser.parse_file()
        self.adjust_offset()
        self.replace_nan_depth()

    def replace_nan_depth(self, inplace: bool = False) -> DataFrame:
        """Replace -10000.99 depth values with user-specified nan_depth_value

        Parameters
        ----------
        inplace : bool
            Modify the current `data` inplace

        Returns
        -------
        DataFrame with depth edges replaced by Lines.nan_depth_value
        """

        def replace_depth(row: Series) -> Series:
            def swap_val(val: Union[int, float]) -> Union[int, float]:
                if val == -10000.99:
                    return self.nan_depth_value
                else:
                    return val

            row.at["depth"] = swap_val(row["depth"])
            return row

        if self.nan_depth_value is None:
            return

        regions = self.data if inplace else self.data.copy()
        regions.loc[:] = regions.apply(replace_depth, axis=1)
        return regions

    def to_csv(self, save_path: str = None, **kwargs) -> None:
        """Convert an EVL file to a CSV

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        kwargs : keyword arguments
            Additional arguments passed to `Lines.parse_file`
        """
        if self.data is None:
            self.parse_file(**kwargs)
        self._parser.to_csv(self.data, save_path=save_path, **kwargs)

    def to_json(self, save_path: str = None, pretty: bool = False, **kwargs) -> None:
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

    def _init_plotter(self) -> None:
        """Initialize the object used to plot lines"""
        if self._plotter is None:
            if self.data is None:
                raise ValueError(
                    "Input file has not been parsed; call `parse_file` to parse."
                )
            from ..plot.line_plot import LinesPlotter

            self._plotter = LinesPlotter(self)

    def plot(
        self,
        fmt: str = "",
        start_time: Timestamp = None,
        end_time: Timestamp = None,
        fill_between: bool = False,
        max_depth: Union[int, float] = 0,
        **kwargs,
    ) -> None:
        """
        Plot the points in the EVL file.

        Parameters
        ----------
        fmt : str, optional
            A format string such as 'bo' for blue circles.
            See matplotlib documentation for more information.
        start_time : datetime64, default ``None``
            Lower time bound.
        end_time : datetime64, default ``None``
            Upper time bound.
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
        if not (isinstance(start_time, Timestamp) and isinstance(end_time, Timestamp)):
            raise TypeError(
                f"start and end times are of type {type(start_time)} and {type(end_time)}. \
                            They must be of of type Pandas Timestamp."
            )
        self._init_plotter()
        self._plotter.plot(
            fmt=fmt,
            start_time=start_time,
            end_time=end_time,
            fill_between=fill_between,
            max_depth=max_depth,
            **kwargs,
        )
