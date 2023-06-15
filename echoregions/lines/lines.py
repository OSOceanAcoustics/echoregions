from typing import Dict, Iterable, List, Union
from pandas import DataFrame, Series, Timestamp
import json
import matplotlib.pyplot as plt

from ..utils.io import validate_path
from .lines_parser import parse_line_file

class Lines():
    def __init__(
        self,
        input_file: str,
        nan_depth_value: float = None
    ):
        self.depth = (
            None  # Single array that can be used to obtain min_depth and max_depth
        )
        self._min_depth = (
            None  # Set to replace -9999.99 depth values which are EVR min range
        )
        self._max_depth = (
            None  # Set to replace 9999.99 depth values which are EVR max range
        )
        self._nan_depth_value = (
            None  # Set to replace -10000.99 depth values with (EVL only)
        )

        self.input_file = input_file
        self._data = parse_line_file(input_file)
        self.output_file = []

        self.nan_depth_value = nan_depth_value

    def __iter__(self) -> Iterable:
        """Get points as an iterable. Allows looping over Lines object."""
        return iter(self.points)

    def __getitem__(self, idx: int) -> Union[Dict, List]:
        """Indexing lines object will return the point at that index"""
        return self.points[idx]

    @property
    def nan_depth_value(self) -> Union[int, float]:
        return self._nan_depth_value
    
    @property
    def data(self) -> DataFrame:
        return self._data

    @nan_depth_value.setter
    def nan_depth_value(self, val: Union[int, float]) -> None:
        """Set the depth in meters that the -10000.99 depth value will be set to"""
        self._nan_depth_value = float(val) if val is not None else None

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

        regions = self._data if inplace else self._data.copy()
        regions.loc[:] = regions.apply(replace_depth, axis=1)
        return regions

    def to_csv(self, save_path: bool = None) -> None:
        """Save a Dataframe to a .csv file

        Parameters
        ----------
        save_path : str
            path to save the CSV file to
        """
        if not isinstance(self._data, DataFrame):
            raise TypeError(
                f"Invalid ds Type: {type(self._data)}. Must be of type DataFrame."
            )

        # Check if the save directory is safe
        save_path = validate_path(
            save_path=save_path, input_file=self.input_file, ext=".csv"
        )
        # Reorder columns and export to csv
        self._data.to_csv(save_path, index=False)
        self.output_file.append(save_path)

    def to_json(self, save_path: str = None, pretty: bool = True, **kwargs) -> None:
        # TODO Currently only EVL files can be exported to JSON
        """Convert supported formats to .json file.

        Parameters
        ----------
        save_path : str
            path to save the JSON file to
        pretty : bool, default True
            Output more human readable JSON
        kwargs
            keyword arguments passed into `parse_file`
        """
        # Check if the save directory is safe
        save_path = validate_path(
            save_path=save_path, input_file=self.input_file, ext=".json"
        )
        indent = 4 if pretty else None

        # Save the entire parsed EVR dictionary as a JSON file
        with open(save_path, "w") as f:
            f.write(json.dumps(self._data.to_json(), indent=indent))
        self.output_file.append(save_path)

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

        df = self._data
        if start_time is not None:
            df = df[df["time"] > start_time]
        if end_time is not None:
            df = df[df["time"] < end_time]

        if fill_between:
            plt.fill_between(df.time, df.depth, max_depth, **kwargs)
        else:
            plt.plot(df.time, df.depth, fmt, **kwargs)
