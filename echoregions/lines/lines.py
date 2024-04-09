import json
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame, Timestamp
from xarray import DataArray

from ..utils.io import validate_path
from .lines_parser import parse_evl, parse_lines_df

ECHOVIEW_NAN_DEPTH_VALUE = -10000.99


class Lines:
    """
    Class that contains and performs operations with Depth/Lines data from Echoview EVL files.
    """

    def __init__(
        self,
        input_file: Union[str, pd.DataFrame],
        nan_depth_value: float = None,
        input_type: str = "EVL",
    ):
        self.input_file = input_file
        if input_type == "EVL":
            self.data = parse_evl(input_file)
        elif input_type == "CSV":
            self.data = parse_lines_df(input_file)
        else:
            raise ValueError(f"Lines input_type must be EVL or CSV. Got {input_type} instead.")
        self.output_file = []

        self._nan_depth_value = None  # Set to replace -10000.99 depth values with (EVL only)
        self._nan_depth_value = nan_depth_value

    def __iter__(self) -> Iterable:
        """Get points as an iterable. Allows looping over Lines object."""
        return iter(self.points)

    def __getitem__(self, idx: int) -> Union[Dict, List]:
        """Indexing lines object will return the point at that index"""
        return self.points[idx]

    def replace_nan_depth(self, inplace: bool = False) -> Union[DataFrame, None]:
        """Replace -10000.99 depth values with user-specified _nan_depth_value

        Parameters
        ----------
        inplace : bool
            Modify the current `data` inplace

        Returns
        -------
        DataFrame with depth edges replaced by Lines._nan_depth_value
        """
        if self._nan_depth_value is None:
            return

        regions = self.data if inplace else self.data.copy()
        regions["depth"] = regions["depth"].apply(
            lambda x: self._nan_depth_value if x == ECHOVIEW_NAN_DEPTH_VALUE else x
        )
        if not inplace:
            return regions

    def to_csv(self, save_path: bool = None, mode="w", **kwaargs) -> None:
        """Save a Dataframe to a .csv file

        Parameters
        ----------
        save_path : str
            Path to save the CSV file to.
        mode : str
            Write mode arg for to_csv.
        """
        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext=".csv")
        # Reorder columns and export to csv
        self.data.to_csv(save_path, mode=mode, **kwaargs)
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
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext=".json")
        indent = 4 if pretty else None

        # Save the entire parsed EVR dictionary as a JSON file
        with open(save_path, "w") as f:
            f.write(json.dumps(self.data.to_json(), indent=indent))
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
        if not (isinstance(start_time, Timestamp)):
            raise TypeError(
                f"start time is of type {type(start_time)} \
                            It must be of of type Pandas Timestamp."
            )
        if not (isinstance(end_time, Timestamp)):
            raise TypeError(
                f"end time is of type {type(end_time)} \
                            It must be of of type Pandas Timestamp."
            )

        df = self.data
        if start_time is not None:
            df = df[df["time"] > start_time]
        if end_time is not None:
            df = df[df["time"] < end_time]

        if fill_between:
            plt.fill_between(df.time, df.depth, max_depth, **kwargs)
        else:
            plt.plot(df.time, df.depth, fmt, **kwargs)

    def mask(self, da_Sv: DataArray, **kwargs):
        """
        Subsets a bottom dataset to the range of an Sv dataset. Create a mask of
        the same shape as data found in the Echogram object:
        Bottom: True, Otherwise: False.

        Parameters
        ----------
        da_Sv : Xarray DataArray
            Matrix of coordinates (ping_time, depth) that contains Sv values.
        **kwargs : dict
            Keyword arguments to be passed to pandas.DataFrame.interpolate.

        Returns
        -------
        bottom_mask : Xarray DataArray
            Matrix of coordinates (ping_time, depth) with values such that bottom: False,
            otherwise: True
        bottom_points : pd.DataFrame
            DataFrame containing depth and time.

        Notes
        -----
        Prior to creating the mask, this method performs interpolation on the bottom data
        points found in the lines.data dataframe.
        The nearest interpolation method from Pandas has a problem when points are far
        from each other.
        """

        if not isinstance(da_Sv, DataArray):
            raise TypeError(
                f"Input da_Sv must be of type DataArray. da_Sv was instead of type {type(da_Sv)}"
            )

        def filter_bottom(bottom, start_date, end_date):
            """
            Selects the values of the bottom between two dates.
            """
            after_start_date = bottom["time"] > start_date
            before_end_date = bottom["time"] < end_date
            between_two_dates = after_start_date & before_end_date
            filtered_bottom = bottom.loc[between_two_dates].set_index("time")
            return filtered_bottom

        lines_df = self.data

        # new index
        echogram_ping_time = list(da_Sv.ping_time.data)

        # filter bottom within start and end time
        start_time = da_Sv.ping_time.data.min()
        end_time = da_Sv.ping_time.data.max()
        filtered_bottom = filter_bottom(lines_df, start_time, end_time)
        filtered_bottom = filtered_bottom[~filtered_bottom.index.duplicated()]

        if len(filtered_bottom) > 0:
            # create joint index
            joint_index = list(
                set(list(pd.DataFrame(echogram_ping_time)[0]) + list(filtered_bottom.index))
            )

            # Interpolate on the echogram coordinates. Note that some interpolation kwaargs
            # will result in some interpolation NaN values. The ffill and bfill lines
            # are there to fill in these NaN values.
            # TODO There exists a problem where when we use .loc prior to reindexing
            # we are hit with a key not found error.
            bottom_points = (
                filtered_bottom[["depth"]]
                .reindex(joint_index)
                .interpolate(**kwargs)
                .loc[echogram_ping_time]
                .ffill()
                .bfill()
            )

            # convert to data array for bottom
            bottom_da = bottom_points["depth"].to_xarray()
            bottom_da = bottom_da.rename({"time": "ping_time"})

            # create a data array of depths
            depth_da = da_Sv["depth"] + xr.zeros_like(da_Sv)

            # create a mask for the bottom:
            # bottom: True, otherwise: False
            bottom_mask = depth_da >= bottom_da

            # Reset bottom_points index so that time index becomes time column
            bottom_points = bottom_points.reset_index()

        else:
            # Set everything to False
            bottom_mask = xr.full_like(da_Sv, False)

            # Set bottom points to empty DataFrame with time and depth columns
            bottom_points = pd.DataFrame(columns=["depth", "time"])

        # Bottom: True becomes 1, False becomes 0
        bottom_mask = bottom_mask.where(True, 1, 0)

        return bottom_mask, bottom_points
