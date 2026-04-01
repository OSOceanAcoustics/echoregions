import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
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

    def to_csv(self, save_path: Union[str, Path], mode: str = "w", **kwaargs) -> None:
        """Save a Dataframe to a .csv file

        Parameters
        ----------
        save_path : Union[str, Path]
            Path to save the CSV file to.
        mode : str
            Write mode arg for to_csv. Defaults to 'w'.
        """
        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext=".csv")
        # Reorder columns and export to csv
        self.data.to_csv(save_path, mode=mode, **kwaargs)
        self.output_file.append(save_path)

    def to_evl(self, save_path: Union[str, Path], mode: str = "w") -> None:
        """Save a Dataframe to a .evl file

        Parameters
        ----------
        save_path : Union[str, Path]
            Path to save the `evl` file to.
        mode : str
            Write mode arg for IO open. Defaults to 'w'.
        """
        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext=".evl")

        # Grab header information
        echoview_version = (
            f"EVBD 3 {self.data.iloc[0]['echoview_version']}"
            if len(self.data) > 0
            else "EVBD 3 12.0.341.42620"
        )
        number_of_regions = str(len(self.data))

        with open(save_path, mode=mode) as f:
            # Write header to `.evl`
            f.write(echoview_version + "\n")
            f.write(number_of_regions + "\n")

            # Write each seafloor point to `.evl`
            for _, row in self.data.iterrows():
                f.write(
                    str(row["time"].strftime("%Y%m%d"))
                    + " "
                    + str(row["time"].strftime("%H%M%S%f"))[:-2]
                    + "  "
                    + str(row["depth"])
                    + " "
                    + str(row["status"])
                    + "\n"
                )
        f.close()

        # Append save_path
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
        if start_time is not None and not (isinstance(start_time, Timestamp)):
            raise TypeError(f"Start time is of type {type(start_time)}. \
                            It must be of of type Pandas Timestamp.")
        if end_time is not None and not (isinstance(end_time, Timestamp)):
            raise TypeError(f"End time is of type {type(end_time)}. \
                            It must be of of type Pandas Timestamp.")

        df = self.data
        if start_time is not None:
            df = df[df["time"] > start_time]
        if end_time is not None:
            df = df[df["time"] < end_time]

        if fill_between:
            plt.fill_between(df.time, df.depth, max_depth, **kwargs)
        else:
            plt.plot(df.time, df.depth, fmt, **kwargs)

    def _filter_seafloor(self, seafloor, start_date, end_date, operation):
        """
        Selects the values of the seafloor between two dates (non-inclusive).
        """
        after_start_date = seafloor["time"] > start_date
        before_end_date = seafloor["time"] < end_date
        between_two_dates = after_start_date & before_end_date
        filtered_seafloor = seafloor.loc[between_two_dates]
        if operation == "above_below":
            filtered_seafloor = filtered_seafloor.set_index("time")
        return filtered_seafloor

    def seafloor_mask(self, da_Sv: DataArray, operation: str = "regionmask", **kwargs):
        """
        Subsets a seafloor dataset to the range of an Sv dataset. Create a mask of
        the same shape as data found in the Echogram object:
        seafloor: 1, Otherwise: 0.

        Parameters
        ----------
        da_Sv : Xarray DataArray
            Matrix of coordinates (ping_time, depth) that contains Sv values.
        operation : str
            Whether to use regionmask or below/above logic to produce the seafloor mask.
        **kwargs : dict
            Keyword arguments to be passed to pandas.DataFrame.interpolate.

        Returns
        -------
        seafloor_mask : Xarray DataArray
            Matrix of coordinates (ping_time, depth) with values such that seafloor: 1,
            otherwise: 0.
        seafloor_points : pd.DataFrame
            DataFrame containing depth and time.

        Notes
        -----
        If operation == 'regionmask':
            We create 4 additional seafloor points that further describe the boundary that we want
            regionmask to mask:

            1) Point at the seafloor leftmost corner. The depth of this point is based on the
            maximum of the EVL seafloor point depth and the Echogram depth, plus an additional
            1.0 float offset.
            2) Point at the leftmost edge of the Echogram where depth is based on the closest
            EVL seafloor point to this leftmost edge. One can think of this as a left facing
            extension of the leftmost EVL seafloor point until the left edge of the Echogram.
            3) Point at the rightmost edge of the Echogram where depth is based on the closest
            EVL seafloor point to this rightmost edge. One can think of this as a right facing
            extension of the rightmost EVL seafloor point until the right edge of the Echogram.
            4) Point at the seafloor rightmost corner. The depth of this point is the same as 1.

            The points are there to ensure that regionmask captures the appropriate area during masking.
            The offset in Point 1 and Point 4 is here to make sure that the line connecting the seafloor-most
            points are clear of any other points. This would be a problem in the case where there is a point
            in the middle that matches the maximum of the Sv and EVL point depth. This would lead to
            regionmask creating possibly 2+ regions, which is behavior that could lead to different outputs.
            The offset ensures that regionmask always creates just 1 region.

            In the dataframe passed into regionmask, the following points are connected in the
            following order: [1, 2, seafloor points, 3, 4, 1].

            For further information on how regionmask deals with edges:
            https://regionmask.readthedocs.io/en/stable/notebooks/method.html
        If operation == 'above_below':
            Prior to creating the mask, this method performs interpolation on the seafloor data
            points found in the lines.data dataframe.
            The nearest interpolation method from Pandas has a problem when points are far
            from each other.
        """

        if not isinstance(da_Sv, DataArray):
            raise TypeError(
                f"Input da_Sv must be of type DataArray. da_Sv was instead of type {type(da_Sv)}"
            )

        if operation not in ["regionmask", "above_below"]:
            raise ValueError(
                "Argument ```option``` must be either 'regionmask' or 'above_below'. "
                f"Cannot be {operation}."
            )

        # Drop channel if it exists
        if "channel" in da_Sv.dims:
            da_Sv = da_Sv.isel(channel=0).drop_vars("channel")

        lines_df = self.data

        # new index
        echogram_ping_time = list(da_Sv.ping_time.data)

        # filter seafloor within start and end time
        start_time = da_Sv.ping_time.data.min()
        end_time = da_Sv.ping_time.data.max()
        filtered_seafloor = self._filter_seafloor(lines_df, start_time, end_time, operation)
        filtered_seafloor = filtered_seafloor[~filtered_seafloor.index.duplicated()]

        if len(filtered_seafloor) > 0:
            if operation == "regionmask":
                # Filter columns and sort rows
                seafloor_points = filtered_seafloor.copy()[["time", "depth"]].sort_values(by="time")

                # Calculate left and right most seafloor point depth values
                seafloor_points_min_time_depth = seafloor_points.loc[
                    seafloor_points["time"].idxmin(), "depth"
                ]
                seafloor_points_max_time_depth = seafloor_points.loc[
                    seafloor_points["time"].idxmax(), "depth"
                ]

                # Calculate maximum depth between seafloor points and Sv and add additional offset
                maximum_depth_plus_offset = (
                    max([da_Sv["depth"].max().data, seafloor_points["depth"].max()]) + 1.0
                )

                # Set new left corner rows:
                # We add a short time offset here to ensure appropriate left side of mask
                # inclusion behavior; otherwise, regionmask will not mask the leftmost edge
                # of the Echogram even if the seafloor annotation indicates that it should be
                # masked.
                # For more edge information: https://regionmask.readthedocs.io/en/stable/notebooks/method.html#edge-behavior #noqa
                # TODO: Figure out a cleaner and less arbitrary way of ensuring correct left edge
                # behavior.
                time_offset = pd.Timedelta(seconds=1)
                left_side_new_rows = pd.DataFrame(
                    {
                        "time": [
                            pd.to_datetime(da_Sv["ping_time"].min().data) - time_offset,
                            pd.to_datetime(da_Sv["ping_time"].min().data) - time_offset,
                        ],
                        "depth": [maximum_depth_plus_offset, seafloor_points_min_time_depth],
                    }
                )

                # Set new right corner rows
                right_side_new_rows = pd.DataFrame(
                    {
                        "time": [
                            pd.to_datetime(da_Sv["ping_time"].max().data),
                            pd.to_datetime(da_Sv["ping_time"].max().data),
                        ],
                        "depth": [seafloor_points_max_time_depth, maximum_depth_plus_offset],
                    }
                )

                # Concat new corner rows to seafloor points
                seafloor_points = pd.concat(
                    [left_side_new_rows, seafloor_points, right_side_new_rows]
                ).reset_index(drop=True)

                # Convert region time to integer timestamp.
                seafloor_points_with_int_timestamp = seafloor_points.copy()
                seafloor_points_with_int_timestamp["time"] = matplotlib.dates.date2num(
                    seafloor_points_with_int_timestamp["time"]
                )

                # Convert ping_time to unix_time since the masking does not work on datetime objects.
                da_Sv = da_Sv.assign_coords(
                    unix_time=(
                        "ping_time",
                        matplotlib.dates.date2num(da_Sv.coords["ping_time"].values),
                    )
                )

                regionmask_region = regionmask.Regions(
                    outlines=[np.array(seafloor_points_with_int_timestamp)],
                    overlap=False,
                )

                if da_Sv.chunksizes:
                    # Define a helper function to operate on individual blocks
                    def _seafloor_mask_block(da_Sv_block, wrap_lon, regionmask_regions):
                        # Grab time and depth blocks
                        unix_time_block = da_Sv_block["unix_time"]
                        depth_block = da_Sv_block["depth"]

                        # Set the filter to ignore the specific warnings
                        # No grid point warning will show up a lot with smaller chunks and
                        warnings.filterwarnings(
                            "ignore", message="No gridpoint belongs to any region"
                        )
                        # TODO Write issue in regionmask repo to convince them not to remove method as an argument
                        warnings.filterwarnings(
                            "ignore",
                            message="The ``method`` argument is internal and  will be removed in the future",
                            category=FutureWarning,
                        )
                        mask_block_da = xr.where(
                            np.isnan(
                                regionmask_regions.mask(
                                    unix_time_block,
                                    depth_block,
                                    wrap_lon=wrap_lon,
                                    method="shapely",
                                ),
                            ),
                            0,
                            1,
                        )
                        return mask_block_da

                    # Apply _mask_block over the blocks of the input array to create 0/1 seafloor mask
                    seafloor_mask_da = xr.map_blocks(
                        _seafloor_mask_block,
                        da_Sv,
                        kwargs={
                            "wrap_lon": False,
                            "regionmask_regions": regionmask_region,
                        },
                    )
                else:
                    # Create 0/1 seafloor mask
                    seafloor_mask_da = xr.where(
                        np.isnan(
                            regionmask_region.mask(
                                da_Sv["unix_time"],
                                da_Sv["depth"],
                                wrap_lon=False,
                            )
                        ),
                        0,
                        1,
                    )

                # Remove all coords other than region_id, depth, ping_time
                seafloor_mask_da = seafloor_mask_da.drop_vars(
                    seafloor_mask_da.coords._names.difference({"depth", "ping_time"})
                )
            else:
                # create joint index
                joint_index = list(
                    set(list(pd.DataFrame(echogram_ping_time)[0]) + list(filtered_seafloor.index))
                )

                # Interpolate on the echogram coordinates. Note that some interpolation kwaargs
                # will result in some interpolation NaN values. The ffill and bfill lines
                # are there to fill in these NaN values.
                # TODO There exists a problem where when we use .loc prior to reindexing
                # we are hit with a key not found error.
                seafloor_points = (
                    filtered_seafloor[["depth"]]
                    .reindex(joint_index)
                    .interpolate(**kwargs)
                    .loc[echogram_ping_time]
                    .ffill()
                    .bfill()
                )

                # convert to data array for seafloor
                seafloor_da = seafloor_points["depth"].to_xarray()
                seafloor_da = seafloor_da.rename({"time": "ping_time"})

                # create a data array of depths
                depth_da = da_Sv["depth"] + xr.zeros_like(da_Sv)

                # create a mask for the seafloor:
                # seafloor: True, otherwise: False
                seafloor_mask_da = depth_da >= seafloor_da

                # seafloor: True becomes 1, False becomes 0
                seafloor_mask_da = seafloor_mask_da.where(True, 1, 0)

                # Reset seafloor_points index so that time index becomes time column
                seafloor_points = seafloor_points.reset_index()

            # Set seafloor points to be pandas datetime
            seafloor_points["time"] = pd.to_datetime(seafloor_points["time"])

        else:
            # Set everything to 0
            seafloor_mask_da = xr.zeros_like(da_Sv)

            # Set seafloor points to empty DataFrame with time and depth columns
            seafloor_points = pd.DataFrame(columns=["depth", "time"])

        return seafloor_mask_da, seafloor_points
