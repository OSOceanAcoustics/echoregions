from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import xarray as xr
from pandas import DataFrame, Series, Timestamp
from xarray import DataArray

from ..utils.io import validate_path
from ..utils.time import parse_simrad_fname_time
from .regions2d_parser import parse_regions_file


class Regions2D:
    """
    Class that contains and performs operations with Regions2D data from Echoview EVR files.
    """

    def __init__(
        self,
        input_file: str,
        min_depth: Union[int, float] = None,
        max_depth: Union[int, float] = None,
    ):
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
        self.data = parse_regions_file(input_file)
        self.output_file = []

        self.max_depth = max_depth
        self.min_depth = min_depth

    def __iter__(self) -> Iterable:
        return self.data.iterrows()

    def __getitem__(self, val: int) -> Series:
        return self.data.iloc[val]

    def to_csv(self, save_path: bool = None) -> None:
        """Save a Dataframe to a .csv file

        Parameters
        ----------
        save_path : str
            path to save the CSV file to
        """
        # Check if the save directory is safe
        save_path = validate_path(
            save_path=save_path, input_file=self.input_file, ext=".csv"
        )
        # Reorder columns and export to csv
        self.data.to_csv(save_path, index=False)
        self.output_file.append(save_path)

    def to_json(self, save_path: str = None) -> None:
        # TODO: Implement this function
        """Convert EVR to a JSON file. Currently Deprecated.

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        pretty : bool, default False
            Output more human readable JSON
        """

    def select_region(
        self,
        region_id: Union[float, int, str, List[Union[float, int, str]]] = None,
        time_range: List[Timestamp] = None,
        depth_range: List[Union[float, int]] = None,
        copy=True,
    ) -> DataFrame:
        """Selects a subset of this Region2D object's dataframe.

        Parameters
        ----------
        region_id : float, int, str, list, ``None``
            A region id provided as a number, a string, or list of these.
        time_range: List of 2 Pandas Timestamps.
            Datetime range for expected output of subselected DataFrame. 1st
            index value must be later than 0th index value.
        depth_range: List of 2 floats.
            Depth range for expected output of subselected DataFrame. 1st
            index value must be larger than 0th index value.
        copy : bool
            Return a copy of the `data` DataFrame
        Returns
        -------
        DataFrame
            There is a row for each region id provided by the ``region_id`` parameter,
            and each row has time and depth within or on the boundaries passed
            in by the ``time_range`` and ``depth_range`` values.
        """
        # Make copy of original dataframe; else, use original dataframe in selection.
        if copy:
            region = self.data.copy()
        else:
            region = self.data
        if region_id is not None:
            if isinstance(region_id, (float, int, str)):
                region_id = [region_id]
            elif not isinstance(region_id, list):
                raise TypeError(
                    f"Invalid region_id type: {type(region_id)}. Must be \
                                of type float, int, str, list, ``None``."
                )
            # Select row by column id
            for value in region_id:
                if not isinstance(value, (float, int, str)):
                    raise TypeError(
                        f"Invalid element in list region_id. Is of \
                            type: {type(value)}Must be \
                            of type float, int, str."
                    )
            region = self.data[self.data["region_id"].isin(region_id)]
        if time_range is not None:
            if isinstance(time_range, List):
                if len(time_range) == 2:
                    if isinstance(time_range[0], Timestamp) and isinstance(
                        time_range[1], Timestamp
                    ):
                        if time_range[0] < time_range[1]:
                            # Select rows with time values that are all within time range
                            region = region[
                                region["time"].apply(
                                    lambda time_array: all(
                                        time_range[0] <= Timestamp(x)
                                        or time_range[1] >= Timestamp(x)
                                        for x in time_array
                                    )
                                )
                            ]
                        else:
                            raise ValueError(
                                f"1st index value must be later than 0th index \
                                             value. Currently 0th index value is {time_range[0]} \
                                             and 1st index value is {time_range[1]}"
                            )
                    else:
                        raise TypeError(
                            f"Invalid time_range value types: \
                                        {type(time_range[0])} and {type(time_range[1])}. Must \
                                        be both of type Timestamp."
                        )
                else:
                    raise ValueError(
                        f"Invalid time_range size: {len(time_range)}. \
                        Must be of size 2."
                    )
            else:
                raise TypeError(
                    f"Invalid time_range type: {type(time_range)}. Must be \
                                of type List."
                )
        if depth_range is not None:
            if isinstance(depth_range, List):
                if len(depth_range) == 2:
                    if isinstance(depth_range[0], (float, int)) and isinstance(
                        depth_range[1], (float, int)
                    ):
                        if depth_range[0] < depth_range[1]:
                            # Select rows with depth values that are all within depth range
                            region = region[
                                region["time"].apply(
                                    lambda depth_array: all(
                                        depth_range[0] <= float(x)
                                        or depth_range[1] >= float(x)
                                        for x in depth_array
                                    )
                                )
                            ]
                        else:
                            raise ValueError(
                                f"1st index value must be later than 0th index \
                                             value. Currently 0th index value is {depth_range[0]} \
                                             and 1st index value is {depth_range[1]}"
                            )
                    else:
                        raise TypeError(
                            f"Invalid depth_range value types: \
                                        {type(depth_range[0])} and {type(depth_range[1])}. Must \
                                        be both of type either float or int."
                        )
                else:
                    raise ValueError(
                        f"Invalid depth_range size: {len(depth_range)}. \
                        Must be of size 2."
                    )
            else:
                raise TypeError(
                    f"Invalid depth_range type: {type(depth_range)}. Must be \
                                of type List."
                )
        return region

    def close_region(
        self, region: Union[float, str, List, Series, DataFrame] = None
    ) -> DataFrame:
        """Close a region by appending the first point to end of the list of points.

        Parameters
        ----------
        region : float, str, list, Series, DataFrame, ``None``
            region(s) to select raw files with
            If ``None``, select all regions. Defaults to ``None``

        Returns
        -------
        DataFrame
            Returns a new DataFrame with closed regions
        """
        region = self.select_region(region, copy=True)
        region["time"] = region.apply(
            lambda row: np.append(row["time"], row["time"][0]), axis=1
        )
        region["depth"] = region.apply(
            lambda row: np.append(row["depth"], row["depth"][0]), axis=1
        )
        return region

    def select_sonar_file(
        self,
        sonar_file_names: List[str],
        region: Union[float, str, list, Series, DataFrame] = None,
    ) -> List:
        """Finds sonar files in the time domain that encompasses region or list of regions

        Parameters
        ----------
        files : list
            raw filenames
        region : float, str, list, Series, DataFrame, ``None``
            region(s) to select sonar files with
            If ``None``, select all regions. Defaults to ``None``

        Returns
        -------
        files: list
            list of raw file(s) spanning the encompassing region or list of regions.
        """
        sonar_file_names.sort()
        filetimes = parse_simrad_fname_time(
            [Path(fname).name for fname in sonar_file_names]
        ).values

        # Ensure that region is a DataFrame
        region = self.select_region(region)

        times = np.hstack(region["time"].values)
        lower_idx = np.searchsorted(filetimes, times.min()) - 1
        upper_idx = np.searchsorted(filetimes, times.max())

        lower_idx = 0 if lower_idx < 0 else lower_idx

        sonar_file_names = sonar_file_names[lower_idx:upper_idx]
        return sonar_file_names

    def replace_nan_depth(self, inplace: bool = False) -> DataFrame:
        """Replace 9999.99 or -9999.99 depth values with user-specified min_depth and max_depth

        Parameters
        ----------
        inplace : bool
            Modify the current `data` inplace

        Returns
        -------
        DataFrame with depth edges replaced by Regions2D.min_depth and  Regions2D.max_depth
        """

        def replace_depth(row: Series) -> Series:
            def swap_val(val: Union[int, float]) -> Union[int, float]:
                if val == 9999.99:
                    return self.max_depth
                elif val == -9999.99:
                    return self.min_depth
                else:
                    return val

            row.at["region_bbox_top"] = swap_val(row["region_bbox_top"])
            row.at["region_bbox_bottom"] = swap_val(row["region_bbox_bottom"])
            for idx, val in enumerate(row["depth"]):
                row["depth"][idx] = swap_val(val)
            return row

        if self.min_depth is None and self.max_depth is None:
            return

        regions = self.data if inplace else self.data.copy()
        regions.loc[:] = regions.apply(replace_depth, axis=1)
        return regions

    def convert_points(
        self,
        points: Union[List, Dict, DataFrame],
        convert_time: bool = True,
        convert_depth_edges: bool = True,
    ) -> Union[List, Dict]:
        """Convert x and y values of points from the EV format.
        Returns a copy of points.
        Parameters
        ----------
        points : list, dict
            point in [x, y] format or list/dict of these
        convert_time : bool
            Whether to convert EV time to datetime64, defaults `True`
        convert_depth_edges : bool
            Whether to convert -9999.99 edges to real range values.
            Min and max ranges must be set manually or by calling `set_range_edge_from_raw`
        unix : bool
            unix : bool
            Whether or not to output the time in the unix time format
        Returns
        -------
        points : list or dict
            single converted point or list/dict of converted points depending on input
        """

        def _swap_depth_edge(self, y: Union[int, float]) -> Union[int, float]:
            if float(y) == 9999.99 and self.max_depth is not None:
                return self.max_depth
            elif float(y) == -9999.99 and self.min_depth is not None:
                return self.min_depth
            else:
                return float(y)

        def _convert_single(point: List) -> None:
            if convert_time:
                point[0] = matplotlib.dates.date2num(point[0])

            if convert_depth_edges:
                point[1] = _swap_depth_edge(point[1])

        if isinstance(points, dict):
            for point in points.values():
                _convert_single(point)
        else:
            for point in points:
                _convert_single(point)

        return points

    def plot(
        self,
        region: Union[str, List, DataFrame] = None,
        close_region: bool = False,
        **kwargs,
    ) -> None:
        """Plot a region from data.
        Automatically convert time and range_edges.

        Parameters
        ---------
        region : float, str, list, Series, DataFrame, ``None``
            Region(s) to select raw files with
            If ``None``, select all regions. Defaults to ``None``
        close_region : bool
            Plot the region as a closed polygon. Defaults to False
        kwargs : keyword arguments
            Additional arguments passed to matplotlib plot
        """

        # Ensure that region is a DataFrame
        region = self.select_region(region)

        if close_region:
            region = self.Regions2D.close_region(region)
        for _, row in region.iterrows():
            plt.plot(row["time"], row["depth"], **kwargs)

    def mask(
        self,
        da_Sv: DataArray,
        region_ids: List,
        mask_var: str = None,
        mask_labels=None,
    ) -> DataArray:
        """Mask data from Data Array containing Sv data based off of a Regions2D object
        and its regions ids.

        Parameters
        ----------
        da_Sv : Data Array
            DataArray of shape (ping_time, depth) containing Sv data.
        region_ids : list
            list IDs of regions to create mask for
        mask_var : str
            If provided, used to name the output mask array, otherwise `mask`
        mask_labels:
            None: assigns labels automatically 0,1,2,...
            "from_ids": uses the region ids
            list: uses a list of integers as labels

        Returns
        -------
        A DataArray with the data_var masked by the specified region.
        """
        if type(region_ids) == list:
            if len(region_ids) == 0:
                raise ValueError("region_ids is empty. Cannot be empty.")
        else:
            raise TypeError(
                f"region_ids must be of type list. Currently is of type {type(region_ids)}"
            )

        if isinstance(mask_labels, list) and (len(mask_labels) != len(region_ids)):
            raise ValueError(
                "If mask_labels is a list, it should be of same length as region_ids."
            )

        # Replace nan depth in regions2d.
        self.replace_nan_depth(inplace=True)

        # Dataframe containing region information.
        region_df = self.select_region(region_ids)

        # Select only columns which are important.
        region_df = region_df[["region_id", "time", "depth"]]

        # Organize the regions in a format for region mask.
        df = region_df.explode(["time", "depth"])

        # Convert region time to integer timestamp.
        df["time"] = matplotlib.dates.date2num(df["time"])

        # Create a list of dataframes for each regions.
        grouped = list(df.groupby("region_id"))

        # Convert to list of numpy arrays which is an acceptable format to create region mask.
        regions_np = [np.array(region[["time", "depth"]]) for id, region in grouped]

        # Corresponding region ids converted to int.
        region_ids = [int(id) for id, region in grouped]

        # Convert ping_time to unix_time since the masking does not work on datetime objects.
        da_Sv = da_Sv.assign_coords(
            unix_time=(
                "ping_time",
                matplotlib.dates.date2num(da_Sv.coords["ping_time"].values),
            )
        )

        # Set up mask labels.
        if mask_labels:
            if mask_labels == "from_ids":
                # Create mask.
                r = regionmask.Regions(outlines=regions_np, numbers=region_ids)
                M = r.mask(
                    da_Sv["unix_time"],
                    da_Sv["depth"],
                    wrap_lon=False,
                )

            elif isinstance(mask_labels, list):
                # Create mask.
                r = regionmask.Regions(outlines=regions_np)
                M = r.mask(
                    da_Sv["unix_time"],
                    da_Sv["depth"],
                    wrap_lon=False,
                )
                # Convert default labels to mask_labels.
                S = xr.where(~M.isnull(), 0, M)
                S = M
                for idx, label in enumerate(mask_labels):
                    S = xr.where(M == idx, label, S)
                M = S
            else:
                raise ValueError("mask_labels must be None, 'from_ids', or a list.")
        else:
            # Create mask.
            r = regionmask.Regions(outlines=regions_np)
            try:
                M = r.mask(
                    da_Sv["unix_time"],
                    da_Sv["depth"],
                    wrap_lon=False,
                )
            except ValueError as ve:
                import warnings

                warnings.warn(
                    "Most likely using deprecated regionmask version."
                    "Make sure to use regionmask==0.8.0 or more recent versions.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                raise ve

        # Assign specific name to mask array, otherwise 'mask'.
        if mask_var:
            M = M.rename(mask_var)

        return M
