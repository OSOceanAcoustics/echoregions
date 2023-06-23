from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series, Timestamp

from ..utils.io import validate_path
from ..utils.time import parse_simrad_fname_time
from .regions2d_parser import parse_regions_file


class Regions2D:
    def __init__(
        self,
        input_file: str,
        min_depth: Union[int, float] = None,
        max_depth: Union[int, float] = None,
        depth: ndarray = None,
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
        self.data = parse_regions_file(input_file)
        self.output_file = []

        self.depth = depth
        self.max_depth = max_depth
        self.min_depth = min_depth

    def __iter__(self) -> Iterable:
        return self.data.iterrows()

    def __getitem__(self, val: int) -> Series:
        return self.data.iloc[val]

    @property
    def max_depth(self) -> Union[int, float]:
        """Get the depth value that the 9999.99 edge will be set to"""
        if self._max_depth is None and self.depth is not None:
            self._max_depth = self.depth.max()
        return self._max_depth

    @max_depth.setter
    def max_depth(self, val: Union[int, float]) -> Union[int, float]:
        """Set the depth value that the 9999.99 edge will be set to"""
        if self.min_depth is not None:
            if val <= self.min_depth:
                raise ValueError("max_depth cannot be less than min_depth")
        self._max_depth = float(val) if val is not None else val

    @property
    def min_depth(self) -> Union[int, float]:
        """Get the depth value that the -9999.99 edge will be set to"""
        if self._min_depth is None and self.depth is not None:
            self._min_depth = self.depth.min()
        return self._min_depth

    @min_depth.setter
    def min_depth(self, val: Union[int, float]) -> None:
        """Set the depth value that the -9999.99 edge will be set to"""
        if self.max_depth is not None:
            if val >= self.max_depth:
                raise ValueError("min_depth cannot be greater than max_depth")
        self._min_depth = float(val) if val is not None else val

    def to_csv(self, save_path: bool = None) -> None:
        """Save a Dataframe to a .csv file

        Parameters
        ----------
        save_path : str
            path to save the CSV file to
        """
        if not isinstance(self.data, DataFrame):
            raise TypeError(
                f"Invalid ds Type: {type(self.data)}. Must be of type DataFrame."
            )

        # Check if the save directory is safe
        save_path = validate_path(
            save_path=save_path, input_file=self.input_file, ext=".csv"
        )
        # Reorder columns and export to csv
        self.data.to_csv(save_path, index=False)
        self.output_file.append(save_path)

    def to_json(self, save_path: str = None) -> None:
        # TODO: Implement this function
        """Convert EVR to a JSON file.

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        pretty : bool, default False
            Output more human readable JSON
        """

    def select_region(
        self,
        region_id: Union[float, str, list, Series, DataFrame] = None,
        time_range: List[Timestamp] = None,
        depth_range: List[Union[float, int]] = None,
        copy=False,
    ) -> DataFrame:
        """Selects a subset of this region object's dataframe.

        Parameters
        ----------
        region_id : float, str, list, Series, DataFrame, ``None``
            A region id provided as a number, string, list of these,
            or a DataFrame/Series containing the region_id column name.
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
            A DataFrame subselected from Regions2D.data.
            There is a row for each region id provided by the region_id parameter,
            and each row has time and depth within or on the boundaries passed
            in by the time_range and depth_range values.
        """
        region = None
        untouched = True
        if region_id is not None:
            if isinstance(region_id, DataFrame):
                region = list(region_id.region_id)
            elif isinstance(region_id, Series):
                region = [region_id.region_id]
            elif (
                isinstance(region_id, float)
                or isinstance(region_id, int)
                or isinstance(region_id, str)
            ):
                region_id = [region_id]
            elif not isinstance(region_id, list):
                raise TypeError(
                    f"Invalid region_id type: {type(region_id)}. Must be \
                                of type float, str, list, Series, DataFrame, ``None``"
                )
            # Select row by column id
            region = self.data[self.data["region_id"].isin(region_id)]
            untouched = False
        if time_range is not None:
            if isinstance(time_range, List):
                if len(time_range) == 2:
                    if isinstance(time_range[0], Timestamp) and isinstance(
                        time_range[1], Timestamp
                    ):
                        if time_range[0] < time_range[1]:
                            if region is None:
                                region = self.data
                            for index, row in region.iterrows():
                                remove_row = False
                                for time in row["time"]:
                                    if time_range[0] > Timestamp(time) or time_range[
                                        1
                                    ] < Timestamp(time):
                                        remove_row = True
                                if remove_row:
                                    region.drop(index, inplace=True)
                            untouched = False
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
                            if region is None:
                                region = self.data
                            for index, row in region.iterrows():
                                remove_row = False
                                for depth in row["depth"]:
                                    if depth_range[0] > depth or depth_range[1] < depth:
                                        remove_row = True
                                if remove_row:
                                    region.drop(index, inplace=True)
                            untouched = False
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
        if untouched:
            region = self.data
        if copy:
            return region.copy()
        else:
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
        files: List[str],
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
        files.sort()
        filetimes = parse_simrad_fname_time(
            [Path(fname).name for fname in files]
        ).values

        # Ensure that region is a DataFrame
        region = self.select_region(region)

        times = np.hstack(region["time"].values)
        lower_idx = np.searchsorted(filetimes, times.min()) - 1
        upper_idx = np.searchsorted(filetimes, times.max())

        lower_idx = 0 if lower_idx < 0 else lower_idx

        files = files[lower_idx:upper_idx]
        return files

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
