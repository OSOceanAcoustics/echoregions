import warnings
from pathlib import Path
from typing import Iterable, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from pandas import DataFrame, Series, Timestamp, isna
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
                                        and time_range[1] >= Timestamp(x)
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
                                        and depth_range[1] >= float(x)
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
        """Finds SIMRAD sonar files in the time domain that encompasses region or list of regions.

        SIMRAD Format Explained with the example Summer2017-D20170625-T205018.nc:

        The letter "D" is a prefix indicating the date in the format following it. In this case,
        "20170625" represents the date June 25, 2017. The letter "T" is a prefix indicating the
        time in the format following it. In this case, "205018" represents the time 20:50:18
        (8:50:18 PM) in 24-hour format. The .nc is a file extension that denotes a NetCDF (Network
        Common Data Form) file.

        Parameters
        ----------
        files : list
            Raw filenames in SIMRAD format.
        region : float, str, list, Series, DataFrame, ``None``
            Region(s) to select sonar files with.
            If ``None``, select all regions. Defaults to ``None``

        Returns
        -------
        files: list
            list of raw/Sv sonar file(s) spanning the encompassing region or list of regions.
        """
        # Check that sonar_file_names is a list
        if not isinstance(sonar_file_names, list):
            raise TypeError(
                f"sonar_file_names must be type list. Filenames is of type {type(sonar_file_names)}"
            )

        # Sort sonar file names
        sonar_file_names.sort()

        # Parse simrad filenames
        sonar_file_times = parse_simrad_fname_time(
            [Path(fname).name for fname in sonar_file_names]
        ).values

        # Ensure that region is a DataFrame
        region = self.select_region(region)

        # Extract region time values
        region_times = np.hstack(region["time"].values)

        # Check if all sonar file times are completely below or above all region times
        if np.all(sonar_file_times < region_times.min()) or np.all(
            sonar_file_times > region_times.max()
        ):
            return []
        else:
            # Get lower and upper index of filetimes
            lower_idx = np.searchsorted(sonar_file_times, region_times.min()) - 1
            upper_idx = np.searchsorted(sonar_file_times, region_times.max())

            # Set lower idx to 0 if at -1
            lower_idx = 0 if lower_idx < 0 else lower_idx

            # Subset sonar file names based on lower and upper index
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
        DataFrame with depth edges replaced by Regions2D.min_depth and Regions2D.max_depth
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
        if isinstance(region_ids, list):
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
        if mask_labels == "from_ids":
            r = regionmask.Regions(outlines=regions_np, numbers=region_ids)
        elif isinstance(mask_labels, list) or mask_labels is None:
            r = regionmask.Regions(outlines=regions_np)
        else:
            raise ValueError("mask_labels must be None, 'from_ids', or a list.")

        # Create mask
        try:
            M = r.mask(
                da_Sv["unix_time"],
                da_Sv["depth"],
                wrap_lon=False,
            )
        except ValueError as ve:
            warnings.warn(
                "Most likely using deprecated regionmask version."
                "Make sure to use regionmask==0.8.0 or more recent versions.",
                DeprecationWarning,
                stacklevel=2,
            )
            raise ve

        if isinstance(mask_labels, list):
            # Convert default labels to mask_labels.
            S = xr.where(~M.isnull(), 0, M)
            S = M
            for idx, label in enumerate(mask_labels):
                S = xr.where(M == idx, label, S)
            M = S

        # Assign specific name to mask array, otherwise 'mask'.
        if mask_var:
            M = M.rename(mask_var)

        return M

    def transect_mask(
        self,
        da_Sv: DataArray,
        transect_dict: dict = {
            "start": "ST",
            "break": "BT",
            "resume": "RT",
            "end": "ET",
        },
        bbox_distance_threshold: float = 1.0,
    ) -> DataArray:
        """Mask data from Data Array containing Sv data based off of a Regions2D object
        and its transect_values.

        We should note that this convention for start, break, resume, end transect is very
        specific to the convention used for the NOAA Hake Survey. If you would like to add
        your own schema for transect logging, please create an issue in the following link:
        https://github.com/OSOceanAcoustics/echoregions/issues.

        Parameters
        ----------
        da_Sv : Data Array
            DataArray of shape (ping_time, depth) containing Sv data.
        transect_dict : dict
            Dictionary for transect values. Values must be unique.
        bbox_distance_threshold: float
            The maximum value for how far apart the left and right bounding box for each transect
            value region. Default is set to 1 minute.
        Returns
        -------
        M : Data Array
            A DataArray masked by the transect values from the Regions2d.data dataframe
            with dimensions (ping_time, depth).
        """

        # Get transect strings
        start_str = transect_dict["start"]
        break_str = transect_dict["break"]
        resume_str = transect_dict["resume"]
        end_str = transect_dict["end"]
        transect_strs = [start_str, break_str, resume_str, end_str]

        # Check that there are 4 unique transect strings
        if len(transect_strs) != len(set(transect_strs)):
            raise ValueError(
                "There exist duplicate values in transect_dict. "
                "All values must be unique."
            )
        for transect_str in transect_strs:
            if not isinstance(transect_str, str):
                raise TypeError(
                    f"Transect dictionary values must be strings. There exists a "
                    f"value of type {type(transect_str)} in transect dictionary."
                )

        # Create transect_df
        region_df = self.data.copy()
        transect_df = region_df.loc[
            region_df.loc[:, "region_name"].str.startswith(start_str)
            | region_df.loc[:, "region_name"].str.startswith(break_str)
            | region_df.loc[:, "region_name"].str.startswith(resume_str)
            | region_df.loc[:, "region_name"].str.startswith(end_str)
        ].copy()

        # Create a new column which stores the transect_type without the transect number
        transect_df.loc[:, "transect_type"] = transect_df.loc[
            :, "region_name"
        ].str.extract(rf"({start_str}|{break_str}|{resume_str}|{end_str})")

        # Check if for all transects, there exists 1 start_str transect type.
        # If there does not exists a start_str transect, set the first region
        # to be the start_str transect.
        if not (
            transect_df.groupby("file_name").apply(
                lambda x: x[x["transect_type"] == start_str].count()
            )["file_name"]
            == 1
        ).all():
            warnings.warn(
                UserWarning(
                    f"There exists a transect that does not contain a single {start_str} "
                    "transect_type."
                )
            )
            # Modify first row of original dataframe such that its transect type has value start_str
            # and add it into transect df as its first row.
            first_row = region_df.loc[region_df.index[0]].copy().to_frame().T
            first_row["transect_type"] = start_str
            transect_df = pd.concat([first_row, transect_df]).reset_index(drop=True)

        # Check if for all transects, there exists 1 end_str transect type.
        if not (
            transect_df.groupby("file_name").apply(
                lambda x: x[x["transect_type"] == end_str].count()
            )["file_name"]
            == 1
        ).all():
            warnings.warn(
                UserWarning(
                    f"There exists a transect that does not contain a single {end_str} "
                    "transect type."
                )
            )
            # Modify last row of original dataframe such that its transect type has value end_str
            # and add it into transect df as its last row.
            last_row = region_df.tail(1).copy()
            last_row["transect_type"] = end_str
            transect_df = pd.concat([transect_df, last_row]).reset_index(drop=True)

        # Checking the maximum width of a transect log region bbox.
        # If over a minute, throw an error.
        max_time = (
            transect_df["region_bbox_right"] - transect_df["region_bbox_left"]
        ).max()
        max_time_minutes = max_time.total_seconds() / 60
        if max_time_minutes > bbox_distance_threshold:
            Warning(
                f"Maximum width in time of transect log region bboxs is "
                f"too large i.e. over {bbox_distance_threshold} minute(s). "
                f"The maximum width is: {max_time_minutes}.",
                UserWarning,
            )

        # Drop time duplicates
        transect_df = transect_df.drop_duplicates(subset=["region_bbox_left"])

        # Sort the dataframe by datetime
        transect_df = transect_df.sort_values(by="region_bbox_left")
        # Create new shifted columns with the next transect log type and next region
        # bbox left datetime value.
        transect_df.loc[:, "transect_type_next"] = transect_df.loc[
            :, "transect_type"
        ].shift(-1)
        transect_df.loc[:, "region_bbox_left_next"] = transect_df.loc[
            :, "region_bbox_left"
        ].shift(-1)

        # Check if start_str followed by break_str/end_str.
        start_transect_rows = transect_df[transect_df["transect_type"] == start_str]
        start_transect_type_next_list = list(
            start_transect_rows["transect_type_next"].values
        )
        for transect_type_next in start_transect_type_next_list:
            if transect_type_next not in [break_str, end_str]:
                raise ValueError(
                    f"Transect start string is followed by invalid value "
                    f"{transect_type_next}. Must be followed by either "
                    f"{break_str} or {end_str}"
                )

        # Check if break_str followed by resume_str.
        break_transect_rows = transect_df[transect_df["transect_type"] == break_str]
        break_transect_type_next_list = list(
            break_transect_rows["transect_type_next"].values
        )
        for transect_type_next in break_transect_type_next_list:
            if transect_type_next != resume_str:
                raise ValueError(
                    f"Transect break string is followed by invalid value "
                    f"{transect_type_next}. Must be followed by {resume_str}."
                )

        # Check if resume_str followed by break_str/end_str.
        resume_transect_rows = transect_df[transect_df["transect_type"] == resume_str]
        resume_transect_type_next_list = list(
            resume_transect_rows["transect_type_next"].values
        )
        for transect_type_next in resume_transect_type_next_list:
            if transect_type_next not in [break_str, end_str]:
                raise ValueError(
                    f"Transect resume string is followed by invalid value "
                    f"{transect_type_next}. Must be followed by either "
                    f"{break_str} or {end_str}."
                )

        # Check if end_str followed by start_str or if NA.
        end_transect_rows = transect_df[transect_df["transect_type"] == end_str]
        end_transect_type_next_list = list(
            end_transect_rows["transect_type_next"].values
        )
        for transect_type_next in end_transect_type_next_list:
            # If this value is not NA, check if it is start_str.
            if not isna(transect_type_next):
                if transect_type_next != start_str:
                    raise ValueError(
                        f"Transect end string is followed by invalid value "
                        f"{transect_type_next}. Must be followed by {start_str}."
                    )

        # Create binary variable indicating within transect segments.
        transect_df["within_transect"] = False

        # Indices where start_str followed by break_str/end_str
        st_indices = (transect_df["transect_type"] == start_str) & transect_df[
            "transect_type_next"
        ].isin([break_str, end_str])
        transect_df.loc[st_indices, "within_transect"] = True

        # Indices where resume_str followed by break_str/end_str
        rt_indices = (transect_df["transect_type"] == resume_str) & transect_df[
            "transect_type_next"
        ].isin([break_str, end_str])
        transect_df.loc[rt_indices, "within_transect"] = True

        # Get all unique file_names in transect_df.
        transect_querying_list = list(transect_df["file_name"].unique())

        # Create list of masks for each file name to be queried.
        mask_list = []
        for transect_querying_file_name in transect_querying_list:
            within_transect_df = transect_df.query(
                f'file_name == "{transect_querying_file_name}" and within_transect == True'
            )
            T = xr.zeros_like(da_Sv)
            for _, row in within_transect_df.iterrows():
                T = T + xr.where(
                    (T.ping_time > row["region_bbox_left"])
                    & (T.ping_time < row["region_bbox_left_next"]),
                    1,
                    0,
                )
            mask_list.append(T)

        # Combine masks.
        M = xr.zeros_like(da_Sv)
        for _, T in enumerate(mask_list):
            M = M + T

        # If M contains channel dimension, then drop it.
        if "channel" in M.dims:
            M = M.isel(channel=0)

        return M
