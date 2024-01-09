import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from pandas import DataFrame, Series, Timestamp, isna
from xarray import DataArray, Dataset

from ..utils.api import convert_mask_3d_to_2d
from ..utils.io import validate_path
from ..utils.time import parse_simrad_fname_time
from .regions2d_parser import parse_evr, parse_regions_df


class Regions2D:
    """
    Class that contains and performs operations with Regions2D data from Echoview EVR files.
    """

    def __init__(
        self,
        input_file: str,
        min_depth: Union[int, float] = None,
        max_depth: Union[int, float] = None,
        input_type: str = "EVR",
    ):
        self.input_file = input_file
        if input_type == "EVR":
            self.data = parse_evr(input_file)
        elif input_type == "CSV":
            self.data = parse_regions_df(input_file)
        else:
            raise ValueError(f"Regions2D input_type must be EVR or CSV. Got {input_type} instead.")
        self.output_file = []

        self.min_depth = min_depth
        self.max_depth = max_depth

    def __iter__(self) -> Iterable:
        return self.data.iterrows()

    def __getitem__(self, val: int) -> Series:
        return self.data.iloc[val]

    def to_csv(self, save_path: bool = None, mode="w") -> None:
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

        # Save to CSV
        self.data.to_csv(save_path, index=False, mode=mode)

        # Append save_path
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
        time_range : List of 2 Pandas Timestamps.
            Datetime range for expected output of subselected DataFrame. 1st
            index value must be later than 0th index value.
        depth_range : List of 2 floats.
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
                                        depth_range[0] <= float(x) and depth_range[1] >= float(x)
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

    def close_region(self, region_id: List = None) -> DataFrame:
        """Close a region by appending the first point to end of the list of points.

        Parameters
        ----------
        region_id : List, ``None``
            region(s) to select raw files with
            If ``None``, select all regions. Defaults to ``None``

        Returns
        -------
        DataFrame
            Returns a new DataFrame with closed regions
        """
        region = self.select_region(region_id, copy=True)
        region["time"] = region.apply(lambda row: np.append(row["time"], row["time"][0]), axis=1)
        region["depth"] = region.apply(lambda row: np.append(row["depth"], row["depth"][0]), axis=1)
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
            print("Sonar file times did not overlap at all with region times. Returning empty list")
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
        region_id: List = None,
        close_regions: bool = False,
        **kwargs,
    ) -> None:
        """Plot a region from data.
        Automatically convert time and range_edges.

        Parameters
        ---------
        region_id : List, ``None``
            Region(s) to select raw files with
            If ``None``, select all regions. Defaults to ``None``
        close_region : bool
            Plot the region as a closed polygon. Defaults to False
        kwargs : keyword arguments
            Additional arguments passed to matplotlib plot
        """

        # Ensure that region is a DataFrame
        region = self.select_region(region_id)

        if close_regions:
            region = self.close_region(region)
        for _, row in region.iterrows():
            plt.plot(row["time"], row["depth"], **kwargs)

    def mask(
        self,
        da_Sv: DataArray,
        region_id: List = None,
        mask_name: str = "mask",
        mask_labels: dict = None,
        collapse_to_2d: bool = False,
    ) -> Optional[Dataset]:
        """
        Mask data from Data Array containing Sv data based off of a Regions2D object
        and its regions ids.

        Parameters
        ----------
        da_Sv : Data Array
            DataArray of shape (ping_time, depth) containing Sv data.
        region_id : list
            list IDs of regions to create mask for
        mask_name : str
            If provided, used to name the output mask array, otherwise `mask`
        mask_labels : dict
            If provided, used to label the region_id dimension of the output mask.
        collapse_to_2d : bool
            If true, then converts 3d mask to 2d mask. If not, keeps output as 3d mask.

        Returns
        -------
        mask_ds : Dataset
            Either a 3D mask or a 2D mask based on the conditions below.
            If collapse_to_2d is False:
                A 3D mask where each layer of the mask will contain a 1s/0s mask for each
                unique label in the 2D mask. The layers will be labeled via region_id
                values. The slices of the 3D array will be in the form of 1s/0s: masked areas,
                and non-masked areas.
            If collapse_to_2d is True:
                A 2D mask where each individual data points will be in the form of integers,
                demarking region_id of masked regions, and nan values, demarking non-masked
                areas.
            Also contains a data variable (`mask_labels`) with mask labels
            corresponding to region_id values.
        region_contours : pd.DataFrame
            DataFrame containing region_id, depth, and time.
        """
        if isinstance(region_id, list):
            if len(region_id) == 0:
                raise ValueError("region_id is empty. Cannot be empty.")
        elif region_id is None:
            # Extract all region_id values
            region_id = self.data.region_id.astype(int).to_list()
        else:
            raise TypeError(
                f"region_id must be of type list. Currently is of type {type(region_id)}"
            )

        if mask_labels is None:
            # Create mask_labels with each region_id as a key and values starting from 0
            mask_labels = {key: idx for idx, key in enumerate(region_id)}

        # Check that region_id and mask_labels are of the same size
        if len(set(region_id) - set(mask_labels.keys())) > 0:
            raise ValueError(
                "Each region_id' must be a key in 'mask_labels'. "
                "If you would prefer 0 based indexing as values for mask_labels, leave "
                "mask_labels as None."
            )

        # Dataframe containing region information.
        region_df = self.select_region(region_id)

        # Select only important columns
        region_df = region_df[["region_id", "time", "depth"]]

        # Filter for rows with depth values within self min and self max depth and
        # for rows that have positive depth.
        region_df = region_df[
            region_df["depth"].apply(
                lambda x: all(
                    (i >= np.max(0, int(self.min_depth)) and i <= int(self.max_depth)) for i in x
                )
            )
        ]

        if region_df.empty:
            print(
                "All rows in Regions DataFrame have NaN Depth values after filtering depth "
                "between min_depth and max_depth."
            )
        else:
            # Organize the regions in a format for region mask.
            df = region_df.explode(["time", "depth"])

            # Convert region time to integer timestamp.
            df["time"] = matplotlib.dates.date2num(df["time"])

            # Create a list of dataframes for each regions.
            grouped = list(df.groupby("region_id"))

            # Convert to list of numpy arrays which is an acceptable format to create region mask.
            regions_np = [np.array(region[["time", "depth"]]) for _, region in grouped]

            # Convert corresponding region_id to int.
            filtered_region_id = [int(id) for id, _ in grouped]

            # Convert ping_time to unix_time since the masking does not work on datetime objects.
            da_Sv = da_Sv.assign_coords(
                unix_time=(
                    "ping_time",
                    matplotlib.dates.date2num(da_Sv.coords["ping_time"].values),
                )
            )

            # Create mask
            r = regionmask.Regions(
                outlines=regions_np, numbers=filtered_region_id, name=mask_name, overlap=True
            )
            mask_da = r.mask_3D(
                da_Sv["unix_time"],
                da_Sv["depth"],
                wrap_lon=False,
            ).astype(
                int
            )  # This maps False to 0 and True to 1

            # Rename region coords with region_id coords
            mask_da = mask_da.rename({"region": "region_id"})

            # Remove all coords other than depth, ping_time, region_id
            mask_da = mask_da.drop_vars(
                mask_da.coords._names.difference({"depth", "ping_time", "region_id"})
            )

            # Transpose mask_da
            mask_da = mask_da.transpose("region_id", "depth", "ping_time")

            # Remove attribute standard_name
            mask_da.attrs.pop("standard_name")

            # Rename Data Array to mask_name
            mask_da = mask_da.rename(mask_name)

            # Set mask_labels_da
            masked_region_id = mask_da.region_id.values.tolist()
            subset_mask_labels = [
                mask_labels[key] for key in masked_region_id if key in mask_labels
            ]
            mask_labels_da = xr.DataArray(
                subset_mask_labels,
                dims="region_id",
                coords={"region_id": masked_region_id},
            )

            # Create dataset
            mask_ds = xr.Dataset()
            mask_ds["mask_labels"] = mask_labels_da
            mask_ds["mask_3d"] = mask_da

            if collapse_to_2d:
                # Convert 3d mask to 2d mask
                mask_ds = convert_mask_3d_to_2d(mask_ds)

            # Get region_contours
            region_contours = region_df[region_df["region_id"].isin(masked_region_id)][
                ["region_id", "time", "depth"]
            ]

            return mask_ds, region_contours

    def transect_mask(
        self,
        da_Sv: DataArray,
        transect_sequence_type_dict: dict = {
            "start": "ST",
            "break": "BT",
            "resume": "RT",
            "end": "ET",
        },
        transect_sequence_type_next_allowable_dict: dict = {
            "ST": ["BT", "ET"],
            "BT": ["RT"],
            "RT": ["BT", "ET"],
            "ET": ["ET", ""],
        },
        bbox_distance_threshold: float = 1.0,
        must_pass_check: bool = False,
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
        transect_sequence_type_dict : dict
            Dictionary for transect sequence type values. The values denote where in the context
            of the transect each region lays in, i.e. whether we are at the beginning of the
            transect, a break in the transect, a resumption of the transect, or at the end
            of the transect.
        transect_sequence_type_next_allowable_dict : dict
            Dictionary for the allowable transect sequence type value(s) that can follow a
            transect sequence type value.
        bbox_distance_threshold: float
            Maximum allowable value between the left and right bounding box timestamps
            for each transect value region. Default is set to 1 minute.
        must_pass_check : bool
            True: Will check transect strings to enforce sequence rules. If this check
            encounters any incorrect transect type sequence orders or wider than bbox distance
            threshold regions, it will raise an exception.
            False: Will still check transect strings but will instead just print out warnings
            for violations of the above mentioned sequence rules.

        Returns
        -------
        M : Data Array
            A binary DataArray with dimensions (ping_time, depth) where 1s are within transect
            and 0s are outside transect.
        """

        def _check_transect_sequences(transect_df, must_pass_check):
            # Create an empty list to collect error messages.
            warning_messages = []

            # Ensure correct sequence of transect types occur.
            # If they do not, append to warning_messages.
            for _, row in transect_df.iterrows():
                transect_type = row["transect_type"]
                transect_type_next = row["transect_type_next"]
                # Check for correct transect_type_next values
                if (
                    transect_type_next
                    not in transect_sequence_type_next_allowable_dict[transect_type]
                ):
                    type_next_warning_message = (
                        f"Error in region_id {row['region_id']}:"
                        f"Transect string {transect_type} is followed by "
                        f"invalid value {transect_type_next}. Must be followed by "
                        f"{transect_sequence_type_next_allowable_dict[transect_type]}"
                    )
                    warning_messages.append(type_next_warning_message)

            # Identify rows wider than bbox distance threshold if they exist
            wider_than_bbox_distance_threshold_rows = transect_df[
                (
                    transect_df["region_bbox_right"] - transect_df["region_bbox_left"]
                ).dt.total_seconds()
                / 60
                > bbox_distance_threshold
            ]
            wider_than_bbox_distance_threshold_region_ids = wider_than_bbox_distance_threshold_rows[
                "region_id"
            ].tolist()
            if wider_than_bbox_distance_threshold_region_ids:
                warning_messages.append(
                    f"Problematic region id values with maximum time width wider than bbox "
                    f"distance threshold: {wider_than_bbox_distance_threshold_region_ids}"
                )

            # Raise an exception if there are any warning messages and must_pass_check is True.
            # Else, print warning messages.
            if len(warning_messages) > 0 and must_pass_check:
                raise Exception("\n".join(warning_messages))
            else:
                print("\n".join(warning_messages))

        # Get transect strings
        start_str = transect_sequence_type_dict["start"]
        break_str = transect_sequence_type_dict["break"]
        resume_str = transect_sequence_type_dict["resume"]
        end_str = transect_sequence_type_dict["end"]
        transect_strs = [start_str, break_str, resume_str, end_str]

        # Check that there are 4 unique transect strings
        if len(transect_strs) != len(set(transect_strs)):
            raise ValueError(
                "There exist duplicate values in transect_sequence_type_dict. "
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

        if not transect_df.empty:
            # Drop time duplicates, sort the dataframe by datetime, and reset Transect Dataframe Index.
            transect_df = (
                transect_df.drop_duplicates(subset=["region_bbox_left"])
                .sort_values(by="region_bbox_left")
                .reset_index()
            )

            # Create a new column which stores the transect_type without the transect number
            transect_df.loc[:, "transect_type"] = transect_df.loc[:, "region_name"].str.extract(
                rf"({start_str}|{break_str}|{resume_str}|{end_str})"
            )

            # Create new shifted columns with the next transect log type and next region
            # bbox left datetime value.
            transect_df.loc[:, "transect_type_next"] = transect_df.loc[:, "transect_type"].shift(-1)
            transect_df.loc[:, "region_bbox_left_next"] = transect_df.loc[
                :, "region_bbox_left"
            ].shift(-1)

            # Set transect_type_next values to be empty strings if they are NAs.
            transect_df["transect_type_next"] = transect_df.apply(
                lambda x: "" if isna(x["transect_type_next"]) else x["transect_type_next"],
                axis=1,
            )

            # Check transect sequences
            _check_transect_sequences(transect_df, must_pass_check)

            # Create binary variable indicating within transect segments.
            transect_df["within_transect"] = False

            # Indices where start_str followed by break_str/end_str
            st_indices = (transect_df["transect_type"] == start_str) & transect_df[
                "transect_type_next"
            ].isin(transect_sequence_type_next_allowable_dict[start_str])
            transect_df.loc[st_indices, "within_transect"] = True

            # Indices where resume_str followed by break_str/end_str
            rt_indices = (transect_df["transect_type"] == resume_str) & transect_df[
                "transect_type_next"
            ].isin(transect_sequence_type_next_allowable_dict[resume_str])
            transect_df.loc[rt_indices, "within_transect"] = True

            # Extract the min and max timestamps for filtering
            min_timestamp = da_Sv.ping_time.min().values
            max_timestamp = da_Sv.ping_time.max().values

            # Find the last of the row right before min_timestamp if it exists.
            # Else choose the minimum row.
            region_bbox_left_prior = transect_df.loc[
                transect_df["region_bbox_left"] < min_timestamp, "region_bbox_left"
            ]
            if region_bbox_left_prior.empty:
                last_index_before_min = transect_df["region_bbox_left"].idxmin()
            else:
                last_index_before_min = region_bbox_left_prior.idxmin()
            # Find the first index after max_timestamp if it exists.
            # Else choose the maximum row.
            region_bbox_right_after = transect_df.loc[
                transect_df["region_bbox_right"] > max_timestamp, "region_bbox_right"
            ]
            if region_bbox_right_after.empty:
                first_index_after_max = transect_df["region_bbox_right"].idxmax()
            else:
                first_index_after_max = region_bbox_right_after.idxmin()

            # Filter transect_df to get the within transect df
            within_transect_df = transect_df[
                (transect_df["within_transect"])
                & (transect_df.index >= last_index_before_min)
                & (transect_df.index <= first_index_after_max)
            ]
        else:
            # Create empty within transect df
            within_transect_df = pd.DataFrame()

        # Create within transect mask
        M = xr.zeros_like(da_Sv)
        for _, row in within_transect_df.iterrows():
            M = M + xr.where(
                (M.ping_time > row["region_bbox_left"])
                & (M.ping_time < row["region_bbox_left_next"]),
                1,
                0,
            )

        # If M contains channel dimension, then drop it.
        if "channel" in M.dims:
            M = M.isel(channel=0)

        return M
