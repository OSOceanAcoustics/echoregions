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
from .regions2d_parser import parse_evr, parse_regions_df


def _check_transect_sequences(
    transect_df: pd.DataFrame,
    transect_sequence_type_next_allowable_dict: dict,
    bbox_distance_threshold: float,
    must_pass_check: bool,
) -> None:
    """
    Checking of transect sequences in the Regions2d transect dataframe.

    Parameters
    ----------
    transect_df : pd.DataFrame
        Inner Regions2d transect dataframe.
    transect_sequence_type_next_allowable_dict : dict
        Dictionary for the allowable transect sequence type value(s) that can follow a
        transect sequence type value.
    bbox_distance_threshold: float
        Maximum allowable value between the left and right bounding box timestamps
        for each region that is marked as a transect log. Default is set to 1 minute.
    must_pass_check : bool
        True: Will check transect strings to enforce sequence rules. If this check
        encounters any incorrect transect type sequence orders or wider than bbox distance
        threshold regions, it will raise an exception.
        False: Will still check transect strings but will instead just print out warnings
        for violations of the above mentioned sequence rules.
    """
    # Create an empty list to collect error messages.
    warning_messages = []

    # Ensure correct sequence of transect types occur.
    # If they do not, append to warning_messages.
    for _, row in transect_df.iterrows():
        transect_type = row["transect_type"]
        transect_type_next = row["transect_type_next"]
        # Check for correct transect_type_next values
        if transect_type_next not in transect_sequence_type_next_allowable_dict[transect_type]:
            type_next_warning_message = (
                f"Error in region_id {row['region_id']}:"
                f"Transect string {transect_type} is followed by "
                f"invalid value {transect_type_next}. Must be followed by "
                f"{transect_sequence_type_next_allowable_dict[transect_type]}"
            )
            warning_messages.append(type_next_warning_message)

    # Identify rows wider than bbox distance threshold if they exist
    wider_than_bbox_distance_threshold_rows = transect_df[
        (transect_df["region_bbox_right"] - transect_df["region_bbox_left"]).dt.total_seconds() / 60
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

        # Save to CSV
        self.data.to_csv(save_path, mode=mode, **kwaargs)

        # Append save_path
        self.output_file.append(save_path)

    def _write_region(self, row, save_path):
        """Helper function to write individual regions to `.evr` file."""
        # Set region variables
        datetimes = pd.to_datetime(row["time"])
        depths = row["depth"]
        region_structure_version = row["region_structure_version"]
        point_count = str(len(datetimes))
        selected = "0"
        region_creation_type = row["region_creation_type"]
        dummy = "-1"
        bounding_rectangle_calculated = "1"
        region_notes = row["region_notes"]
        number_of_lines_of_notes = len(region_notes)
        number_of_lines_of_detection_settings = "0"
        region_class = row["region_class"]
        region_type = row["region_type"]
        region_name = row["region_name"]
        region_id = row["region_id"]

        # Append to existing `.evr` file
        with open(save_path, "a") as f:
            # Calculate bounding box
            left_x = (
                str(min(datetimes).strftime("%Y%m%d"))
                + " "
                + str(min(datetimes).strftime("%H%M%S%f"))
            )
            top_y = str(min(depths))
            right_x = (
                str(max(datetimes).strftime("%Y%m%d"))
                + " "
                + str(max(datetimes).strftime("%H%M%S%f"))
            )
            bottom_y = str(max(depths))
            bbox = left_x + " " + top_y + " " + right_x + " " + bottom_y

            # Write first line
            f.write(
                "\n"
                + region_structure_version
                + " "
                + point_count
                + " "
                + str(region_id)
                + " "
                + selected
                + " "
                + region_creation_type
                + " "
                + dummy
                + " "
                + bounding_rectangle_calculated
                + " "
                + bbox
                + "\n"
            )
            f.write(str(number_of_lines_of_notes) + "\n")
            if number_of_lines_of_notes > 0:
                for region_note in region_notes:
                    f.write(region_note + "\n")
            f.write(number_of_lines_of_detection_settings + "\n")
            f.write(region_class + "\n")

            # Write points
            for datetime, depth in zip(datetimes, depths):
                date = str(datetime.strftime("%Y%m%d"))
                time = str(datetime.strftime("%H%M%S%f"))[:-2]
                depth = str(depth)
                point = date + " " + time + " " + depth
                f.write("%s " % point)
            f.write(region_type + "\n")
            f.write(region_name + "\n")
        f.close()

    def to_evr(self, save_path: Union[str, Path], mode: str = "w") -> None:
        """Save a Dataframe to a .evr file

        Parameters
        ----------
        save_path : Union[str, Path]
            Path to save the `evr` file to.
        mode : str
            Write mode arg for IO open. Defaults to 'w'.
        """
        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext=".evr")

        # Grab header information
        echoview_version = (
            f"EVRG 7 {self.data.iloc[0]['echoview_version']}"
            if len(self.data) > 0
            else "EVRG 7 12.0.341.42620"
        )
        number_of_regions = str(len(self.data))

        # Write header to `.evr`
        with open(save_path, mode=mode) as f:
            f.write(echoview_version + "\n")
            f.write(number_of_regions + "\n")
        f.close()

        # Write each region to `.evr`
        for _, row in self.data.iterrows():
            self._write_region(row, save_path=save_path)

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
        region_class: Union[str, List[str]] = None,
        time_range: List[Timestamp] = None,
        depth_range: List[Union[float, int]] = None,
        copy=True,
    ) -> DataFrame:
        """Selects a subset of this Region2D object's dataframe.

        Parameters
        ----------
        region_id : Union[float, int, str, List[Union[float, int, str]]], ``None``
            A region id provided as a number, a string, or a list of these.
            Only one of ``region_id`` or ``region_class`` should be given.
        region_class : Union[str, List[str]], ``None``
            A region class or a list of region classes.
            Only one of ``region_id`` or ``region_class`` should be given.
        time_range : List of 2 Pandas Timestamps.
            Datetime range for the expected output of subselected DataFrame. 1st
            index value must be later than 0th index value.
        depth_range : List of 2 floats.
            Depth range for the expected output of subselected DataFrame. 1st
            index value must be larger than 0th index value.
        copy : bool
            Return a copy of the `data` DataFrame

        Returns
        -------
        DataFrame
            The filtered Region2D dataframe (``Region2D.data``) that
            either contains rows of the specified ``region_id``,
            or rows of the specified ``region_class``.
            The Region2D dataframe is also filtered to be within the boundaries
            specified by the input ``time_range``, and ``depth_range`` values.
        """
        # Check that at least one of either region_class or region_id are None.
        if region_id and region_class:
            raise ValueError(
                "Only one of region_id or region_class should be non-NaN. "
                "If the user wishes to select a specific region_id with a specific "
                "region_class they should just pass in the region_id."
            )

        # Make copy of the original dataframe; else, use the original dataframe in selection.
        region = self.data.copy() if copy else self.data

        # Check and subset for region_id
        if region_id is not None:
            if isinstance(region_id, (float, int, str)):
                region_id = [region_id]
            elif isinstance(region_id, list):
                if len(region_id) == 0:
                    raise ValueError("region_id list is empty. Cannot be empty.")
                for value in region_id:
                    if not isinstance(value, (float, int, str)):
                        raise TypeError(
                            f"Invalid element in list region_id of type: "
                            f"{type(value)}. Must be of type float, int, str."
                        )
            else:
                raise TypeError(
                    f"Invalid region_id type: {type(region_id)}. "
                    "Must be of type float, int, str, list, or None."
                )
            region = region[region["region_id"].isin(region_id)]

        # Check and subset for region_class
        if region_class is not None:
            if isinstance(region_class, str):
                region_class = [region_class]
            elif isinstance(region_class, list):
                for value in region_class:
                    if not isinstance(value, str):
                        raise TypeError(
                            f"Invalid element in list region_class of type: "
                            f"{type(value)}. Must be of type str."
                        )
            else:
                raise TypeError(
                    f"Invalid region_class type: {type(region_class)}. "
                    "Must be of type str, list, or None."
                )
            region = region[region["region_class"].isin(region_class)]

        # Check and subset for time_range
        if time_range is not None:
            if not isinstance(time_range, list):
                raise TypeError("Invalid time_range type. It must be a list.")
            if len(time_range) != 2:
                raise ValueError("Invalid time_range size. It must be a list of size 2.")
            if not all(isinstance(t, Timestamp) for t in time_range):
                raise TypeError(
                    "Invalid time_range format. It must be a list of 2 Pandas Timestamps."
                )
            if time_range[0] >= time_range[1]:
                raise ValueError(
                    "1st index value must be later than 0th index value. "
                    f"Currently 0th index value is {time_range[0]} "
                    f"and 1st index value is {time_range[1]}"
                )
            region = region[
                region["time"].apply(
                    lambda time_array: all(
                        time_range[0] <= Timestamp(x) <= time_range[1] for x in time_array
                    )
                )
            ]

        # Check and subset for depth_range
        if depth_range is not None:
            if not isinstance(depth_range, list):
                raise TypeError("Invalid depth_range type. It must be a list.")
            if len(depth_range) != 2:
                raise ValueError("Invalid depth_range size. It must be a list of size 2.")
            if not all(isinstance(d, (float, int)) for d in depth_range):
                raise TypeError(
                    "Invalid depth_range format. It must be a list of 2 floats or ints."
                )
            if depth_range[0] >= depth_range[1]:
                raise ValueError(
                    f"1st index value must be later than 0th index value. Currently "
                    f"0th index value is {depth_range[0]} and 1st index value is "
                    f"{depth_range[1]}."
                )
            region = region[
                region["time"].apply(
                    lambda depth_array: all(
                        depth_range[0] <= float(x) <= depth_range[1] for x in depth_array
                    )
                )
            ]

        return region

    def close_region(
        self,
        region_id: Union[float, int, str, List[Union[float, int, str]]] = None,
        region_class: Union[str, List[str]] = None,
    ) -> DataFrame:
        """Close a region by appending the first point to end of the list of points.

        Parameters
        ----------
        region_id : Union[float, int, str, List[Union[float, int, str]]], ``None``
            region(s) to select raw files with
            If ``None``, select all regions. Defaults to ``None``
        region_class : Union[str, List[str]], ``None``
            A region class or a list of region classes.

        Returns
        -------
        DataFrame
            Returns a new DataFrame with closed regions
        """
        region = self.select_region(region_id, region_class, copy=True)
        region["time"] = region.apply(lambda row: np.append(row["time"], row["time"][0]), axis=1)
        region["depth"] = region.apply(lambda row: np.append(row["depth"], row["depth"][0]), axis=1)
        return region

    def select_sonar_file(
        self,
        Sv_list: Union[xr.DataArray, xr.Dataset, List[Union[xr.DataArray, xr.Dataset]]],
        time_variable: str = "ping_time",
        region_id: Union[float, int, str, List[Union[float, int, str]]] = None,
        region_class: Union[str, List[str]] = None,
    ) -> List:
        """
        Selects Echopype processed Sv files (xarray datasets) based on variable
        start and end time.

        Parameters
        ----------
        Sv_list : Union[xr.DataArray, xr.Dataset, List[xr.DataArray, xr.Dataset]]
            Echopype processed Sv files.
        time_variable : str
            Time variable for Sv files.
            Defaults to 'ping_time'.
        region_id : Union[float, int, str, List[Union[float, int, str]]], ``None``
            Region IDs to select sonar files with.
            If ``None``, select all regions. Defaults to ``None``
        region_class : Union[str, List[str]], ``None``
            A region class or a list of region classes.

        Returns
        -------
        selected_Sv : list[Union[xr.DataArray, xr.Dataset]]
            list of Sv data spanning the encompassing region or list of regions.
        """
        # Check Sv files type
        if not isinstance(Sv_list, list):
            if isinstance(Sv_list, (xr.DataArray, xr.Dataset)):
                Sv_list = [Sv_list]
            else:
                raise TypeError(
                    f"Sv_list is of type {type(Sv_list)}. "
                    "Must be of type Union[xr.DataArray, xr.Dataset] or "
                    "List[Union[xr.DataArray, xr.Dataset]]."
                )
        else:
            for Sv in Sv_list:
                # Check if each element is an xarray DataArray or Dataset
                if not isinstance(Sv, (xr.DataArray, xr.Dataset)):
                    raise TypeError(
                        f"Element {Sv} in Sv_list is of type {type(Sv)}. "
                        "All elements must be of type Union[xr.DataArray, xr.Dataset]."
                    )

        # Select region(s)
        region = self.select_region(region_id, region_class)

        # Extract region time values
        region_times = np.hstack(region["time"].values)

        # Initialize empty list to store selected Sv
        selected_Sv = []

        for Sv in Sv_list:
            # Get Sv time variable
            Sv_time_var = Sv[time_variable]
            if len(Sv_time_var.data) != 0:
                # Get the time range from the xarray dataset
                Sv_time_min = pd.to_datetime(Sv_time_var.min().item())
                Sv_time_max = pd.to_datetime(Sv_time_var.max().item())

                # Check if the time range overlaps with region times
                if Sv_time_max >= region_times.min() and Sv_time_min <= region_times.max():
                    selected_Sv.append(Sv)

        return selected_Sv

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
        region_id: Union[float, int, str, List[Union[float, int, str]]] = None,
        region_class: Union[str, List[str]] = None,
        close_regions: bool = False,
        **kwargs,
    ) -> None:
        """Plot a region from data.
        Automatically convert time and range_edges.

        Parameters
        ---------
        region_id : Union[float, int, str, List[Union[float, int, str]]], ``None``
            Region ID(s) to select raw files with
        region_class : Union[str, List[str]], ``None``
            A region class or a list of region classes.
        close_region : bool
            Plot the region as a closed polygon. Defaults to False
        kwargs : keyword arguments
            Additional arguments passed to matplotlib plot
        """

        # Select Region(s)
        if close_regions:
            region = self.close_region(region_id, region_class)
        else:
            region = self.select_region(region_id, region_class)
        for _, row in region.iterrows():
            plt.plot(row["time"], row["depth"], **kwargs)

    def region_mask(
        self,
        da_Sv: DataArray,
        region_id: Union[float, int, str, List[Union[float, int, str]]] = None,
        region_class: Union[str, List[str]] = None,
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
        region_id : List[Union[float, int, str]]], ``None``
            list IDs of regions to create mask for.
        region_class : Union[str, List[str]], ``None``
            A region class or a list of region classes.
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
        region_points : pd.DataFrame
            DataFrame containing region_id, depth, and time.
        """
        if not isinstance(da_Sv, DataArray):
            raise TypeError(
                f"Input da_Sv must be of type DataArray. da_Sv was instead of type {type(da_Sv)}"
            )

        # Dataframe containing region information.
        region_df = self.select_region(region_id, region_class)

        # Drop channel if it exists
        if "channel" in da_Sv.dims:
            da_Sv = da_Sv.isel(channel=0).drop_vars("channel")

        # Compute valid boundaries
        min_depth_valid = float(da_Sv["depth"].min())
        max_depth_valid = float(da_Sv["depth"].max())
        min_time_valid = da_Sv["ping_time"].min()
        max_time_valid = da_Sv["ping_time"].max()
        region_depth_lower = max(0, int(self.min_depth))
        region_depth_upper = int(self.max_depth)

        # Apply filters using inline lambdas
        region_df = region_df[
            region_df["depth"].apply(
                lambda depths: any(min_depth_valid <= d <= max_depth_valid for d in depths)
            )
            & region_df["time"].apply(
                lambda times: any(min_time_valid <= t <= max_time_valid for t in times)
            )
            & region_df["depth"].apply(
                lambda depths: all(region_depth_lower <= d <= region_depth_upper for d in depths)
            )
        ]

        if region_df.empty:
            warnings.warn(
                "All rows in Regions DataFrame have NaN Depth values after filtering depth "
                "between min_depth and max_depth.",
                UserWarning,
                stacklevel=2,
            )
            mask_ds = xr.Dataset()
            if collapse_to_2d:
                mask_ds["mask_2d"] = xr.full_like(da_Sv, np.nan)

            else:
                mask_ds["mask_3d"] = xr.zeros_like(da_Sv).expand_dims(
                    {"region_id": ["dummy_region_id"]}
                )
            return mask_ds, region_df.copy()
        else:
            # Grab subset region ids
            subset_region_ids = region_df.region_id.astype(int).to_list()

            if mask_labels is None:
                # Create mask_labels with each subset_region_ids as a key and values starting from 0
                mask_labels = {key: idx for idx, key in enumerate(subset_region_ids)}

            # Check that subset_region_ids and mask_labels are matching
            if len(set(subset_region_ids) - set(mask_labels.keys())) > 0:
                raise ValueError(
                    "Each value in subset_region_ids must be a key in 'mask_labels' and vice versa. "
                    "If you would prefer 0 based indexing as values for mask_labels, leave "
                    "mask_labels as None."
                )
            # Select only important columns
            region_df = region_df[["region_id", "time", "depth"]]

            # Organize the regions in a format for region mask.
            df = region_df.explode(["time", "depth"])

            # Convert region time to integer timestamp.
            df["time"] = matplotlib.dates.date2num(df["time"])

            # Create a list of dataframes for each regions.
            grouped = list(df.groupby("region_id"))

            # Convert to list of numpy arrays which is an acceptable format to create region mask.
            regions_np = [np.array(region[["time", "depth"]]) for _, region in grouped]

            # Convert corresponding region_id to int.
            filtered_region_ids = [int(id) for id, _ in grouped]

            # Convert ping_time to unix_time since the masking does not work on datetime objects.
            da_Sv = da_Sv.assign_coords(
                unix_time=(
                    "ping_time",
                    matplotlib.dates.date2num(da_Sv.coords["ping_time"].values),
                )
            )

            # Create regionmask object
            regionmask_regions = regionmask.Regions(
                outlines=regions_np, numbers=filtered_region_ids, name=mask_name, overlap=True
            )

            if da_Sv.chunksizes:
                # Define a helper function to operate on individual blocks
                def _region_mask_block(
                    da_Sv_block, wrap_lon, regionmask_regions, filtered_region_ids
                ):
                    # Grab time and depth blocks
                    unix_time_block = da_Sv_block["unix_time"]
                    depth_block = da_Sv_block["depth"]

                    # Set the filter to ignore the specific warnings
                    # No grid point warning will show up a lot with smaller chunks and
                    warnings.filterwarnings("ignore", message="No gridpoint belongs to any region")
                    # TODO Write issue in regionmask repo to convince them not to remove method as an argument
                    warnings.filterwarnings(
                        "ignore",
                        message="The ``method`` argument is internal and  will be removed in the future",
                        category=FutureWarning,
                    )
                    mask_block_da = (
                        regionmask_regions.mask_3D(
                            unix_time_block, depth_block, wrap_lon=wrap_lon, method="shapely"
                        )
                        .astype(int)
                        .drop_vars(["abbrevs", "names"])
                    )

                    # Reindex to fill in missing filtered region IDs
                    mask_block_da = mask_block_da.reindex(region=filtered_region_ids, fill_value=0)

                    return mask_block_da

                # Apply _mask_block over the blocks of the input array
                mask_da = xr.map_blocks(
                    _region_mask_block,
                    da_Sv,
                    kwargs={
                        "wrap_lon": False,
                        "regionmask_regions": regionmask_regions,
                        "filtered_region_ids": filtered_region_ids,
                    },
                ).chunk({"region": 1})

            else:
                # Create mask
                mask_da = regionmask_regions.mask_3D(
                    da_Sv["unix_time"],
                    da_Sv["depth"],
                    wrap_lon=False,
                ).astype(
                    int
                )  # This maps False to 0 and True to 1

            # Drop unused attributes
            mask_da.attrs.pop("standard_name")

            # Rename region coords with region_id coords
            mask_da = mask_da.rename({"region": "region_id"})

            # Remove all coords other than region_id, depth, ping_time
            mask_da = mask_da.drop_vars(
                mask_da.coords._names.difference({"region_id", "depth", "ping_time"})
            )

            # Transpose mask_da
            mask_da = mask_da.transpose("region_id", "depth", "ping_time")

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

            # Get region_points
            region_points = region_df[region_df["region_id"].isin(masked_region_id)][
                ["region_id", "depth", "time"]
            ]

            return mask_ds, region_points

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
            for each region that is marked as a transect log. Default is set to 1 minute.
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
            _check_transect_sequences(
                transect_df,
                transect_sequence_type_next_allowable_dict,
                bbox_distance_threshold,
                must_pass_check,
            )

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

            # Find the last index right before min_timestamp if it exists.
            # Else choose the minimum row.
            region_bbox_left_prior = transect_df.loc[
                transect_df["region_bbox_left"] < min_timestamp, "region_bbox_left"
            ]
            if region_bbox_left_prior.empty:
                last_index_before_min = transect_df["region_bbox_left"].idxmin()
            else:
                last_index_before_min = region_bbox_left_prior.idxmax()
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
