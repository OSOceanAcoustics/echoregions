from typing import List

import matplotlib
import numpy as np
import regionmask
import xarray as xr
from xarray import DataArray

from .regions2d import Regions2D


def regions2d_mask(
    da_Sv: DataArray,
    regions2d: Regions2D,
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
    regions2d : Regions2D
        Regions2D Object containing polygons corresponding to different regions.
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

    if type(regions2d) != Regions2D:
        raise TypeError(
            f"regions2d should be of type Regions2D. regions2d is currently of \
                        type {type(regions2d)}"
        )

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
    regions2d.replace_nan_depth(inplace=True)

    # Dataframe containing region information.
    region_df = regions2d.select_region(region_ids)

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
                da_Sv,
                lon_name="unix_time",
                lat_name="depth",
                wrap_lon=False,
            )

        elif isinstance(mask_labels, list):
            # Create mask.
            r = regionmask.Regions(outlines=regions_np)
            M = r.mask(
                da_Sv,
                lon_name="unix_time",
                lat_name="depth",
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
        M = r.mask(
            da_Sv,
            lon_name="unix_time",
            lat_name="depth",
            wrap_lon=False,
        )

    # Assign specific name to mask array, otherwise 'mask'.
    if mask_var:
        M = M.rename(mask_var)

    return M

def convert_mask_2d_to_3d(M: DataArray):
    """Convert 2D Mask data into its 3D one-hot encoded form.

    Parameters
    ----------
    M : Data Array
        A DataArray with the data_var masked by a specified region.

    Returns
    -------
    A Dataset with a 3D DataArray with the data_var masked by the specified
    region in one-hot encoded form and a dictionary to remember original
    non-nan values.
    """
    # Get unique non nan values from the 2d mask
    unique_non_nan = list(np.unique(M.data[~np.isnan(M.data)]))

    # Create a list of mask objects from one-hot encoding M.data non-nan values
    # and a dictionary to remember said values from one-hot encoded data arrays.
    mask_list = []
    mask_dictionary = {"dims": "label", "data": []}
    for index, value in enumerate(unique_non_nan):
        mask_data = M.copy()
        mask_data_np = mask_data.data
        for index in np.ndindex(mask_data_np.shape):
            if mask_data_np[index] == value:
                mask_data_np[index] = 1
            else:
                mask_data_np[index] = 0
        mask_data.data = mask_data_np
        mask_list.append(mask_data)
        mask_dictionary_list = mask_dictionary["data"]
        mask_dictionary_list.append(value)
        mask_dictionary["data"] = mask_dictionary_list

    # Initialize empty Dataset
    mask_3d_ds = xr.Dataset()

    # Place one-hot encoded masks and dictionary in dataset
    mask_3d_da = xr.concat(mask_list, dim="label")
    mask_3d_ds["mask_3d"] = mask_3d_da
    mask_dictionary_da = xr.DataArray.from_dict(mask_dictionary)
    mask_3d_ds["mask_dictionary"] = mask_dictionary_da

    return mask_3d_ds

def convert_mask_3d_to_2d(mask_3d_ds: DataArray):
    """Convert 2D Mask data into its 3D one-hot encoded form.

    Parameters
    ----------
    mask_3d_ds : Dataset
        A Dataset with a 3D DataArray with the data_var masked by the specified
        region in one-hot encoded form and a dictionary to remember original
        non-nan values.

    Returns
    -------
    mask_2d_da: Data Array
        A 2D Data Array with appropriate mask values from mask_3d_ds.
    """
    # Get unique non nan values from the 2d mask
    unique_non_nan= list(mask_3d_ds.mask_dictionary.data)

    # Create copies and placeholder values for 2D and 3D mask objects
    mask_3d_da = mask_3d_ds.mask_3d.copy()
    np_mask_3d_da = mask_3d_da.data
    np_mask_2d_da = None

    # Iterate through 3D numpy array layers and set 1.0 to associated label values
    # dependent on which layer is being worked on, and sequentially add layers
    # together to form 2D numpy mask array.
    for index, label_value in enumerate(unique_non_nan):
        label_layer = np_mask_3d_da[index]
        label_layer[label_layer == 1.0] = label_value
        if index == 0:
            np_mask_2d_da = label_layer
        else:
            np_mask_2d_da = label_layer + np_mask_2d_da

    # Convert all such 0.0 values in 2D numpy mask array into NaN values.
    np_mask_2d_da[np_mask_2d_da == 0.0] = np.nan

    # Set new 2D mask data array
    mask_2d_da = xr.DataArray(
        data=np_mask_2d_da,
        dims=["depth", "ping_time"],
        coords=mask_3d_da.coords
    )

    return mask_2d_da
