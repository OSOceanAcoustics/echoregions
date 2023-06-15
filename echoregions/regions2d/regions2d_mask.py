from typing import List
import matplotlib
import numpy as np
import regionmask
import xarray as xr
from xarray import DataArray, Dataset

from .regions2d import Regions2D

def regions2d_mask(ds_Sv: DataArray, regions2d: Regions2D, 
                   region_ids: List , mask_var: str = None, mask_labels=None) -> DataArray:
    """Mask an xarray DataArray.

    Parameters
    ----------
    ds : Xarray DataArray
        calibrated data (Sv or Sp) with range
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
    A DataArray with the data_var masked by the specified region
    """

    if type(regions2d) != Regions2D:
        raise TypeError(f"regions2d should be of type Regions2D. regions2d is currently of \
                        type {type(regions2d)}")
    
    if isinstance(mask_labels, list) and (len(mask_labels) != len(region_ids)):
        raise ValueError("If mask_labels is a list, it should be of same length as region_ids.")

    if not isinstance(ds_Sv, DataArray):
        raise TypeError(f"Invalid ds Type: {type(ds_Sv)}. Must be of type DataArray")

    regions2d.replace_nan_depth(inplace=True)

     # dataframe containing region information
    region_df = regions2d.select_region(region_ids)
    
    # select only columns which are important
    region_df = region_df[["region_id", "time", "depth"]]

    # organize the regions in a format for region mask
    df = region_df.explode(["time", "depth"])

    # convert region time to integer timestamp
    df["time"] = matplotlib.dates.date2num(df["time"])

    # create a list of dataframes for each regions
    grouped = list(df.groupby("region_id"))

    # convert to list of numpy arrays which is an acceptable format to create region mask
    regions_np = [np.array(region[["time", "depth"]]) for id, region in grouped]

    # corresponding region ids converted to int
    region_ids = [int(id) for id, region in grouped]

    # Convert ping_time to unix_time since the masking does not work on datetime objects
    ds_Sv = ds_Sv.assign_coords(
        unix_time=(
            "ping_time",
            matplotlib.dates.date2num(ds_Sv.coords["ping_time"].values),
        )
    )

    # set up mask labels
    if mask_labels:
        if mask_labels == "from_ids":
            # create mask
            r = regionmask.Regions(outlines=regions_np, numbers=region_ids)
            M = r.mask(
                ds_Sv,
                lon_name="unix_time",
                lat_name="depth",
                wrap_lon=False,
            )

        elif isinstance(mask_labels, list):
            # create mask
            r = regionmask.Regions(outlines=regions_np)
            M = r.mask(
                ds_Sv,
                lon_name="unix_time",
                lat_name="depth",
                wrap_lon=False,
            )
            # convert default labels to mask_labels
            S = xr.where(~M.isnull(), 0, M)
            S = M
            for idx, label in enumerate(mask_labels):
                S = xr.where(M == idx, label, S)
            M = S
        else:
            raise ValueError("mask_labels must be None, 'from_ids', or a list.")
    else:
        # create mask
        r = regionmask.Regions(outlines=regions_np)
        M = r.mask(
            ds_Sv,
            lon_name="unix_time",
            lat_name="depth",
            wrap_lon=False,
        )

    # assign specific name to mask array, otherwise 'mask'
    if mask_var:
        M = M.rename(mask_var)

    return M
