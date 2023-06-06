import matplotlib
import numpy as np
import regionmask
import xarray as xr
from xarray import DataArray
from pandas import DataFrame
from typing import Union, NewType, List, Any

# Regions 2D Type Place Holder
Regions2DType = NewType('Regions2D', Any)

class Regions2DMasker:
    """Class for masking Regions. Should Only be used by Regions2D"""

    def __init__(self, Regions2D: Regions2DType):
        self.Regions2D = Regions2D
        self.Regions2D.replace_nan_depth(inplace=True)

    def mask(self, ds: DataArray, region_df: DataFrame, mask_var: str=None, 
            mask_labels: Union[List, str]=None, offset: Union[int, float]=0) -> DataArray:
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
        ds = ds.assign_coords(
            unix_time=(
                "ping_time",
                matplotlib.dates.date2num(ds.coords["ping_time"].values),
            )
        )

        # set up mask labels
        if mask_labels:
            if mask_labels == "from_ids":
                # create mask
                r = regionmask.Regions(outlines=regions_np, numbers=region_ids)
                M = r.mask(
                    ds,
                    lon_name="unix_time",
                    lat_name="depth",
                    wrap_lon=False,
                )

            elif isinstance(mask_labels, list):
                # create mask
                r = regionmask.Regions(outlines=regions_np)
                M = r.mask(
                    ds,
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
                ds,
                lon_name="unix_time",
                lat_name="depth",
                wrap_lon=False,
            )

        # assign specific name to mask array, otherwise 'mask'
        if mask_var:
            M = M.rename(mask_var)

        return M
