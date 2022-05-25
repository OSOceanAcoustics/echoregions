import matplotlib
import numpy as np
import pandas as pd
import regionmask
import xarray as xr


class Regions2DMasker:
    """Class for masking Regions. Should Only be used by Regions2D"""

    def __init__(self, Regions2D):
        self.Regions2D = Regions2D
        self.Regions2D.replace_nan_depth(inplace=True)

    def mask(self, ds, region_df, data_var="Sv", mask_var=None, mask_labels=None, offset=0):
        # Collect points that make up the region
        # points = [
        #    list(item)
        #    for item in zip(list(region_df["time"]), list(region_df["depth"]))
        #]

        region_df = region_df[["region_id", "time", "depth"]]

        # organize the regions in a format for region mask
        df = region_df.apply(pd.Series.explode)

        # convert region time to integer timestamp
        df['time'] = matplotlib.dates.date2num(df['time'])
  
        # create a list of dataframes for each regions
        grouped = list(df.groupby("region_id"))
  
        # convert to list of numpy arrays which is an acceptable format to create region mask
        regions_np = [np.array(region[["time", "depth"]]) for id, region in grouped]

        # corresponding region ids converted to int
        region_ids = [int(id) for id, region in grouped]

        # points = self.Regions2D.convert_points(
        #    points,
        #    convert_time=True,
        #    convert_depth_edges=False,
        #    offset=offset,
        #    unix=True,
        #)

        # Convert ping_time to unix_time since the masking does not work on datetime objects
        ds = ds.assign_coords(
            unix_time=(
                "ping_time",
                matplotlib.dates.date2num(ds.coords["ping_time"].values),
            )
        )
        # Select range in one dimension
        if ds["range"].ndim == 3:
            ds["range"] = ds.range.isel(frequency=0, ping_time=0)
        if "range_bin" in ds.dims:
            ds = ds.swap_dims({"range_bin": "range"})

        # set up mask labels
        if mask_labels:
            if mask_labels=="from_ids":
                r = regionmask.Regions(regions_np, numbers=region_ids)
            elif isinstance(mask_labels, list):
                r = regionmask.Regions(regions_np, numbers=mask_labels)
            else:
                ValueError("mask_labels must be None, 'from_ids', or a list.")
        
        # Initialize mask object
        # TODO: make selection of frequency outside
        M = r.mask(
            ds[data_var].isel(frequency=0),
            lon_name="unix_time",
            lat_name="depth",
            wrap_lon=False,
        )

        # assign specific name to mask array, otherwise 'mask'
        if mask_var:
            M = M.rename(mask_var)
        M = M.drop("frequency")
        return M
