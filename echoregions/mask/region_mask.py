import matplotlib
import numpy as np
import regionmask
import xarray as xr


class Regions2DMasker:
    """Class for masking Regions. Should Only be used by Regions2D"""

    def __init__(self, Regions2D):
        self.Regions2D = Regions2D
        self.Regions2D.replace_nan_depth(inplace=True)

    def mask(self, ds, region, data_var="Sv", mask_var=None, offset=0):
        # Collect points that make up the region
        points = [list(item) for item in zip(list(region['time'].iloc[0]), list(region['depth'].iloc[0]))]
        
        points = self.Regions2D.convert_points(
                points,
                convert_time=True,
                convert_depth_edges=False,
                offset=offset,
                unix=True,
            )
        
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
        
        # Initialize mask object
        r = regionmask.Regions([points])
        #TODO: make selection of frequency outside
        M = r.mask(ds[data_var].isel(frequency=0), lon_name="unix_time", lat_name="depth", wrap_lon=False)
        
        # assign specific name to mask array, otherwise 'mask'
        if mask_var:
            M = M.rename(mask_var)
        M = M.drop('frequency')
        return(M)