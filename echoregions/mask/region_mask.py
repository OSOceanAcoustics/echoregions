import numpy as np
import xarray as xr
import regionmask
import matplotlib


class Regions2DMasker():
    """Class for masking Regions. Should Only be used by Regions2D"""
    def __init__(self, Regions2D):
        self.Regions2D = Regions2D

    def mask(self, ds, region, data_var='Sv', offset=0):
        # Collect points that make up the region
        points = np.array(self.Regions2D.convert_points(
            self.Regions2D.get_points_from_region(region),
            convert_time=True,
            convert_depth_edges=True,
            offset=offset,
            unix=True
        ))
        points = self.Regions2D.close_region(points)
        # Convert ping_time to unix_time since the masking does not work on datetime objects
        ds = ds.assign_coords(unix_time=('ping_time', matplotlib.dates.date2num(ds.coords['ping_time'].values)))
        # Select range in one dimension
        if ds['range'].ndim == 3:
            ds['range'] = ds.range.isel(frequency=0, ping_time=0)
        if 'range_bin' in ds.dims:
            ds = ds.swap_dims({'range_bin': 'range'})
        # Initialize mask object
        M = regionmask.Regions([points])
        masked_da = []
        # Loop over frequencies since regionmask only masks in 2 dimensions
        for freq in ds[data_var].transpose('frequency', ...):
            masked_da.append(M.mask(
                freq, lon_name='unix_time', lat_name='range'
            ).assign_coords(frequency=freq.frequency))
        masked_da = xr.concat(masked_da, 'frequency')
        return ds.assign({data_var: masked_da})
