import numpy as np
import regionmask
import matplotlib


class Region2DMasker():
    def __init__(self, Regions2D):
        self.Regions2D = Regions2D

    def mask_region(self, ds, region, offset=0):
        points = self.Regions2D.convert_points(
            self.Regions2D.get_points_from_region(region),
            convert_time=True,
            convert_range_edges=True,
            offset=offset,
            unix=True
        )
        points = []
        ds = ds.assign_coords(unix_time=('ping_time', matplotlib.dates.date2num(ds.coords['ping_time'].values)))
        ds['range'] = ds.range.isel(frequency=0, ping_time=0)
        # ds = ds.swap_dims({'range_bin': 'range'})
        M = regionmask.Regions([points])
        return M.mask(ds.isel(frequency=0), lon_name='unix_time', lat_name='range')
