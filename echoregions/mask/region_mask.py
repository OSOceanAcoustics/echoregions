import numpy as np
import regionmask
import matplotlib


class Region2DMasker():
    """Class for masking Regions. Should Only be used by Regions2D"""
    def __init__(self, Regions2D):
        self.Regions2D = Regions2D

    def mask_region(self, ds, region, freq_idx=0, offset=0):
        points = np.array(self.Regions2D.convert_points(
            self.Regions2D.get_points_from_region(region),
            convert_time=True,
            convert_range_edges=True,
            offset=offset,
            unix=True
        ))
        points = self.Regions2D.close_region(points)
        ds = ds.assign_coords(unix_time=('ping_time', matplotlib.dates.date2num(ds.coords['ping_time'].values)))
        ds['range'] = ds.range.isel(frequency=0, ping_time=0)
        M = regionmask.Regions([points])
        return M.mask(ds.isel(frequency=freq_idx), lon_name='unix_time', lat_name='range')
