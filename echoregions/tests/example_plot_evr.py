import os
import numpy as np
import xarray as xr
import echopype as ep
import matplotlib.pyplot as plt
from ..formats import Regions2D
from ..convert.evr_parser import Region2DParser
from ..convert.ecs_parser import CalibrationParser
from ..convert.evl_parser import LineParser


data_dir = './echoregions/test_data/ek60/'
output_csv = data_dir + 'output_CSV/'
output_json = data_dir + 'output_JSON/'
raw_file = 'Summer2017-D20170625-T195927.raw'
raw_files = [data_dir + 'hake_nc/' + f for f in os.listdir(data_dir + 'hake_nc')]

def test_region_plot():
    # Test plotting a region on top of an echogram

    # Parse region file
    evr_paths = data_dir + 'x1.evr'
    regions = Regions2D(evr_paths)
    regions.parse_file()
    raw = regions.select_raw(raw_files, 11)


    # Calibrate raw
    ed = ep.process.EchoData(raw)
    proc = ep.process.Process('EK60', ed)
    proc.get_Sv(ed)
    plat = xr.open_dataset(raw, group='Platform')
    water_level = plat.water_level[0, 0].values

    ed.Sv['range'] = ed.Sv.range.isel(frequency=0, ping_time=0).load()
    ed.Sv = ed.Sv.swap_dims({'range_bin': 'range'})
    # Plot Sv for one frequency
    ed.Sv.Sv.isel(frequency=0).plot(x='ping_time', vmax=-40, vmin=-100, yincrease=False)

    # Plot region
    regions.raw_range = ed.Sv['range']
    regions.convert_output()
    regions.plot_region(11, offset=water_level)
    plt.show()

def test_plot_multi():
    # Test plot all regions
    # Calibrate raw
    ed = ep.process.EchoData(data_dir + 'hake_nc/Summer2017-D20170625-T195927.nc')
    proc = ep.process.Process('EK60', ed)
    proc.get_Sv(ed)
    plat = xr.open_dataset(data_dir + 'hake_nc/Summer2017-D20170625-T195927.nc', group='Platform')
    water_level = plat.water_level[0, 0].values

    ed.Sv['range'] = ed.Sv.range.isel(frequency=0, ping_time=0).load()
    ed.Sv = ed.Sv.swap_dims({'range_bin': 'range'})
    # Plot Sv for one frequency
    ed.Sv.Sv.isel(frequency=0).plot(x='ping_time', vmax=-40, vmin=-100, yincrease=False)

    # Parse region file
    evr_paths = data_dir + 'x1.evr'
    regions = Regions2D(evr_paths)
    regions.raw_range = ed.Sv['range']
    regions.parse_file(convert_range_edges=True)

    for region in regions:
        regions.plot_region(region, offset=water_level)

    plt.show()
