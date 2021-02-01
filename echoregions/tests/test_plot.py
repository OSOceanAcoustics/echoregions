import os
import numpy as np
import xarray as xr
import echopype as ep
import matplotlib.pyplot as plt
from ..formats import Region2D
from ..convert.evr_parser import Region2DParser
from ..convert.ecs_parser import CalibrationParser
from ..convert.evl_parser import LineParser


data_dir = './echoregions/test_data/ek60/'
output_csv = data_dir + 'output_CSV/'
output_json = data_dir + 'output_JSON/'
raw_file = 'Summer2017-D20170625-T195927.raw'


def test_region_plot():
    # Convert raw to netcdf
    tmp_c = ep.Convert(data_dir + raw_file, model='EK60')
    tmp_c.to_netcdf(save_path=data_dir + 'output')
    # Calibrate raw
    ed = ep.process.EchoData(tmp_c.output_file)
    proc = ep.process.Process('EK60', ed)
    proc.get_Sv(ed)
    plat = xr.open_dataset(tmp_c.output_file, group='Platform')
    water_level = plat.water_level[0, 0].values

    ed.Sv['range'] = ed.Sv.range.isel(frequency=0, ping_time=0).load()
    ed.Sv = ed.Sv.swap_dims({'range_bin': 'range'})
    # Plot Sv for one frequency
    ed.Sv.Sv.isel(frequency=0).plot(x='ping_time', vmax=-40, vmin=-100, yincrease=False)

    # Parse region file
    evr_paths = data_dir + 'x1.evr'
    regions = Region2D(evr_paths)
    regions.raw_range = ed.Sv['range']
    regions.parse_file(convert_range_edges=True)
    # Plot region 11
    regions.plot_region(11, offset=water_level)

    plt.show()
