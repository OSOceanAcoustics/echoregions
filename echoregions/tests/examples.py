import glob
import xarray as xr
import echopype as ep
import matplotlib.pyplot as plt
from ..formats import Regions2D
import echoregions as er
from pathlib import Path


data_dir = Path('./echoregions/test_data/ek60/')
output_csv = data_dir / 'output_CSV/'
output_json = data_dir / 'output_JSON/'
nc_file = data_dir / 'hake_nc' / 'Summer2017-D20170625-T195927.nc'
raw_files = list((data_dir / 'hake_nc').glob('*.nc'))


def get_sv():
    # Calibrate raw
    ed = ep.process.EchoData(str(nc_file))
    proc = ep.process.Process('EK60', ed)
    proc.get_Sv(ed)
    plat = xr.open_dataset(nc_file, group='Platform')
    water_level = plat.water_level[0, 0].values

    ed.Sv['range'] = ed.Sv.range.isel(frequency=0, ping_time=0).load()
    ed.Sv = ed.Sv.swap_dims({'range_bin': 'range'})

    return ed, water_level


def test_region_plot():
    # Test plotting a region on top of an echogram

    # Parse region file
    evr_path = data_dir / 'x1.evr'
    regions = Regions2D(str(evr_path))
    regions.parse_file()
    raw = regions.select_raw(raw_files, 11)

    # Calibrate raw
    ed = ep.process.EchoData(str(raw))
    proc = ep.process.Process('EK60', ed)
    proc.get_Sv(ed)
    plat = xr.open_dataset(raw, group='Platform')
    water_level = plat.water_level[0, 0].values

    ed.Sv['range'] = ed.Sv.range.isel(frequency=0, ping_time=0).load()
    ed.Sv = ed.Sv.swap_dims({'range_bin': 'range'})
    # Plot Sv for one frequency
    ed.Sv.Sv.isel(frequency=0).plot(x='ping_time', vmax=-40, vmin=-100, yincrease=False)

    # Plot region
    regions.depth = ed.Sv['range']
    regions.convert_output()
    regions.plot_region(11, offset=-water_level)
    plt.show()


def test_plot_multi():
    # Test ploting all regions of an EVR file
    # Parse region file
    # ed, water_level = get_sv()

    evr_paths = data_dir / 'x1.evr'
    regions = Regions2D(str(evr_paths))
    regions.to_dataframe()
    # regions.depth = ed.Sv['range']
    # regions.parse_file(convert_depth_edges=True)

    for region in regions:
        regions.plot_region(region)

    plt.show()


def test_mask():
    """Test masking a region with the regionmask library."""
    ed, water_level = get_sv()

    files = glob.glob('echoregions/test_data/ek60/regions/*.evr')
    r2d = Regions2D(files[0])
    masked = r2d.mask_region(ed.Sv, '11', offset=-water_level)
    masked.Sv.isel(frequency=0).plot(yincrease=False)
    plt.show()


def test_region_line_plot():
    """Test plotting EVL files"""
    evl_path = data_dir / 'x1.bottom.evl'
    evr_path = data_dir / 'x1.evr'

    ed, water_level = get_sv()
    # Plot Sv for one frequency
    ed.Sv.Sv.isel(frequency=0).plot(x='ping_time', vmax=-40, vmin=-100, yincrease=False)

    # Plot bottom as filled in section
    line = er.read_evl(evl_path)
    line.plot(calibrated_dataset=ed.Sv, alpha=0.7, color='k')

    # Plot region
    r2d = er.read_evr(evr_path)
    r2d.plot_region(11, offset=-water_level)

    plt.show()
