import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset

import echoregions as er

data_dir = "./echoregions/test_data/"
output_csv = data_dir + "output_CSV/"
output_json = data_dir + "output_JSON/"


def test_plot():
    """
    Test region plotting running without error.
    """
    evr_path = data_dir + "transect.evr"
    r2d = er.read_evr(evr_path, min_depth=0, max_depth=100)
    r2d.plot([11], color="k")


def test_select_sonar_file():
    """
    Test sonar file selection based on region bounds.
    """
    raw_files = [
        "Summer2017-D20170625-T124834.nc",
        "Summer2017-D20170625-T132103.nc",
        "Summer2017-D20170625-T134400.nc",
        "Summer2017-D20170625-T140924.nc",
        "Summer2017-D20170625-T143450.nc",
        "Summer2017-D20170625-T150430.nc",
        "Summer2017-D20170625-T153818.nc",
        "Summer2017-D20170625-T161209.nc",
        "Summer2017-D20170625-T164600.nc",
        "Summer2017-D20170625-T171948.nc",
        "Summer2017-D20170625-T175136.nc",
        "Summer2017-D20170625-T181701.nc",
        "Summer2017-D20170625-T184227.nc",
        "Summer2017-D20170625-T190753.nc",
        "Summer2017-D20170625-T193400.nc",
        "Summer2017-D20170625-T195927.nc",
        "Summer2017-D20170625-T202452.nc",
        "Summer2017-D20170625-T205018.nc",
    ]

    # Parse region file
    evr_paths = data_dir + "transect.evr"
    r2d = er.read_evr(evr_paths)
    raw = r2d.select_sonar_file(raw_files, 11)
    assert raw == ["Summer2017-D20170625-T205018.nc"]


def test_select_region():
    """
    tests select region functionality
    """
    evr_path = data_dir + "transect.evr"
    r2d = er.read_evr(evr_path)
    region_id = 2
    time_range = [
        pd.to_datetime("2019-07-02T19:00:00.000000000"),
        pd.to_datetime("2019-07-02T20:00:00.000000000"),
    ]
    depth_range = [-10000.0, 10000.0]
    df_1 = r2d.select_region(region_id=region_id)
    df_2 = r2d.select_region(time_range=time_range)
    df_3 = r2d.select_region(depth_range=depth_range)
    for df_region_id in df_1["region_id"]:
        assert df_region_id == region_id
    for time_array in df_2["time"]:
        for time in time_array:
            assert pd.to_datetime(time) >= time_range[0]
            assert pd.to_datetime(time) <= time_range[1]
    for depth_array in df_3["depth"]:
        for depth in depth_array:
            assert depth >= depth_range[0]
            assert depth <= depth_range[1]


@pytest.mark.filterwarnings("ignore:No gridpoint belongs to any region")
def test_mask_no_overlap():
    """
    test if mask is empty when there is no overlap
    """
    evr_path = data_dir + "transect.evr"
    r2d = er.read_evr(evr_path)

    Sv_no_overlap = xr.open_zarr(os.path.join(data_dir, "transect.zarr")).Sv

    M = r2d.mask(Sv_no_overlap.isel(channel=0), [8])
    M.plot()
    # from matplotlib import pyplot as plt
    # plt.show()
    assert isinstance(M, DataArray)
    assert M.isnull().data.all()


def test_mask_correct_labels():
    """testing if the generated id labels are as expected"""

    evr_path = data_dir + "transect.evr"
    r2d = er.read_evr(evr_path)
    region_ids = r2d.data.region_id.values  # Output is that of IntegerArray
    region_ids = list(region_ids)  # Convert to List
    # Convert numpy numeric values to basic Python float values
    region_ids = [region_id.item() for region_id in region_ids]
    da_Sv = xr.open_zarr(os.path.join(data_dir, "transect.zarr")).Sv
    M = r2d.mask(da_Sv.isel(channel=0), region_ids, mask_labels=region_ids)
    M.plot()
    # from matplotlib import pyplot as plt
    # plt.show()
    # it matches only 13th and 18th region and there exists a nan value at point of no overlap
    values = list(np.unique(M))
    assert values[0] == 13.0
    assert values[1] == 18.0
    assert np.isnan(values[2])


def test_select_type_error():
    """
    Tests select error functionality for regions.
    """

    evr_paths = data_dir + "transect.evr"
    r2d = er.read_evr(evr_paths)
    with pytest.raises(TypeError):
        empty_dataset = Dataset()
        _ = r2d.select_region(empty_dataset)
    with pytest.raises(TypeError):
        empty_tuple = ()
        _ = r2d.select_region(empty_tuple)


def test_mask_type_error():
    """
    Tests mask error functionality for regions.
    """

    evr_paths = data_dir + "transect.evr"
    r2d = er.read_evr(evr_paths)
    da_Sv = xr.open_zarr(os.path.join(data_dir, "transect.zarr")).Sv
    with pytest.raises(TypeError):
        empty_tuple = ()
        _ = r2d.mask(da_Sv, empty_tuple)
    with pytest.raises(ValueError):
        empty_list = []
        _ = r2d.mask(da_Sv, empty_list)
