import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import echoregions as er

data_dir = Path("./echoregions/test_data/")
evl_path = data_dir / "transect.evl"


# TODO: Make a new EVL file with only 1 line,
# and check for the exact value for all fields


def test_plot():
    """
    Test plotting Lines with options.
    """
    start_date = pd.to_datetime("2019-07-02")
    end_date = pd.to_datetime("2019-07-03")
    lines = er.read_evl(evl_path)
    lines.plot(
        start_time=start_date,
        end_time=end_date,
        max_depth=800,
        fill_between=True,
    )


def test_plot_type_error():
    """
    Test plotting Lines with options.
    """
    start_date = pd.to_datetime("2019-07-02")
    end_date = pd.to_datetime("2019-07-03")
    bad_start_date = "2017-06-25"
    bad_end_date = "2017-06-26"
    lines = er.read_evl(evl_path)
    with pytest.raises(TypeError):
        lines.plot(
            start_time=bad_start_date,
            end_time=bad_end_date,
            max_depth=800,
            fill_between=True,
        )
    with pytest.raises(TypeError):
        lines.plot(
            bad_start_time=bad_start_date,
            bad_end_time=end_date,
            max_depth=800,
            fill_between=True,
        )
    with pytest.raises(TypeError):
        lines.plot(
            bad_start_time=start_date,
            bad_end_time=bad_end_date,
            max_depth=800,
            fill_between=True,
        )


def test_replace_nan_depth():
    """
    Test replacing NaN values in line for both inplace=True and inplace=False.
    """
    lines_1 = er.read_evl(evl_path, nan_depth_value=20)
    lines_1.data.loc[
        0, "depth"
    ] = -10000.99  # Replace a value with the one used for nans
    lines_1.replace_nan_depth(inplace=True)
    assert lines_1.data.loc[0, "depth"] == 20

    lines_2 = er.read_evl(evl_path, nan_depth_value=20)
    lines_2.data.loc[
        0, "depth"
    ] = -10000.99  # Replace a value with the one used for nans
    regions = lines_2.replace_nan_depth(inplace=False)
    assert regions.loc[0, "depth"] == 20


def test_lines_mask():
    """
    Tests lines_mask on an overlapping (over time) evl file.
    """
    lines = er.read_evl(data_dir / "transect.evl")
    da_Sv = xr.open_zarr(os.path.join(data_dir, "transect.zarr")).Sv
    M = lines.mask(da_Sv.isel(channel=0))
    # from matplotlib import pyplot as plt
    # plt.show()
    M.plot(yincrease=False)
    unique_values = np.unique(M.data.compute(), return_counts=True)
    values = unique_values[0]
    counts = unique_values[1]
    assert values[0] == 0 and values[1] == 1
    assert counts[1] > counts[0]
