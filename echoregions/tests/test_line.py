from pathlib import Path
import pandas as pd
import pytest
import echoregions as er

data_dir = Path("./echoregions/test_data/")
evl_path = data_dir / "x1.evl"


# TODO: Make a new EVL file with only 1 line,
# and check for the exact value for all fields


def test_plot():
    """
    Test plotting Lines with options.
    """
    start_date = pd.to_datetime("2017-06-25")
    end_date = pd.to_datetime("2017-06-26")
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
    start_date = pd.to_datetime("2017-06-25")
    end_date = pd.to_datetime("2017-06-26")
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
    Test replacing NaN values in line.
    """
    lines = er.read_evl(evl_path)
    lines.data.loc[0, "depth"] = -10000.99  # Replace a value with the one used for nans
    lines.nan_depth_value = 20
    lines.replace_nan_depth(inplace=True)
    assert lines.data.loc[0, "depth"] == 20
