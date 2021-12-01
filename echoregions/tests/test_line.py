from pathlib import Path

import echoregions as er

data_dir = Path("./echoregions/test_data/")
evl_path = data_dir / "x1.evl"


# TODO: Make a new EVL file with only 1 line,
# and check for the exact value for all fields


def test_plot():
    # Test plotting Lines with options
    start_date = "2017-06-25"
    end_date = "2017-06-26"
    lines = er.read_evl(evl_path)
    lines.plot(
        start_ping_time=start_date,
        end_ping_time=end_date,
        max_depth=800,
        fill_between=True,
    )


def test_replace_nan_depth():
    lines = er.read_evl(evl_path)
    lines.data.loc[0, "depth"] = -10000.99  # Replace a value with the one used for nans
    lines.nan_depth_value = 20
    lines.replace_nan_depth(inplace=True)
    assert lines.data.loc[0, "depth"] == 20
