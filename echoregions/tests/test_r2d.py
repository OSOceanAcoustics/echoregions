import numpy as np

import echoregions as er

data_dir = "./echoregions/test_data/"
output_csv = data_dir + "output_CSV/"
output_json = data_dir + "output_JSON/"

# TODO: Make a new region file with only 1 region,
# and check for the exact value for all fields


def test_plot():
    """
    Test region plotting.
    """
    evr_path = data_dir + "x1.evr"
    r2d = er.read_evr(evr_path, min_depth=0, max_depth=100, offset=5)
    df = r2d.data.loc[r2d.data["region_name"] == "Chicken nugget"]
    r2d.plot([11], color="k")
    assert df["depth"][10][0] == 102.2552007996
    assert df["time"][10][0] == np.datetime64("2017-06-25T20:01:47.093000000")


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
    evr_paths = data_dir + "x1.evr"
    r2d = er.read_evr(evr_paths)
    raw = r2d.select_sonar_file(raw_files, 11)
    assert raw == ["Summer2017-D20170625-T195927.nc"]
