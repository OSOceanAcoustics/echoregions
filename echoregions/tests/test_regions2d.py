import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset

import echoregions as er

from ..regions2d.regions2d import Regions2D

DATA_DIR = Path("./echoregions/test_data/")
EVR_PATH = DATA_DIR / "transect.evr"
ZARR_PATH = DATA_DIR / "transect.zarr"


@pytest.fixture(scope="function")
def regions2d_fixture() -> Regions2D:
    """
    Regions2D object fixture.

    Returns
    -------
    r2d : Regions2D
        Object containing data of test EVR file.
    """

    r2d = er.read_evr(EVR_PATH)
    return r2d


@pytest.fixture(scope="function")
def da_Sv_fixture() -> DataArray:
    """
    Sonar ZARR data fixture.

    Returns
    -------
    da_Sv : DataArray
        DataArray containing Sv data of test zarr file.
    """

    da_Sv = xr.open_zarr(ZARR_PATH).Sv
    return da_Sv


@pytest.mark.regions2d
def test_empty_regions2d_parsing() -> None:
    """
    Tests empty EVR parsing.
    """

    # Read evr into regions2d
    r2d = er.read_evr(DATA_DIR / "transect_empty.evr")

    # Check shapes
    assert r2d.data.shape == (0, 22)
    assert r2d.select_region([11]).shape == (0, 22)


@pytest.mark.regions2d
def test_missing_bbox_regions2d_parsing() -> None:
    """
    Tests missing bbox EVR parsing.
    """

    # Read evr into regions2d
    r2d = er.read_evr(DATA_DIR / "transect_missing_bbox.evr")

    # Test shape
    assert r2d.data.shape == (2, 22)

    # Test selected region good bbox
    df_select_good = r2d.select_region([1])
    assert df_select_good.shape == (1, 22)
    df_select_good_columns = df_select_good[
        [
            "region_bbox_left",
            "region_bbox_right",
            "region_bbox_top",
            "region_bbox_bottom",
        ]
    ]
    assert df_select_good_columns.iloc[0]["region_bbox_left"] == pd.to_datetime(
        "2019-07-02 03:50:54.629500"
    )
    assert df_select_good_columns.iloc[0]["region_bbox_right"] == pd.to_datetime(
        "2019-07-02 08:10:09.425500"
    )
    assert df_select_good_columns.iloc[0]["region_bbox_top"] == -9999.99
    assert df_select_good_columns.iloc[0]["region_bbox_bottom"] == 9999.99

    # Test selected region bad bbox
    df_select_bad = r2d.select_region([2])
    assert df_select_bad.shape == (1, 22)
    df_select_bad_columns = df_select_bad[
        [
            "region_bbox_left",
            "region_bbox_right",
            "region_bbox_top",
            "region_bbox_bottom",
        ]
    ]
    assert pd.isna(df_select_bad_columns.iloc[0]["region_bbox_left"])
    assert pd.isna(df_select_bad_columns.iloc[0]["region_bbox_right"])
    assert pd.isna(df_select_bad_columns.iloc[0]["region_bbox_top"])
    assert pd.isna(df_select_bad_columns.iloc[0]["region_bbox_bottom"])


@pytest.mark.regions2d
def test_regions2d_parsing(regions2d_fixture: Regions2D) -> None:
    """
    Test parsing of Regions2D object via checking individual values and aggregate values.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    """

    # Get dataframe
    df_r2d = regions2d_fixture.data

    # Check shape
    assert df_r2d.shape == (18, 22)

    # Check individual values of specific row
    assert df_r2d.loc[4]["file_name"] == "transect.evr"
    assert df_r2d.loc[4]["file_type"] == "EVRG"
    assert df_r2d.loc[4]["evr_file_format_number"] == "7"
    assert df_r2d.loc[4]["echoview_version"] == "13.0.378.44817"
    assert df_r2d.loc[4]["region_id"] == 5
    assert df_r2d.loc[4]["region_structure_version"] == "13"
    assert df_r2d.loc[4]["region_name"] == "ST22"
    assert df_r2d.loc[4]["region_class"] == "Log"
    assert df_r2d.loc[4]["region_creation_type"] == "6"
    assert df_r2d.loc[4]["region_bbox_left"] == pd.to_datetime(
        "2019-07-02 13:14:00.488000"
    )
    assert df_r2d.loc[4]["region_bbox_right"] == pd.to_datetime(
        "2019-07-02 13:14:01.888000"
    )
    assert df_r2d.loc[4]["region_bbox_top"] == -9999.99
    assert df_r2d.loc[4]["region_bbox_bottom"] == 9999.99
    assert df_r2d.loc[4]["region_point_count"] == "4"
    assert df_r2d.loc[4]["region_selected"] == "0"
    assert df_r2d.loc[4]["dummy"] == "-1"
    assert df_r2d.loc[4]["region_bbox_calculated"] == 1
    assert df_r2d.loc[4]["region_type"] == "2"
    assert (
        df_r2d.loc[4]["time"]
        == [
            np.datetime64("2019-07-02T13:14:00.488000000"),
            np.datetime64("2019-07-02T13:14:00.488000000"),
            np.datetime64("2019-07-02T13:14:01.888000000"),
            np.datetime64("2019-07-02T13:14:01.888000000"),
        ]
    ).all()
    assert (df_r2d.loc[4]["depth"] == [-9999.99, 9999.99, 9999.99, -9999.99]).all()
    assert df_r2d.loc[4]["region_notes"] == [
        "Starting transect x22 at 22.0, going ~5 kts to allow a barge to pass"
    ]
    assert df_r2d.loc[4]["region_detection_settings"] == []


@pytest.mark.regions2d
def test_evr_to_file(regions2d_fixture: Regions2D) -> None:
    """
    Test converting an Echoview 2D Regions files (.EVR).
    """

    # Get output path
    output_csv = DATA_DIR / "output_CSV/"

    # Create CSV
    regions2d_fixture.to_csv(output_csv)

    # Remove files
    assert os.path.exists(regions2d_fixture.output_file[0])
    os.remove(regions2d_fixture.output_file[0])

    # Remove directories
    os.rmdir(output_csv)


@pytest.mark.regions2d
def test_plot(regions2d_fixture: Regions2D) -> None:
    """
    Test plotting Regions2D object without errors.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """

    regions2d_fixture.plot([11], color="k")


@pytest.mark.regions2d
def test_select_sonar_file(regions2d_fixture: Regions2D) -> None:
    """
    Test sonar file selection based on region bounds.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
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

    # Check for correct sonar file
    raw = regions2d_fixture.select_sonar_file(raw_files, 11)
    assert raw == ["Summer2017-D20170625-T205018.nc"]


@pytest.mark.regions2d
def test_select_region(regions2d_fixture: Regions2D) -> None:
    """
    Tests select region functionality.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """

    # Set parameters to select on
    region_id = 2
    time_range = [
        pd.to_datetime("2019-07-02T19:00:00.000000000"),
        pd.to_datetime("2019-07-02T20:00:00.000000000"),
    ]
    depth_range = [-10000.0, 10000.0]

    # Perform selection and receive sampled dataframes
    df_1 = regions2d_fixture.select_region(region_id=region_id)
    df_2 = regions2d_fixture.select_region(time_range=time_range)
    df_3 = regions2d_fixture.select_region(depth_range=depth_range)

    # Check for correct region_id
    for df_region_id in df_1["region_id"]:
        assert df_region_id == region_id
    # Check for correct time ranges
    for time_array in df_2["time"]:
        for time in time_array:
            assert pd.to_datetime(time) >= time_range[0]
            assert pd.to_datetime(time) <= time_range[1]
    # Check for correct depth ranges
    for depth_array in df_3["depth"]:
        for depth in depth_array:
            assert depth >= depth_range[0]
            assert depth <= depth_range[1]


@pytest.mark.regions2d
def test_select_region_errors(regions2d_fixture: Regions2D) -> None:
    """
    Tests select region error functionality.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """

    # Check incorrect region_id type
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(region_id=())
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(region_id=[1, []])

    # Check time range type:
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(
            time_range=(
                pd.to_datetime("2019-07-02T19:00:00.000000000"),
                pd.to_datetime("2019-07-02T20:00:00.000000000"),
            )
        )
    # Check time range list size
    with pytest.raises(ValueError):
        regions2d_fixture.select_region(
            time_range=[
                pd.to_datetime("2019-07-02T19:00:00.000000000"),
                pd.to_datetime("2019-07-02T20:00:00.000000000"),
                pd.to_datetime("2019-07-02T21:00:00.000000000"),
            ]
        )
    # Check time range value types
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(
            time_range=[pd.to_datetime("2019-07-02T20:00:00.000000000"), 10]
        )
    # Check time range second index bigger than first index
    with pytest.raises(ValueError):
        regions2d_fixture.select_region(
            time_range=[
                pd.to_datetime("2019-07-02T20:00:00.000000000"),
                pd.to_datetime("2019-07-02T19:00:00.000000000"),
            ]
        )

    # Check depth range type:
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(depth_range=(-10000.0, 10000.0))
    # Check depth range list size
    with pytest.raises(ValueError):
        regions2d_fixture.select_region(
            depth_range=[
                -10000.0,
                0.0,
                10000.0,
            ]
        )
    # Check depth range value types
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(depth_range=[-10000.0, []])
    # Check depth range second index bigger than first index
    with pytest.raises(ValueError):
        regions2d_fixture.select_region(
            depth_range=[
                10000.0,
                -10000.0,
            ]
        )


@pytest.mark.filterwarnings("ignore:No gridpoint belongs to any region")
@pytest.mark.regions2d
def test_mask_no_overlap(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Test if mask is empty when there is no overlap.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create empty mask
    M = regions2d_fixture.mask(da_Sv_fixture.isel(channel=0), [8])

    # Check that all values are null
    assert M.isnull().data.all()


@pytest.mark.regions2d
def test_mask_correct_labels(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Test if the generated id labels are as expected

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Extract Region IDs and convert to Python float values
    region_ids = (
        regions2d_fixture.data.region_id.values
    )  # Output is that of IntegerArray
    region_ids = list(region_ids)  # Convert to List
    region_ids = [
        region_id.item() for region_id in region_ids  # Convert to basic python values
    ]

    # Create mask.
    M = regions2d_fixture.mask(
        da_Sv_fixture.isel(channel=0), region_ids, mask_labels=region_ids
    )

    # Check that the mask's values matches only 13th and 18th region and there exists a nan value
    # and that there exists a point of no overlap (nan value)
    values = list(np.unique(M))
    assert values[0] == 13.0
    assert values[1] == 18.0
    assert np.isnan(values[2])


@pytest.mark.regions2d
def test_select_type_error(regions2d_fixture: Regions2D) -> None:
    """
    Test for select region errors.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """

    # Check empty dataset error
    with pytest.raises(TypeError):
        empty_dataset = Dataset()
        _ = regions2d_fixture.select_region(empty_dataset)
    # Check empty tuple error
    with pytest.raises(TypeError):
        empty_tuple = ()
        _ = regions2d_fixture.select_region(empty_tuple)


@pytest.mark.regions2d
def test_mask_type_error(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Tests mask error functionality for regions.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Check empty tuple error
    with pytest.raises(TypeError):
        empty_tuple = ()
        _ = regions2d_fixture.mask(da_Sv_fixture, empty_tuple)
    # Check empty list error
    with pytest.raises(ValueError):
        empty_list = []
        _ = regions2d_fixture.mask(da_Sv_fixture, empty_list)
