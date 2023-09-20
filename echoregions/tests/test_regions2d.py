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
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """

    # Get dataframe
    df_r2d = regions2d_fixture.data

    # Check shape
    assert df_r2d.shape == (19, 22)

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

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
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
        "Summer2019-D20190702-T171948.raw",
        "Summer2019-D20190702-T175136.raw",
        "Summer2019-D20190702-T181701.raw",
        "Summer2019-D20190702-T184227.raw",
        "Summer2019-D20190702-T190753.raw",
        "Summer2019-D20190702-T193400.raw",
        "Summer2019-D20190702-T195927.raw",
        "Summer2019-D20190702-T202452.raw",
        "Summer2019-D20190702-T205018.raw",
    ]

    # Check for correct sonar file
    # Note, below are the region times for region 11:
    # [
    #   '2019-07-02T18:11:51.190000000' '2019-07-02T18:11:51.190000000'
    #   '2019-07-02T18:11:52.540000000' '2019-07-02T18:11:52.540000000'
    # ]
    # So the idea is that the file Summer2019-D20190702-T175136.raw
    # encompasses the time period between 2019-07-02T17:51:36.000000000
    # and 2019-07-02T18:17:01.000000000 since it is proceeded by file
    # Summer2019-D20190702-T181701.raw.
    raw = regions2d_fixture.select_sonar_file(raw_files, 11)
    assert raw == ["Summer2019-D20190702-T175136.raw"]


@pytest.mark.regions2d
def test_empty_select_sonar_file(regions2d_fixture: Regions2D) -> None:
    """
    Test sonar file selection for raw files not in the specified year.
    This should come out as null.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """
    raw_2017_files = [
        "Summer2017-D20170625-T124834.raw",
        "Summer2017-D20170625-T132103.raw",
        "Summer2017-D20170625-T134400.raw",
        "Summer2017-D20170625-T140924.raw",
    ]
    raw_2021_files = [
        "Summer2021-D20210625-T124834.raw",
        "Summer2021-D20210625-T132103.raw",
        "Summer2021-D20210625-T134400.raw",
        "Summer2021-D20210625-T140924.raw",
    ]

    # Check for empty sonar files
    subset_raw_2017_files = regions2d_fixture.select_sonar_file(raw_2017_files, 11)
    assert subset_raw_2017_files == []
    subset_raw_2021_files = regions2d_fixture.select_sonar_file(raw_2021_files, 11)
    assert subset_raw_2021_files == []


@pytest.mark.regions2d
def test_invalid_select_sonar_file(regions2d_fixture: Regions2D) -> None:
    """
    Test that sonar file selection functions raise the correct type errors.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """
    raw_files_str = "Summer2017-D20170625-T124834.raw"
    raw_files_with_int = [
        "Summer2017-D20170625-T124834.raw",
        10,
    ]
    raw_files_with_invalid_simrad_format = [
        "Summer2019-D20190625-T181701.raw",
        "Summer2019-D20190625-T184227.raw",
        "Summer2019-Z20190625-M190753.raw",
        "Summer2019-D20190625-T193400.raw",
    ]

    # Check that type errors are correctly thrown
    with pytest.raises(TypeError):
        regions2d_fixture.select_sonar_file(raw_files_str, 11)
    with pytest.raises(TypeError):
        regions2d_fixture.select_sonar_file(raw_files_with_int, 11)

    # Check that value error is correctly thrown
    with pytest.raises(ValueError):
        regions2d_fixture.select_sonar_file(raw_files_with_invalid_simrad_format, 11)


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
    assert M.isnull().all()


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
    region_ids = regions2d_fixture.data.region_id.astype(float).to_list()

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


@pytest.mark.regions2d
def test_mask_2d_3d_2d_3d(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Testing if converting 2d-3d-2d-3d masks works.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Extract region_ids
    region_ids = regions2d_fixture.data.region_id.astype(float).to_list()

    # Create mask
    M = regions2d_fixture.mask(da_Sv_fixture, region_ids, mask_labels=region_ids)

    # Test values from converted 3D array (previous 2D array)
    mask_3d_ds = er.convert_mask_2d_to_3d(M)
    assert mask_3d_ds.mask_3d.shape == (2, 3955, 1681)
    assert list(mask_3d_ds.mask_dictionary) == [13.0, 18.0]

    # Test values from converted 2D array (previously 3D array)
    mask_2d_da = er.convert_mask_3d_to_2d(mask_3d_ds)
    assert mask_2d_da.equals(M)

    # Test values from 3D array (previously 2D array)
    second_mask_3d_ds = er.convert_mask_2d_to_3d(mask_2d_da)
    assert second_mask_3d_ds.equals(mask_3d_ds)


@pytest.mark.filterwarnings("ignore:No gridpoint belongs to any region")
@pytest.mark.regions2d
def test_nan_mask_2d_3d_2d_3d(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Testing if converting 2d-3d-2d-3d masks works for nan mask.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create mask
    M = regions2d_fixture.mask(da_Sv_fixture, [5.0])

    # Check if mask is null/empty
    assert isinstance(M, DataArray)
    assert M.isnull().all()

    # Test values from converted 3D array (previous 2D array)
    mask_3d_ds = er.convert_mask_2d_to_3d(M)
    assert np.unique(mask_3d_ds.mask_3d.data)[0] == 0

    # Test values from converted 2D array (previously 3D array)
    mask_2d_da = er.convert_mask_3d_to_2d(mask_3d_ds)
    assert mask_2d_da.equals(M)
    assert mask_2d_da.isnull().all()

    # Test values from 3D array (previously 2D array)
    second_mask_3d_ds = er.convert_mask_2d_to_3d(mask_2d_da)
    assert second_mask_3d_ds.equals(mask_3d_ds)


@pytest.mark.regions2d
def test_one_label_mask_2d_3d_2d_3d(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Testing if converting 2d-3d-2d-3d masks works for 1 label mask.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Extract region_ids
    region_ids = regions2d_fixture.data.region_id.astype(float).to_list()

    # Create mask
    M = regions2d_fixture.mask(da_Sv_fixture, region_ids, mask_labels=region_ids)

    # Remove 18.0 from mask
    M = xr.where(M == 18.0, np.nan, M)

    # Test values of 2D Mask
    M_values = M.values
    assert set(np.unique(M_values[~np.isnan(M_values)])) == {13.0}
    assert M.shape == (3955, 1681)

    # Test values from converted 3D array (previous 2D array)
    mask_3d_ds = er.convert_mask_2d_to_3d(M)
    assert list(mask_3d_ds.mask_dictionary) == [13.0]
    assert mask_3d_ds.mask_3d.shape == (1, 3955, 1681)

    # Test values from converted 2D array (previously 3D array)
    mask_2d_da = er.convert_mask_3d_to_2d(mask_3d_ds)
    assert mask_2d_da.equals(M)

    # Test values from 3D array (previously 2D array)
    second_mask_3d_ds = er.convert_mask_2d_to_3d(mask_2d_da)
    assert second_mask_3d_ds.equals(mask_3d_ds)


@pytest.mark.regions2d
def test_overlapping_mask_3d_2d(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Testing if converting 3d to 2d with overlapping mask produces error.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Extract region_ids
    region_ids = regions2d_fixture.data.region_id.astype(float).to_list()

    # Create mask
    M = regions2d_fixture.mask(da_Sv_fixture, region_ids, mask_labels=region_ids)

    # Test values from converted 3D array (previous 2D array)
    mask_3d_ds = er.convert_mask_2d_to_3d(M)
    assert mask_3d_ds.mask_3d.shape == (2, 3955, 1681)
    assert list(mask_3d_ds.mask_dictionary) == [13.0, 18.0]

    # Turn first (0th index) array into all 1s to guarantee overlap
    mask_3d_ds.mask_3d[0] = xr.ones_like(mask_3d_ds.mask_3d[0])

    # Trying to convert 3d mask to 2d should raise ValueError
    with pytest.raises(ValueError):
        er.convert_mask_3d_to_2d(mask_3d_ds)


@pytest.mark.regions2d
def test_within_transect(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Tests functionality for transect_mask.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create transect mask with no errors
    transect_dict = {"start": "ST", "break": "BT", "resume": "RT", "end": "ET"}
    M = regions2d_fixture.transect_mask(
        da_Sv=da_Sv_fixture, transect_dict=transect_dict
    ).compute()

    # Check M dimensions
    assert M.shape == (3955, 1681)

    # This entire .zarr file should be covered by the single start and end transect period
    # found in the EVR file, so the only values listed should be 1, implying everything is
    # within-transect.
    assert len(list(np.unique(M.data))) == 1
    assert list(np.unique(M.data))[0] == 1

    # Test number of 1 values
    assert np.unique(M.data, return_counts=True)[1][0] == 6648355


@pytest.mark.regions2d
def test_within_transect_no_ET_ST(da_Sv_fixture: DataArray) -> None:
    """
    Tests functionality for evr file with no ST and for evr file with no ET.
    Should raise appropriate UserWarning and should use first row for ST
    and last row for ET.


    Parameters
    ----------
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    transect_dict = {"start": "ST", "break": "BT", "resume": "RT", "end": "ET"}
    with pytest.warns(UserWarning):
        evr_path = DATA_DIR / "transect_no_ST.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(da_Sv=da_Sv_fixture, transect_dict=transect_dict)
    with pytest.warns(UserWarning):
        evr_path = DATA_DIR / "transect_no_ET.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(da_Sv=da_Sv_fixture, transect_dict=transect_dict)


@pytest.mark.regions2d
def test_within_transect_bad_dict(da_Sv_fixture: DataArray) -> None:
    """
    Tests functionality for transect_mask with invalid dictionary values.

    Parameters
    ----------
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Get Regions2D Object
    evr_path = DATA_DIR / "transect.evr"
    r2d = er.read_evr(evr_path)

    # Create dictionary with duplicates
    transect_dict_duplicate = {
        "start": "BT",
        "break": "BT",
        "resume": "RT",
        "end": "ET",
    }
    with pytest.raises(ValueError):
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture, transect_dict=transect_dict_duplicate
        )

    # Create dictionary with integers
    transect_dict_int = {"start": "ST", "break": "BT", "resume": "RT", "end": 4}
    with pytest.raises(TypeError):
        _ = r2d.transect_mask(da_Sv=da_Sv_fixture, transect_dict=transect_dict_int)


@pytest.mark.regions2d
def test_within_transect_invalid_next(da_Sv_fixture: DataArray) -> None:
    """
    Tests functionality for evr file with invalid next transect type values.

    Parameters
    ----------
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Initialize proper dictionary
    transect_dict = {"start": "ST", "break": "BT", "resume": "RT", "end": "ET"}

    # Should raise value error as ST is followed by ST
    with pytest.raises(ValueError):
        evr_path = DATA_DIR / "x1_ST_ST.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(da_Sv=da_Sv_fixture, transect_dict=transect_dict)

    # Should raise value error as RT is followed by RT
    with pytest.raises(ValueError):
        evr_path = DATA_DIR / "x1_RT_RT.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(da_Sv=da_Sv_fixture, transect_dict=transect_dict)

    # Should raise value error as BT is followed by ET
    with pytest.raises(ValueError):
        evr_path = DATA_DIR / "x1_BT_ET.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(da_Sv=da_Sv_fixture, transect_dict=transect_dict)

    # Should raise value error as ET is followed by RT
    with pytest.raises(ValueError):
        evr_path = DATA_DIR / "x1_ET_RT.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(da_Sv=da_Sv_fixture, transect_dict=transect_dict)
