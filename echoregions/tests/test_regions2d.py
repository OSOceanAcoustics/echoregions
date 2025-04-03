import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset

import echoregions as er
from echoregions import write_evr
from echoregions.regions2d.regions2d import Regions2D

DATA_DIR = Path("./echoregions/test_data/")
EVR_PATH = DATA_DIR / "transect_multi_mask.evr"
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

    da_Sv = xr.open_zarr(ZARR_PATH)["Sv"].compute()
    return da_Sv


@pytest.mark.regions2d
def test_read_regions_csv(regions2d_fixture: Regions2D) -> None:
    """
    Ensures that read_region_csv provides the same Regions2D object
    as read_evr.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """

    # Get Regions2D object and DataFrame
    r2d_1 = regions2d_fixture
    r2d_1_df = r2d_1.data

    # Send to CSV
    csv_file_path = DATA_DIR / "r2d_to_csv_file.csv"
    r2d_1.to_csv(csv_file_path)

    # Read Regions CSV and extract DataFrame
    r2d_2 = er.read_regions_csv(csv_file_path)
    r2d_2_df = r2d_2.data

    # Check for precision between depth columns
    assert np.all(
        [
            np.isclose(d_arr_1, d_arr_2).all()
            for d_arr_1, d_arr_2 in zip(r2d_1_df["depth"], r2d_2_df["depth"])
        ]
    )

    # Check for equality between the elements in both time columns
    assert np.all(
        [(t_arr_1 == t_arr_2).all() for t_arr_1, t_arr_2 in zip(r2d_1_df["time"], r2d_2_df["time"])]
    )

    # Check equality between the elements in both region_id columns
    assert np.all(
        [
            region_id_1 == region_id_2
            for region_id_1, region_id_2 in zip(r2d_1_df["region_id"], r2d_2_df["region_id"])
        ]
    )

    # Delete the file
    csv_file_path.unlink()


@pytest.mark.regions2d
def test_to_evr() -> None:
    """
    Tests that when we save a `Regions2D` object to `.evr` and read
    back that `.evr` file, we end up with the same inner dataframe.
    """
    # Get Regions2D object and DataFrame
    r2d_1 = er.read_evr(DATA_DIR / "transect.evr")
    r2d_1_df = r2d_1.data

    # Send to `.evr` file
    evr_file_path = DATA_DIR / "r2d_to_evr_file.evr"
    r2d_1.to_evr(evr_file_path)

    # Read back `.evr` file and extract DataFrame
    r2d_2 = er.read_evr(evr_file_path)
    r2d_2_df = r2d_2.data

    # Check that the dataframes are equal everywhere (not including the file name)
    assert r2d_1_df.drop("file_name", axis=1).equals(r2d_2_df.drop("file_name", axis=1))

    # Delete the file
    evr_file_path.unlink()


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
    assert df_r2d.loc[4]["file_name"] == "transect_multi_mask.evr"
    assert df_r2d.loc[4]["file_type"] == "EVRG"
    assert df_r2d.loc[4]["evr_file_format_number"] == "7"
    assert df_r2d.loc[4]["echoview_version"] == "13.0.378.44817"
    assert df_r2d.loc[4]["region_id"] == 5
    assert df_r2d.loc[4]["region_structure_version"] == "13"
    assert df_r2d.loc[4]["region_name"] == "ST22"
    assert df_r2d.loc[4]["region_class"] == "Log"
    assert df_r2d.loc[4]["region_creation_type"] == "6"
    assert df_r2d.loc[4]["region_bbox_left"] == pd.to_datetime("2019-07-02 13:14:00.488000")
    assert df_r2d.loc[4]["region_bbox_right"] == pd.to_datetime("2019-07-02 13:14:01.888000")
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
    # Plotting without closing
    regions2d_fixture.plot([11], color="k")

    # Plotting with closing
    regions2d_fixture.plot([11], close_regions=True, color="k")


@pytest.mark.regions2d
def test_select_sonar_file(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Tests select_sonar_file.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Test with full fixture and empty array
    selected_Sv = regions2d_fixture.select_sonar_file(
        [da_Sv_fixture, xr.DataArray([], dims="ping_time")]
    )
    assert len(selected_Sv) == 1
    assert selected_Sv[0].equals(da_Sv_fixture)

    # Test with subseted and partitioned fixture
    ping_time_index_partitions = [slice(0, 100), slice(200, 500), slice(600, 800)]
    partitioned_fixture_list = [
        da_Sv_fixture.isel(ping_time=ping_time_index_partition)
        for ping_time_index_partition in ping_time_index_partitions
    ]
    selected_Sv = regions2d_fixture.select_sonar_file(partitioned_fixture_list)
    assert len(selected_Sv) == 3
    assert selected_Sv == partitioned_fixture_list

    # Test with original full fixture and modified two index fixture with 1 ping_time value outside
    # of regions2d_fixture ping_time, and the other ping_time value inside
    inside_ping_time = np.datetime64(
        np.hstack(regions2d_fixture.data["time"].values).max()
    ) - np.timedelta64(1, "ns")
    outside_ping_time = np.datetime64(
        np.hstack(regions2d_fixture.data["time"].values).max()
    ) + np.timedelta64(1, "ns")
    two_ping_times_isel_da_Sv = da_Sv_fixture.isel(ping_time=slice(839, 841)).compute()
    two_ping_times_isel_da_Sv = two_ping_times_isel_da_Sv.assign_coords(
        {"ping_time": [inside_ping_time, outside_ping_time]}
    )
    selected_Sv = regions2d_fixture.select_sonar_file([da_Sv_fixture, two_ping_times_isel_da_Sv])
    assert len(selected_Sv) == 2
    assert selected_Sv == [da_Sv_fixture, two_ping_times_isel_da_Sv]

    # Test with original full fixture and modified two index fixture with both ping_time values
    # outside of regions2d_fixture ping_time
    first_outside_ping_time = np.datetime64(
        np.hstack(regions2d_fixture.data["time"].values).max()
    ) + np.timedelta64(1, "ns")
    second_outside_ping_time = np.datetime64(
        np.hstack(regions2d_fixture.data["time"].values).max()
    ) + np.timedelta64(2, "ns")
    two_ping_times_isel_da_Sv = da_Sv_fixture.isel(ping_time=slice(839, 841)).compute()
    two_ping_times_isel_da_Sv = two_ping_times_isel_da_Sv.assign_coords(
        {"ping_time": [first_outside_ping_time, second_outside_ping_time]}
    )
    selected_Sv = regions2d_fixture.select_sonar_file([da_Sv_fixture, two_ping_times_isel_da_Sv])
    assert len(selected_Sv) == 1
    assert selected_Sv[0].equals(da_Sv_fixture)


@pytest.mark.regions2d
def test_empty_select_sonar_file(regions2d_fixture: Regions2D) -> None:
    """
    Test sonar file selection for empty datasets. This should come out as empty.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """
    empty_da = xr.DataArray([], dims="ping_time")

    # Check for empty Sv files
    selected_Sv = regions2d_fixture.select_sonar_file(empty_da)
    assert selected_Sv == []


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

    # Check that type errors are correctly thrown
    with pytest.raises(TypeError):
        regions2d_fixture.select_sonar_file(raw_files_str, "ping_time")
    with pytest.raises(TypeError):
        regions2d_fixture.select_sonar_file([raw_files_str], "ping_time")


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
    region_class = "Trawl"
    time_range = [
        pd.to_datetime("2019-07-02T19:00:00.000000000"),
        pd.to_datetime("2019-07-02T20:00:00.000000000"),
    ]
    depth_range = [-10000.0, 10000.0]

    # Perform selection and receive sampled dataframes
    df_1 = regions2d_fixture.select_region(region_id=region_id)
    df_2 = regions2d_fixture.select_region(region_class=region_class)
    df_3 = regions2d_fixture.select_region(time_range=time_range)
    df_4 = regions2d_fixture.select_region(depth_range=depth_range)

    # Check for correct region_id
    for df_region_id in df_1["region_id"]:
        assert df_region_id == region_id
    # Check for correct region_class
    for df_region_class in df_2["region_class"]:
        assert df_region_class == region_class
    # Check for correct time ranges
    for time_array in df_3["time"]:
        for time in time_array:
            assert pd.to_datetime(time) >= time_range[0]
            assert pd.to_datetime(time) <= time_range[1]
    # Check for correct depth ranges
    for depth_array in df_4["depth"]:
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
    # Check incorrect user input behavior of both non-NaN region id and region class
    with pytest.raises(ValueError):
        regions2d_fixture.select_region(region_id=1, region_class="Hake")

    # Check incorrect region_id type
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(region_id=())
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(region_id=[1, []])

    # Check incorrect region_class type
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(region_class=())
    with pytest.raises(TypeError):
        regions2d_fixture.select_region(region_class=["1", []])

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


@pytest.mark.filterwarnings("ignore:No gridpoint belongs to any region.")
@pytest.mark.regions2d
def test_mask_empty_no_overlap(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Test if mask is empty when a region's depth values are invalid
    and also test mask is empty when there is no overlap.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Attempt to create mask on region with invalid depth values
    mask_output_0, region_points_0 = regions2d_fixture.region_mask(
        da_Sv_fixture.isel(channel=0), [8]
    )

    # Check that output is zeros like array and region points is empty
    assert mask_output_0["mask_3d"].equals(
        xr.zeros_like(da_Sv_fixture.isel(channel=0)).expand_dims({"region_id": ["dummy_region_id"]})
    )
    assert region_points_0.empty

    # Also attempt to create mask on region with invalid depth values and collapsing to 2D and check that
    # it is a fully NaN array and the region points are empty
    mask_output_1, region_points_1 = regions2d_fixture.region_mask(
        da_Sv_fixture.isel(channel=0), [8], collapse_to_2d=True
    )
    assert mask_output_1.isnull().all()
    assert region_points_1.empty

    # Create mask with regions that have no overlap with the Sv Data Array
    mask_output_2, region_points_2 = regions2d_fixture.region_mask(
        da_Sv_fixture.isel(channel=0), [8, 9, 10]
    )

    # Check that this mask is empty
    assert (mask_output_2["mask_3d"] == 0).all()

    # Check that region_points_2 is empty
    assert region_points_2.empty

    # Use region points to create Regions2D object
    r2d_2 = Regions2D(region_points_2, min_depth=0, max_depth=1000, input_type="CSV")

    # Run Regions2d masking to check if masking runs
    mask_output_3, region_points_3 = r2d_2.region_mask(da_Sv_fixture.isel(channel=0))

    # Check that output is zeros like array
    assert mask_output_3["mask_3d"].equals(
        xr.zeros_like(da_Sv_fixture.isel(channel=0)).expand_dims({"region_id": ["dummy_region_id"]})
    )
    assert region_points_3.empty


@pytest.mark.regions2d
def test_mask_type_error(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Tests mask error functionality for regions.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Check dataset error
    with pytest.raises(TypeError):
        _ = regions2d_fixture.region_mask(da_Sv_fixture.to_dataset("Sv"))
    # Check empty tuple error
    with pytest.raises(TypeError):
        empty_tuple = ()
        _ = regions2d_fixture.region_mask(da_Sv_fixture, empty_tuple)
    # Check empty list error
    with pytest.raises(ValueError):
        empty_list = []
        _ = regions2d_fixture.region_mask(da_Sv_fixture, empty_list)


@pytest.mark.regions2d
def test_mask_labels_region_ids_not_matching_error(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Tests when mask labels and region ids are not matching:
    When not all values in subset_region_ids are keys in 'mask_labels' and vice versa.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Check not matching error
    with pytest.raises(ValueError):
        _ = regions2d_fixture.region_mask(da_Sv_fixture, region_id=13, mask_labels={14: "TEST"})


@pytest.mark.regions2d
def test_mask_2d(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Testing if 2d mask with collapse_to_do=True works.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Get region_id and mask_labels
    region_id = regions2d_fixture.data.region_id.astype(int).to_list()
    mask_labels = {key: idx for idx, key in enumerate(region_id)}
    mask_labels[13] = "Mask1"

    # Create mask
    mask_2d_ds, _ = regions2d_fixture.region_mask(
        da_Sv_fixture, mask_labels=mask_labels, collapse_to_2d=True
    )

    # Check mask_2d values and counts
    mask_2d_values = np.unique(mask_2d_ds.mask_2d.data, return_counts=True)[0]
    mask_2d_counts = np.unique(mask_2d_ds.mask_2d.data, return_counts=True)[1]
    assert mask_2d_values[0] == 13
    assert mask_2d_values[1] == 18
    assert np.isnan(mask_2d_values[2])
    assert mask_2d_counts[0] == 120
    assert mask_2d_counts[1] == 59631
    assert mask_2d_counts[2] == 68081

    # Check mask_labels values
    assert (np.unique(mask_2d_ds.mask_labels.data) == ["17", "Mask1"]).all()
    assert (mask_2d_ds.mask_labels.region_id.values == [13, 18]).all()


@pytest.mark.regions2d
def test_mask_region_points(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Testing if masking, saving region points into new regions2d,
    and masking again produces the same region points.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Get region_id and mask_labels

    # Create mask
    _, region_points_1 = regions2d_fixture.region_mask(da_Sv_fixture.isel(channel=0))

    # Use region points to create Regions2D object
    r2d_2 = Regions2D(region_points_1, min_depth=0, max_depth=1000, input_type="CSV")

    # Run Regions2D masking to check if masking runs
    _, region_points_2 = r2d_2.region_mask(da_Sv_fixture.isel(channel=0))

    # Check if the two points are equal
    region_points_1.equals(region_points_2)


@pytest.mark.regions2d
def test_mask_3d_2d_3d_2d(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Testing if converting 3d-2d-3d-2d masks works.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Get region_id and mask_labels
    region_id = regions2d_fixture.data.region_id.astype(int).to_list()
    mask_labels = {key: idx for idx, key in enumerate(region_id)}
    mask_labels[13] = "Mask1"

    # Create mask
    mask_3d_ds, _ = regions2d_fixture.region_mask(da_Sv_fixture, mask_labels=mask_labels)

    # Check mask values
    assert (mask_3d_ds.mask_3d.region_id.values == [13, 18]).all()
    assert np.unique(mask_3d_ds.mask_3d.isel(region_id=0).data, return_counts=True)[1][0] == 127712
    assert np.unique(mask_3d_ds.mask_3d.isel(region_id=0).data, return_counts=True)[1][1] == 120
    assert np.unique(mask_3d_ds.mask_3d.isel(region_id=1).data, return_counts=True)[1][0] == 68201
    assert np.unique(mask_3d_ds.mask_3d.isel(region_id=1).data, return_counts=True)[1][1] == 59631

    # Check mask_labels values
    assert (np.unique(mask_3d_ds.mask_labels.data) == ["17", "Mask1"]).all()
    assert (mask_3d_ds.mask_labels.region_id.values == [13, 18]).all()

    # Convert 2d mask to 3d mask
    mask_2d_ds = er.convert_mask_3d_to_2d(mask_3d_ds)

    # Check mask_2d values and counts
    mask_2d_values = np.unique(mask_2d_ds.mask_2d.data, return_counts=True)[0]
    mask_2d_counts = np.unique(mask_2d_ds.mask_2d.data, return_counts=True)[1]
    assert mask_2d_values[0] == 13
    assert mask_2d_values[1] == 18
    assert np.isnan(mask_2d_values[2])
    assert mask_2d_counts[0] == 120
    assert mask_2d_counts[1] == 59631
    assert mask_2d_counts[2] == 68081

    # Convert 2d mask to 3d mask and test for equality with previous 3d mask
    mask_3d_ds_2nd = er.convert_mask_2d_to_3d(mask_2d_ds)
    mask_3d_ds_2nd.equals(mask_3d_ds)

    # Convert 2nd 3d mask to 2d mask and test for equality with previous 2d mask
    mask_2d_2nd = er.convert_mask_3d_to_2d(mask_3d_ds_2nd)
    mask_2d_2nd.equals(mask_2d_ds)


@pytest.mark.regions2d
def test_one_label_mask_3d_2d_3d_2d(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Testing if converting 3d-2d-3d-2d masks works for 1 label mask.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create 3d mask
    mask_3d_ds, _ = regions2d_fixture.region_mask(
        da_Sv_fixture, region_id=[18], mask_labels={18: "Mask1"}
    )

    # Check mask values
    assert (mask_3d_ds.mask_3d.region_id.values == [18]).all()
    assert np.unique(mask_3d_ds.mask_3d.isel(region_id=0).data, return_counts=True)[1][0] == 68201
    assert np.unique(mask_3d_ds.mask_3d.isel(region_id=0).data, return_counts=True)[1][1] == 59631

    # Check mask_labels values
    assert (np.unique(mask_3d_ds.mask_labels.data) == ["Mask1"]).all()
    assert (mask_3d_ds.mask_labels.region_id.values == [18]).all()

    # Convert 2d mask to 3d mask
    mask_2d_ds = er.convert_mask_3d_to_2d(mask_3d_ds)

    # Check mask_2d values and counts
    mask_2d_values = np.unique(mask_2d_ds.mask_2d.data, return_counts=True)[0]
    mask_2d_counts = np.unique(mask_2d_ds.mask_2d.data, return_counts=True)[1]
    assert mask_2d_values[0] == 18
    assert np.isnan(mask_2d_values[1])
    assert mask_2d_counts[0] == 59631
    assert mask_2d_counts[1] == 68201

    # Convert 2d mask to 3d mask and test for equality with previous 3d mask
    mask_3d_ds_2nd = er.convert_mask_2d_to_3d(mask_2d_ds)
    mask_3d_ds_2nd.equals(mask_3d_ds)

    # Convert 2nd 3d mask to 2d mask and test for equality with previous 2d mask
    mask_2d_ds_2nd = er.convert_mask_3d_to_2d(mask_3d_ds_2nd)
    mask_2d_ds_2nd.equals(mask_2d_ds)


@pytest.mark.filterwarnings("ignore:No gridpoint belongs to any region")
@pytest.mark.filterwarnings("ignore:Returning No Mask")
@pytest.mark.regions2d
def test_nan_mask_3d_2d_and_2d_3d(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Testing if both converting functions returns none for empty mask inputs.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create 3d mask
    mask_3d_ds, region_points = regions2d_fixture.region_mask(da_Sv_fixture, [8, 9, 10])

    # Check if mask is null/empty and check that region points is empty
    assert (mask_3d_ds.mask_3d == 0).all()
    assert region_points.empty

    # Attempt to convert empty 3d mask to 2d mask
    assert er.convert_mask_3d_to_2d(mask_3d_ds).isnull().all()

    # Create emtpy 2d mask with None values
    depth_values = [9.15, 9.34, 9.529, 9.719, 758.5]
    ping_time_values = ["2019-07-02T18:40:00", "2019-07-02T19:00:00"]
    mask_2d_ds = xr.Dataset()
    mask_2d_ds["mask_2d"] = xr.DataArray(
        np.full((len(depth_values), len(ping_time_values)), np.nan),
        coords={"depth": depth_values, "ping_time": ping_time_values},
        dims=["depth", "ping_time"],
    )

    # Check if mask is null/empty
    assert mask_2d_ds.mask_2d.isnull().all()

    # Attempt to convert empty 2d mask to 3d mask
    assert er.convert_mask_2d_to_3d(mask_2d_ds) is None


@pytest.mark.regions2d
def test_overlapping_mask_3d_2d(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Testing if converting 3d to 2d with overlapping mask produces error.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Extract region_id
    region_id = regions2d_fixture.data.region_id.astype(int).to_list()

    # Create 3d mask
    mask_3d_ds, _ = regions2d_fixture.region_mask(da_Sv_fixture, region_id)

    # Turn first (0th index) array corresponding to region id 13 into all 1s
    # to guarantee overlap with array corresponding to region id 18
    mask_3d_ds["mask_3d"] = xr.concat([xr.ones_like(mask_3d_ds.mask_3d[0])], dim="region_id")

    # Trying to convert 3d mask to 2d should raise ValueError
    with pytest.raises(ValueError):
        er.convert_mask_3d_to_2d(mask_3d_ds)


@pytest.mark.regions2d
def test_within_transect(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
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
    transect_sequence_type_dict = {"start": "ST", "break": "BT", "resume": "RT", "end": "ET"}
    df = regions2d_fixture.data
    df.loc[df["region_id"] == 5, "region_name"] = "Log"  # Remove early ST
    df.loc[df["region_id"] == 13, "region_name"] = "ST22"  # Place ST towards middle
    df.loc[df["region_id"] == 19, "region_name"] = "Log"  # Remove late ET
    df.loc[df["region_id"] == 14, "region_name"] = "ET22"  # Place ET towards middle
    regions2d_fixture.data = df
    M = regions2d_fixture.transect_mask(
        da_Sv=da_Sv_fixture, transect_sequence_type_dict=transect_sequence_type_dict
    ).compute()

    # Check M dimensions
    assert M.shape == (841, 152)

    # Check values
    assert len(list(np.unique(M.data))) == 2

    # Test number of 1 values
    assert np.unique(M.data, return_counts=True)[1][0] == 109440


@pytest.mark.regions2d
def test_within_transect_all(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Tests functionality for transect_mask with all values in the da_Sv within transect.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create transect mask with no errors
    transect_sequence_type_dict = {"start": "ST", "break": "BT", "resume": "RT", "end": "ET"}
    M = regions2d_fixture.transect_mask(
        da_Sv=da_Sv_fixture, transect_sequence_type_dict=transect_sequence_type_dict
    ).compute()

    # Check M dimensions
    assert M.shape == (841, 152)

    # This entire .zarr file should be covered by the single start and end transect period
    # found in the EVR file, so the only values listed should be 1, implying everything is
    # within-transect.
    assert len(list(np.unique(M.data))) == 1
    assert list(np.unique(M.data))[0] == 1

    # Test number of 1 values
    assert np.unique(M.data, return_counts=True)[1][0] == 127832


@pytest.mark.regions2d
def test_within_transect_no_regions(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Tests functionality for transect_mask for empty r2d object.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create transect mask with no errors
    r2d_empty = er.read_regions_csv(pd.DataFrame(columns=regions2d_fixture.data.columns))
    M = r2d_empty.transect_mask(da_Sv=da_Sv_fixture).compute()

    # Check M dimensions
    assert M.shape == (841, 152)

    # This entire output should be empty.
    assert len(list(np.unique(M.data))) == 1
    assert list(np.unique(M.data))[0] == 0


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
    transect_sequence_type_dict_duplicate = {
        "start": "BT",
        "break": "BT",
        "resume": "RT",
        "end": "ET",
    }
    with pytest.raises(ValueError):
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture, transect_sequence_type_dict=transect_sequence_type_dict_duplicate
        )

    # Create dictionary with integers
    transect_sequence_type_dict_int = {"start": "ST", "break": "BT", "resume": "RT", "end": 4}
    with pytest.raises(TypeError):
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture, transect_sequence_type_dict=transect_sequence_type_dict_int
        )


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
    transect_sequence_type_dict = {"start": "ST", "break": "BT", "resume": "RT", "end": "ET"}

    # Should raise Exception if ST is followed by ST
    with pytest.raises(Exception):
        evr_path = DATA_DIR / "x1_ST_ST.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture,
            transect_sequence_type_dict=transect_sequence_type_dict,
            must_pass_check=True,
        )

    # Should raise Exception if RT is followed by RT
    with pytest.raises(Exception):
        evr_path = DATA_DIR / "transect_RT_RT.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture,
            transect_sequence_type_dict=transect_sequence_type_dict,
            must_pass_check=True,
        )

    # Should raise value Exception if BT is followed by ET
    with pytest.raises(Exception):
        evr_path = DATA_DIR / "transect_BT_ET.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture,
            transect_sequence_type_dict=transect_sequence_type_dict,
            must_pass_check=True,
        )

    # Should raises Exception if ET is followed by RT
    with pytest.raises(Exception):
        evr_path = DATA_DIR / "transect_ET_RT.evr"
        r2d = er.read_evr(evr_path)
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture,
            transect_sequence_type_dict=transect_sequence_type_dict,
            must_pass_check=True,
        )


@pytest.mark.regions2d
def test_within_transect_small_bbox_distance_threshold(da_Sv_fixture: DataArray) -> None:
    """
    Tests functionality for transect_mask with small bbox distance threshold.

    Parameters
    ----------
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Get Regions2D Object
    evr_path = DATA_DIR / "transect.evr"
    r2d = er.read_evr(evr_path)

    with pytest.raises(Exception):
        _ = r2d.transect_mask(
            da_Sv=da_Sv_fixture, bbox_distance_threshold=0.001, must_pass_check=True
        )


@pytest.mark.regions2d
def test_evr_write(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Tests evr_write functionality.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Create mask
    region_id = regions2d_fixture.data.region_id.astype(int).to_list()
    mask_labels = {key: idx for idx, key in enumerate(region_id)}
    mask_labels[13] = "Mask1"
    mask_2d_ds, _ = regions2d_fixture.region_mask(
        da_Sv_fixture, mask_labels=mask_labels, collapse_to_2d=True
    )
    mask = xr.where(mask_2d_ds["mask_2d"].fillna(0) != 0, 1, 0)

    # Output file path
    evr_path = "test.evr"

    # Write to EVR file
    write_evr(
        evr_path=str(evr_path),
        mask=mask,
        region_classification="test_region_classification",
    )

    # Read the EVR file
    evr_data = er.read_evr(str(evr_path)).data

    # Check the number of regions
    assert len(evr_data) == 2

    # Check first region attributes
    region_1 = evr_data.iloc[0]
    assert region_1["region_id"] == 1
    assert region_1["region_structure_version"] == "13"
    assert region_1["region_point_count"] == "28"
    assert region_1["region_selected"] == "0"
    assert region_1["echoview_version"] == "12.0.341.42620"

    # Check second region attributes
    region_2 = evr_data.iloc[1]
    assert region_2["region_id"] == 2
    assert region_2["region_structure_version"] == "13"
    assert region_2["region_point_count"] == "4"
    assert region_2["region_selected"] == "0"
    assert region_2["echoview_version"] == "12.0.341.42620"

    # Check common metadata
    for _, region in evr_data.iterrows():
        assert region["file_name"] == "test.evr"
        assert region["file_type"] == "EVRG"
        assert region["evr_file_format_number"] == "7"

    os.remove(evr_path)


@pytest.mark.regions2d
def test_evr_write_exceptions(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Tests evr_write exceptions for incorrect mask input.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Create mask
    region_id = regions2d_fixture.data.region_id.astype(int).to_list()
    mask_labels = {key: idx for idx, key in enumerate(region_id)}
    mask_labels[13] = "Mask1"
    mask_2d_ds, _ = regions2d_fixture.region_mask(
        da_Sv_fixture, mask_labels=mask_labels, collapse_to_2d=True
    )

    # Output file path
    evr_path = "test.evr"

    # Test bad input cases
    with pytest.raises(TypeError, match="The 'mask' parameter must be an xarray.DataArray."):
        mask = mask_2d_ds["mask_2d"].data  # numpy array
        write_evr(
            evr_path=str(evr_path),
            mask=mask,
            region_classification="test_region_classification",
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The 'mask' must have only 'ping_time' and 'depth' as coordinates, but found ['depth', 'ping_time', 'region_id']."
        ),
    ):
        mask = xr.where(mask_2d_ds["mask_2d"].fillna(0) != 0, 1, 0).expand_dims({"region_id": [1]})
        write_evr(
            evr_path=str(evr_path),
            mask=mask,
            region_classification="test_region_classification",
        )
    with pytest.raises(
        ValueError,
        match="The 'mask' contains NaN values. Please remove or fill them before proceeding.",
    ):
        mask = mask_2d_ds["mask_2d"]
        write_evr(
            evr_path=str(evr_path),
            mask=mask,
            region_classification="test_region_classification",
        )


@pytest.mark.regions2d
def test_chunked_region_mask(regions2d_fixture: Regions2D, da_Sv_fixture: DataArray) -> None:
    """
    Testing if chunked region masking and computed region masking operations result in the same array and
    points and checks the array chunks for both 3D and 2D operation subtypes.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Set chunks
    chunk_dict = {"ping_time": 400, "depth": 100}
    # Create 3D masks, check chunks, and check that outputs are equal
    mask_3d_ds_chunked, mask_3d_points_chunked = regions2d_fixture.region_mask(
        da_Sv_fixture.chunk(chunk_dict), collapse_to_2d=False
    )
    mask_3d_ds_computed, mask_3d_points_computed = regions2d_fixture.region_mask(
        da_Sv_fixture.compute(), collapse_to_2d=False
    )
    assert mask_3d_ds_chunked.chunksizes["region_id"][0] == 1
    assert mask_3d_ds_chunked.chunksizes["ping_time"][0] == 400
    assert mask_3d_ds_chunked.chunksizes["depth"][0] == 100
    assert mask_3d_ds_chunked.equals(mask_3d_ds_computed)
    assert mask_3d_points_chunked.equals(mask_3d_points_computed)

    # Create 2D computed masks, check chunks, and check that outputs are equal
    mask_2d_ds_chunked, mask_2d_points_chunked = regions2d_fixture.region_mask(
        da_Sv_fixture.chunk(chunk_dict), collapse_to_2d=True
    )
    mask_2d_ds_computed, mask_2d_points_computed = regions2d_fixture.region_mask(
        da_Sv_fixture.compute(), collapse_to_2d=True
    )
    assert mask_2d_ds_chunked.chunksizes["ping_time"][0] == 400
    assert mask_2d_ds_chunked.chunksizes["depth"][0] == 100
    assert mask_2d_ds_chunked.equals(mask_2d_ds_computed)
    assert mask_2d_points_chunked.equals(mask_2d_points_computed)
