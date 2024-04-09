import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray

import echoregions as er
from echoregions.lines.lines import Lines

DATA_DIR = Path("./echoregions/test_data/")
EVL_PATH = DATA_DIR / "transect.evl"
ZARR_PATH = DATA_DIR / "transect.zarr"


@pytest.fixture(scope="function")
def lines_fixture() -> Lines:
    """
    Lines object fixture.

    Returns
    -------
    lines : Lines
        Object containing data of test EVL file.
    """

    lines = er.read_evl(EVL_PATH)
    return lines


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


@pytest.mark.lines
def test_lines_region_df(lines_fixture: Lines) -> None:
    """
    Ensures that read_region_df provides the same Regions2D object
    as read_evr.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test EVR file.
    """

    # Get Lines object and DataFrame
    lines_1 = lines_fixture
    lines_1_df = lines_1.data

    # Send to CSV
    csv_file_path = DATA_DIR / "lines_to_csv_file.csv"
    lines_1.to_csv(csv_file_path)

    # Read Lines CSV and extract DataFrame
    lines_2 = er.read_lines_csv(csv_file_path)
    lines_2_df = lines_2.data

    # Check equality between the elements in both depth columns
    assert np.all([d_1 == d_2 for d_1, d_2 in zip(lines_1_df["depth"], lines_2_df["depth"])])

    # Check equality between the elements in both time columns
    assert np.all([t_1 == t_2 for t_1, t_2 in zip(lines_1_df["time"], lines_2_df["time"])])

    # Delete the file
    csv_file_path.unlink()


@pytest.mark.lines
def test_lines_parsing(lines_fixture: Lines) -> None:
    """
    Test parsing of lines via checking individual values and aggregate values.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    """

    # Get dataframe
    df_lines = lines_fixture.data

    # Check shape
    assert df_lines.shape == (3171, 7)

    # Check individual values of specific row
    assert df_lines.loc[4]["file_name"] == "transect.evl"
    assert df_lines.loc[4]["file_type"] == "EVBD"
    assert df_lines.loc[4]["evl_file_format_version"] == "3"
    assert df_lines.loc[4]["echoview_version"] == "13.0.378.44817"
    assert df_lines.loc[4]["time"] == pd.to_datetime("2019-07-02 18:39:46.728000")
    assert df_lines.loc[4]["depth"] == 442.551006
    assert df_lines.loc[4]["status"] == "3"


@pytest.mark.lines
def test_evl_to_file(lines_fixture: Lines) -> None:
    """
    Test EVL to csv and to json; Creates and removes EVL .csv and .json objects.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    """

    # Get output paths
    output_csv = DATA_DIR / "output_CSV/"
    output_json = DATA_DIR / "output_JSON/"

    # Create CSV and JSON files
    lines_fixture.to_csv(output_csv)
    lines_fixture.to_json(output_json)

    # Remove files
    for path in lines_fixture.output_file:
        assert os.path.exists(path)
        os.remove(path)

    # Remove directories
    os.rmdir(output_csv)
    os.rmdir(output_json)


@pytest.mark.lines
def test_plot(lines_fixture: Lines) -> None:
    """
    Test plotting Lines with options.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    """

    start_date = pd.to_datetime("2019-07-02")
    end_date = pd.to_datetime("2019-07-03")
    lines_fixture.plot(
        start_time=start_date,
        end_time=end_date,
        max_depth=800,
        fill_between=True,
    )


@pytest.mark.lines
def test_plot_type_error(lines_fixture: Lines) -> None:
    """
    Test plotting lines errors.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    """

    start_date = pd.to_datetime("2019-07-02")
    end_date = pd.to_datetime("2019-07-03")
    bad_start_date = "2017-06-25"
    bad_end_date = "2017-06-26"
    with pytest.raises(TypeError):
        lines_fixture.plot(
            start_time=bad_start_date,
            end_time=bad_end_date,
            max_depth=800,
            fill_between=True,
        )
    with pytest.raises(TypeError):
        lines_fixture.plot(
            bad_start_time=bad_start_date,
            bad_end_time=end_date,
            max_depth=800,
            fill_between=True,
        )
    with pytest.raises(TypeError):
        lines_fixture.plot(
            bad_start_time=start_date,
            bad_end_time=bad_end_date,
            max_depth=800,
            fill_between=True,
        )


@pytest.mark.lines
def test_replace_nan_depth() -> None:
    """
    Test replacing NaN values in line for both inplace=True and inplace=False.
    """

    lines_1 = er.read_evl(EVL_PATH, nan_depth_value=20)
    lines_1.data.loc[0, "depth"] = -10000.99  # Replace a value with the one used for nans
    lines_1.replace_nan_depth(inplace=True)
    assert lines_1.data.loc[0, "depth"] == 20

    lines_2 = er.read_evl(EVL_PATH, nan_depth_value=20)
    lines_2.data.loc[0, "depth"] = -10000.99  # Replace a value with the one used for nans
    regions = lines_2.replace_nan_depth(inplace=False)
    assert regions.loc[0, "depth"] == 20


@pytest.mark.lines
def test_lines_mask(lines_fixture: Lines, da_Sv_fixture: DataArray) -> None:
    """
    Tests lines_mask on an overlapping (over time) evl file.
    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    M, bottom_points = lines_fixture.mask(
        da_Sv_fixture.isel(channel=0),
        method="slinear",
        limit=5,
        limit_area=None,
        limit_direction="both",
    )

    # Compute unique values
    unique_values = np.unique(M.compute().data, return_counts=True)

    # Extract counts and values
    values = unique_values[0]
    counts = unique_values[1]

    # Assert that there are more masked points then there are unmasked points
    assert values[0] == 0 and values[1] == 1
    assert counts[0] > counts[1]

    # Assert that time is datetime64
    assert pd.api.types.is_datetime64_any_dtype(bottom_points["time"])

    # Assert that depth is float64
    assert pd.api.types.is_float_dtype(bottom_points["depth"])

    # Place bottom points in Lines object
    lines_2 = Lines(bottom_points, None, input_type="CSV")

    # Run lines masking to check if masking runs
    _, _ = lines_2.mask(da_Sv_fixture.isel(channel=0))


@pytest.mark.lines
def test_lines_mask_empty(lines_fixture: Lines, da_Sv_fixture: DataArray) -> None:
    """
    Tests lines_mask on an empty evl file.
    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Create empty lines object
    lines_fixture.data = lines_fixture.data[0:0]

    M, bottom_points_1 = lines_fixture.mask(da_Sv_fixture.isel(channel=0))

    # Compute unique values
    unique_values = np.unique(M.compute().data, return_counts=True)

    # Extract counts and values
    values = unique_values[0]
    counts = unique_values[1]

    # Assert that the only unique value is 0
    assert len(values) == 1 and len(counts) == 1
    assert values[0] == 0

    # Assert that bottom_points is empty
    assert bottom_points_1.empty

    # Use bottom points to create Lines object
    lines_2 = Lines(bottom_points_1, None, input_type="CSV")

    # Run lines masking to check if masking runs
    _, bottom_points_2 = lines_2.mask(da_Sv_fixture.isel(channel=0))

    # Assert that bottom_points is empty
    assert bottom_points_2.empty


@pytest.mark.lines
def test_lines_mask_errors(lines_fixture: Lines, da_Sv_fixture: DataArray) -> None:
    """
    Tests lines_mask on an overlapping (over time) evl file with improper
    input arguments.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """

    # Test invalid da_Sv argument.
    with pytest.raises(TypeError):
        lines_fixture.mask(da_Sv_fixture.isel(channel=0).data)
    # Test invalid method argument.
    with pytest.raises(ValueError):
        lines_fixture.mask(da_Sv_fixture.isel(channel=0), method="INVALID")
    # Test invalid limit area argument.
    with pytest.raises(ValueError):
        lines_fixture.mask(da_Sv_fixture.isel(channel=0), limit_area="INVALID")
