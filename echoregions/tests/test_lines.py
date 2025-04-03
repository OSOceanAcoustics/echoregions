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

    da_Sv = xr.open_zarr(ZARR_PATH)["Sv"].compute()
    return da_Sv


@pytest.mark.lines
def test_lines_csv(lines_fixture: Lines) -> None:
    """
    Ensures that read_lines_csv provides the same Lines object
    as read_evl.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
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
def test_to_evl() -> None:
    """
    Tests that when we save a `Lines` object to `.evl` and read
    back that `.evl` file, we end up with the same inner dataframe.
    """
    # Get Lines object and DataFrame
    lines_1 = er.read_evl(DATA_DIR / "transect.evl")
    lines_1_df = lines_1.data

    # Send to `.evl` file
    evl_file_path = DATA_DIR / "lines_to_evl_file.evl"
    lines_1.to_evl(evl_file_path)

    # Read back `.lines` file and extract DataFrame
    lines_2 = er.read_evl(evl_file_path)
    lines_2_df = lines_2.data

    # Check that the dataframes are equal everywhere (not including the file name)
    assert lines_1_df.drop("file_name", axis=1).equals(lines_2_df.drop("file_name", axis=1))

    # Delete the file
    evl_file_path.unlink()


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
    Test plotting Lines to check for no raised errors.

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
    lines2 = lines_2.replace_nan_depth(inplace=False)
    assert lines2.loc[0, "depth"] == 20


@pytest.mark.lines
def test_lines_bottom_mask_operation_regionmask(
    lines_fixture: Lines, da_Sv_fixture: DataArray
) -> None:
    """
    Tests lines.bottom_mask with operation 'regionmask' on an overlapping (over time) evl file.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Compute mask and bottom points
    bottom_mask, bottom_points = lines_fixture.bottom_mask(da_Sv_fixture, operation="regionmask")

    # Compute unique values
    unique_values = np.unique(bottom_mask.compute().data, return_counts=True)

    # Extract counts and values
    values = unique_values[0]
    counts = unique_values[1]

    # Assert that there are more masked points then there are unmasked points
    assert values[0] == 0 and values[1] == 1
    assert counts[0] > counts[1]

    # Check time dimension equality/inequality
    # Bottom points shouldn't align time wise with bottom_mask / da_Sv since we don't do
    # time alignment for the regionmask operation. We do time alignment in the above_below
    # operation (demonstrated in the test below this one).
    assert len(bottom_points) != len(bottom_mask["ping_time"]) == len(da_Sv_fixture["ping_time"])

    # Assert that time is datetime64
    assert pd.api.types.is_datetime64_any_dtype(bottom_points["time"])

    # Assert that depth is float64
    assert pd.api.types.is_float_dtype(bottom_points["depth"])

    # Assuming bottom_points is your pandas DataFrame
    # Drop first and last two rows
    bottom_points_dropped = bottom_points.iloc[2:-2]

    # Create Lines object
    lines_2 = Lines(bottom_points_dropped, None, input_type="CSV")

    # Run lines masking to check if masking runs
    bottom_mask_2, bottom_points_2 = lines_2.bottom_mask(da_Sv_fixture)

    # Assert that these two masks are equal
    assert bottom_mask.equals(bottom_mask_2)

    # Assert that these two dataframes are equal
    assert bottom_points.equals(bottom_points_2)


@pytest.mark.lines
def test_lines_bottom_mask_operation_above_below(
    lines_fixture: Lines, da_Sv_fixture: DataArray
) -> None:
    """
    Tests lines.bottom_mask with operation 'above_below' on an overlapping (over time) evl file.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Compute mask and bottom points
    bottom_mask, bottom_points = lines_fixture.bottom_mask(
        da_Sv_fixture,
        operation="above_below",
        method="slinear",
        limit=5,
        limit_area=None,
        limit_direction="both",
    )

    # Compute unique values
    unique_values = np.unique(bottom_mask.compute().data, return_counts=True)

    # Extract counts and values
    values = unique_values[0]
    counts = unique_values[1]

    # Assert that there are more masked points then there are unmasked points
    assert values[0] == 0 and values[1] == 1
    assert counts[0] > counts[1]

    # Check time dimension equality
    assert len(bottom_points) == len(bottom_mask["ping_time"]) == len(da_Sv_fixture["ping_time"])

    # Assert that time is datetime64
    assert pd.api.types.is_datetime64_any_dtype(bottom_points["time"])

    # Assert that depth is float64
    assert pd.api.types.is_float_dtype(bottom_points["depth"])

    # Place bottom points in Lines object
    lines_2 = Lines(bottom_points, None, input_type="CSV")

    # Run lines masking to check if masking runs
    bottom_mask_2, bottom_points_2 = lines_2.bottom_mask(
        da_Sv_fixture,
        operation="above_below",
        method="slinear",
        limit=5,
        limit_area=None,
        limit_direction="both",
    )

    # Assert that these two masks are equal
    assert bottom_mask.equals(bottom_mask_2)

    # Assert that these two dataframes are equal
    assert bottom_points.equals(bottom_points_2)


@pytest.mark.lines
@pytest.mark.parametrize("operation", ["regionmask", "above_below"])
def test_lines_bottom_mask_empty(
    lines_fixture: Lines, da_Sv_fixture: DataArray, operation: str
) -> None:
    """
    Tests lines.bottom_mask on an empty evl file.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    operation : str
        Operation to perform ('regionmask' or 'above_below').
    """

    # Create empty lines object
    lines_fixture.data = lines_fixture.data[0:0]

    M, bottom_points_1 = lines_fixture.bottom_mask(da_Sv_fixture, operation=operation)

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
    _, bottom_points_2 = lines_2.bottom_mask(da_Sv_fixture, operation=operation)

    # Assert that bottom_points is empty
    assert bottom_points_2.empty


@pytest.mark.lines
def test_lines_bottom_mask_errors(lines_fixture: Lines, da_Sv_fixture: DataArray) -> None:
    """
    Tests lines.bottom_mask on an overlapping (over time) evl file with improper
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
        lines_fixture.bottom_mask(da_Sv_fixture.data)
    # Test invalid method argument.
    with pytest.raises(ValueError):
        lines_fixture.bottom_mask(da_Sv_fixture, operation="above_below", method="INVALID")
    # Test invalid limit area argument.
    with pytest.raises(ValueError):
        lines_fixture.bottom_mask(da_Sv_fixture, operation="above_below", limit_area="INVALID")
    # Test invalid operation argument.
    with pytest.raises(ValueError):
        lines_fixture.bottom_mask(da_Sv_fixture, operation="TEST")


@pytest.mark.lines
def test_chunked_bottom_mask(lines_fixture: Lines, da_Sv_fixture: DataArray) -> None:
    """
    Testing if chunked bottom masking and computed bottom masking (using `region_mask`) result
    in the same array and points, and checks the array chunks.

    Parameters
    ----------
    lines_fixture : Lines
        Object containing data of test EVL file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    # Set chunks
    chunk_dict = {"ping_time": 400, "depth": 100}

    # Create bottom masks, check chunks, and check that outputs are equal
    bottom_mask_chunked, bottom_points_chunked = lines_fixture.bottom_mask(
        da_Sv_fixture.chunk(chunk_dict), operation="regionmask"
    )
    bottom_mask_computed, bottom_points_computed = lines_fixture.bottom_mask(
        da_Sv_fixture.compute(), operation="regionmask"
    )
    assert bottom_mask_chunked.chunksizes["ping_time"][0] == 400
    assert bottom_mask_chunked.chunksizes["depth"][0] == 100
    assert bottom_mask_chunked.equals(bottom_mask_computed)
    assert bottom_points_chunked.equals(bottom_points_computed)
