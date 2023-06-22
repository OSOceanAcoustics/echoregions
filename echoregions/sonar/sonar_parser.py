import xarray as xr
from xarray import DataArray, Dataset

from ..utils.io import check_file


def parse_sonar_file(input_file: str) -> DataArray:
    """
    Parses either an .nc or .zarr sonar file.

    Arguments:
        input_file: str; represents input sonar file.

    Returns:
        da_Sv: DataArray; represents DataArray for sonar file.
    """

    # Check for validity of input_file.
    check_file(input_file, ["NC", "ZARR"])
    if input_file.endswith(".zarr"):
        ds_Sv = xr.open_zarr(input_file)
    else:
        ds_Sv = xr.open_dataset(input_file)
    da_Sv = check_ds_Sv(ds_Sv)
    return da_Sv


def check_ds_Sv(ds_Sv: Dataset) -> DataArray:
    """
    Checks ds_Sv and returns Sv DataArray.

    Arguments:
        input_file: str; represents input sonar file.

    Returns:
        da_Sv: DataArray; contains ping and depth data.
    """

    # Check if file is a Dataset.
    if type(ds_Sv) != Dataset:
        raise TypeError(f"ds_Sv.Sv is of type {type(ds_Sv)}. Must be of type Dataset.")

    # Extract Sv DataArray from it and return error if not found.
    try:
        da_Sv = ds_Sv.Sv
    except UnboundLocalError:
        raise UnboundLocalError(
            "There does not exist a data array Sv in the input sonar file."
        )
    finally:
        # Check if is of type DataArray
        if type(da_Sv) == DataArray:
            # Checks dimensions of da_Sv
            if (da_Sv.dims) == ("ping_time", "depth"):
                # Drop unnecessary variables/coordinates
                da_Sv = da_Sv.drop_vars(["channel", "range_sample"])
                # Reorder coords
                da_Sv = da_Sv.transpose()
                # TODO: Remove mask that exists in plotting? Hidden variable value?
                return da_Sv
            elif (da_Sv.dims) == ("channel", "ping_time", "range_sample"):
                # Create depth coordinate:
                echo_range = ds_Sv["echo_range"].isel(channel=0, ping_time=0)
                # Assuming water levels are same for different frequencies and location_time
                depth = ds_Sv["water_level"].isel(channel=0, ping_time=0) + echo_range
                depth = depth.drop_vars("channel")
                # Creating a new depth dimension
                ds_Sv["depth"] = depth
                ds_Sv = ds_Sv.swap_dims({"range_sample": "depth"})
                manipulated_da_Sv = ds_Sv.Sv.isel(channel=0).drop_vars(
                    ["channel", "range_sample"]
                )
                # Reorder coords
                manipulated_da_Sv = manipulated_da_Sv.transpose()
                return manipulated_da_Sv
            else:
                raise ValueError(
                    f"Input Sv data array has dimensions {da_Sv.dims}. Must have dimensions \
                    ('ping_time', 'depth) or dimensions ('channel', 'ping_time', 'range_sample')"
                )
        else:
            raise TypeError(
                f"da_Sv.Sv is of type {type(da_Sv)}. Must be of type DataArray."
            )
