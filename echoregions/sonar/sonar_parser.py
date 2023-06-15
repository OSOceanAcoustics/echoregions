import xarray as xr
from xarray import DataArray, Dataset

from ..utils.io import check_file

def parse_sonar_file(input_file: str) -> DataArray:
    """
    Parses .nc sonar files.

    Arguments:
        input_file: str; represents input sonar file.

    Returns:
        da_Sv: DataArray; represents DataArray for sonar file.
    """

    # Check for validity of input_file.
    check_file(input_file, "NC")
    ds_Sv = xr.open_dataset(input_file)
    da_Sv = check_ds_Sv(ds_Sv)
    return da_Sv
    
def check_ds_Sv(ds_Sv: Dataset) -> DataArray:
    """
    Checks ds_Sv and returns Sv DataArray.

    Arguments:
        input_file: str; represents input sonar file.

    Returns:
        da_Sv: DataArray; represents 
    """

    # Check if file is a Dataset.
    if type(ds_Sv) != Dataset:
        raise TypeError(f"ds_Sv.Sv is of type {type(ds_Sv)}. Must be of type Dataset.")

    # Extract Sv DataArray from it and return error if not found.
    try:
        da_Sv = ds_Sv.Sv
        # Check if is of type DataArray.
        if type(da_Sv) == DataArray:
            # Checks dimensions of da_Sv.
            if (da_Sv.dims) != ('ping_time', 'depth'):
                raise ValueError(f"da_Sv has dimensions {da_Sv.dims}. Must have dimensions \
                                 ('ping_time', 'depth)")
            return da_Sv
        else:
            raise TypeError(f"da_Sv.Sv is of type {type(da_Sv)}. Must be of type DataArray.")
    except:
        raise ValueError(".Sv is not found in ds_Sv.")
