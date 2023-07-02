from typing import List

import numpy as np
import xarray as xr
from xarray import DataArray

from ..regions2d.regions2d import Regions2D


def convert_mask_2d_to_3d(M: DataArray):
    """Convert 2D Mask data into its 3D one-hot encoded form.
    Parameters
    ----------
    M : Data Array
        A DataArray with the data_var masked by a specified region.
    Returns
    -------
    A Dataset with a 3D DataArray with the data_var masked by the specified
    region in one-hot encoded form and a dictionary to remember original
    non-nan values.
    """
    # Get unique non nan values from the 2d mask
    unique_non_nan = list(np.unique(M.data[~np.isnan(M.data)]))

    # Create a list of mask objects from one-hot encoding M.data non-nan values
    # and a dictionary to remember said values from one-hot encoded data arrays.
    mask_list = []
    mask_dictionary = {"dims": "label", "data": []}
    for index, value in enumerate(unique_non_nan):
        mask_data = M.copy()
        mask_data_np = mask_data.data
        for index in np.ndindex(mask_data_np.shape):
            if mask_data_np[index] == value:
                mask_data_np[index] = 1
            else:
                mask_data_np[index] = 0
        mask_data.data = mask_data_np
        mask_list.append(mask_data)
        mask_dictionary_list = mask_dictionary["data"]
        mask_dictionary_list.append(value)
        mask_dictionary["data"] = mask_dictionary_list

    # Initialize empty Dataset
    mask_3d_ds = xr.Dataset()

    # Place one-hot encoded masks and dictionary in dataset
    mask_3d_da = xr.concat(mask_list, dim="label")
    mask_3d_ds["mask_3d"] = mask_3d_da
    mask_dictionary_da = xr.DataArray.from_dict(mask_dictionary)
    mask_3d_ds["mask_dictionary"] = mask_dictionary_da

    return mask_3d_ds


def convert_mask_3d_to_2d(mask_3d_ds: DataArray):
    """Convert 2D Mask data into its 3D one-hot encoded form.
    Parameters
    ----------
    mask_3d_ds : Dataset
        A Dataset with a 3D DataArray with the data_var masked by the specified
        region in one-hot encoded form and a dictionary to remember original
        non-nan values.
    Returns
    -------
    mask_2d_da: Data Array
        A 2D Data Array with appropriate mask values from mask_3d_ds.
    """
    # Get unique non nan values from the 2d mask
    unique_non_nan = list(mask_3d_ds.mask_dictionary.data)

    # Create copies and placeholder values for 2D and 3D mask objects
    mask_3d_da = mask_3d_ds.mask_3d.copy()
    np_mask_3d_da = mask_3d_da.data
    np_mask_2d_da = None

    # Iterate through 3D numpy array layers and set 1.0 to associated label values
    # dependent on which layer is being worked on, and sequentially add layers
    # together to form 2D numpy mask array.
    for index, label_value in enumerate(unique_non_nan):
        label_layer = np_mask_3d_da[index]
        label_layer[label_layer == 1.0] = label_value
        if index == 0:
            np_mask_2d_da = label_layer
        else:
            np_mask_2d_da = label_layer + np_mask_2d_da

    # Convert all such 0.0 values in 2D numpy mask array into NaN values.
    np_mask_2d_da[np_mask_2d_da == 0.0] = np.nan

    # Set new 2D mask data array
    mask_2d_da = xr.DataArray(
        data=np_mask_2d_da, dims=["depth", "ping_time"], coords=mask_3d_da.coords
    )

    return mask_2d_da


def merge(objects: List[Regions2D], reindex_ids: bool = False) -> Regions2D:
    # TODO currently deprecated must be fixed before further tests.
    """Merge echoregion objects.
    Currently only supports merging Regions2D objects.

    Parameters
    ----------
    objects : list
        a list collection of Regions2D objects

    Returns
    -------
    combined : Regions2D
        A Regions2D object with region ids prepended by the EVR original filename.
    """
    if isinstance(objects, list):
        if len(objects) == 0:
            raise ValueError("objects must contain elements. objects sent in is empty.")
        if not all(isinstance(o, Regions2D) for o in objects):
            raise TypeError("Invalid elements in objects. Must be of type Regions2D")
    else:
        raise TypeError(
            f"Invalid objects Type: {type(objects)}. Must be of type List[DataFrame]"
        )

    merged_idx = []
    merged_data = []
    for regions in objects:
        if not regions.data:
            raise ValueError("EVR file has not been parsed. Call `parse_file` first.")
        merged_data += regions.data["regions"].values()
        if reindex_ids:
            merged_idx += list(range(len(regions.data["regions"])))
        else:
            merged_idx += [
                f"{regions.data['metadata']['file_name']}_{r}"
                for r in regions.region_ids
            ]
    # Attach region information to region ids
    merged = dict(zip(merged_idx, merged_data))
    # Create new Regions2D object
    merged_obj = Regions2D()
    # Populate metadata
    merged_obj.data["metadata"] = objects[0].data["metadata"]
    # Combine metadata of all Regions2D objects
    for field in objects[0].data["metadata"].keys():
        merged_obj.data["metadata"][field] = [
            o.data["metadata"][field] for o in objects
        ]
    # Set region data
    merged_obj.data["regions"] = merged
    return merged_obj
