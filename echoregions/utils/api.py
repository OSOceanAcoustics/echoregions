from typing import List

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from ..regions2d.regions2d import Regions2D


def convert_mask_2d_to_3d(mask_2d_da: DataArray) -> Dataset:
    """Convert 2D multi-labeled mask data into its 3D one-hot encoded form.
    Parameters
    ----------
    mask_2d_da: DataArray
        A DataArray with the data_var masked by a specified region. This data will
        be in the form of integer, demarking labels of masked regions, and nan values,
        demarking non-masked areas.
    Returns
    -------
    mask_3d_ds : Dataset
        A Dataset with a 3D DataArray/mask and each layer of the 3D mask will contain
        a 1s/0s mask for each unique label in the 2D mask. The layers will be labeled
        by a dictionary that maps the individual label layers of the 3D mask to an integer
        label in the 2D mask.
    Notes
    -----
    Emtpy dictionary data of mask_3d_ds means that there exists no masked
    values.
    """
    # Get unique non nan values from the 2d mask
    unique_non_nan = list(np.unique(mask_2d_da.data[~np.isnan(mask_2d_da.data)]))
    if len(unique_non_nan) == 0:
        unique_non_nan = None

    # Create a list of mask objects from one-hot encoding M.data non-nan values
    # and a dictionary to remember said values from one-hot encoded data arrays.
    # If unique_non_nan is None, make mask_dictionary None.
    mask_list = []
    mask_dictionary = {"dims": "label", "data": []}
    if unique_non_nan is not None:
        mask_dictionary = {"dims": "label", "data": []}
        for _, value in enumerate(unique_non_nan):
            # Create new 1d mask
            new_mask_data = xr.where(mask_2d_da == value, 1.0, 0.0)
            # Append data to mask_list and mask_dictionary
            mask_list.append(new_mask_data)
            mask_dictionary_list = mask_dictionary["data"]
            mask_dictionary_list.append(value)
            mask_dictionary["data"] = mask_dictionary_list
        mask_3d_da = xr.concat(mask_list, dim="label")
    else:
        mask_3d_da = xr.zeros_like(mask_2d_da)
    mask_dictionary_da = xr.DataArray.from_dict(mask_dictionary)

    # Initialize Dataset
    mask_3d_ds = xr.Dataset()
    mask_3d_ds["mask_3d"] = mask_3d_da
    mask_3d_ds["mask_dictionary"] = mask_dictionary_da
    return mask_3d_ds


def convert_mask_3d_to_2d(mask_3d_ds: Dataset) -> DataArray:
    """Convert 3D one-hot encoded mask data into its 2D multi-labeled form.
    Parameters
    ----------
    mask_3d_ds : Dataset
        A Dataset with a 3D DataArray with the data_var masked by the specified
        region in one-hot encoded form and a dictionary that will be used to map
        the individual label layers of the 3D mask to an integer label in the 2D mask.
        The 3D DataArray will be in the form of 1s/0s: masked areas, and
        non-masked areas.
    Returns
    -------
    mask_2d_da: DataArray
        A DataArray with the data_var masked by a specified region. This data will
        be in the form of integer, demarking labels of masked regions, and nan values,
        demarking non-masked areas.
    Notes
    -----
    Emtpy dictionary data of mask_3d_ds means that there exists no masked
    values.
    """
    # Get unique non nan values from the 2d mask
    unique_non_nan = list(mask_3d_ds.mask_dictionary.data)

    # Create copies and placeholder values for 2D and 3D mask objects
    mask_3d_da = mask_3d_ds.mask_3d.copy()

    # Check if there is overlap between layers.
    # TODO For now, overlap between layers will not be allowed.
    # Allowing overlapping layers can be explored in later development.
    if len(unique_non_nan) > 1:
        non_zero_indices_list = [
            np.transpose(np.nonzero(np_mask)) for np_mask in mask_3d_da.data
        ]
        for index_main, non_zero_indices_main in enumerate(non_zero_indices_list):
            main_set = set([tuple(x) for x in non_zero_indices_main])
            for index_sub, non_zero_indices_sub in enumerate(non_zero_indices_list):
                if index_main != index_sub:
                    # Compare non zero indice arrays and check for overlap
                    sub_set = set([tuple(x) for x in non_zero_indices_sub])
                    overlap = [x for x in main_set & sub_set]
                    if len(overlap) > 0:
                        raise ValueError(
                            "There exists overlapping values in the 3D mask."
                            " Overlapping values are not allowed."
                        )

    if len(unique_non_nan) > 0:
        # Iterate through 3D array layers and set 1.0 to associated label values
        # dependent on which layer is being worked on and create append layers to
        # form 2D mask array.
        for index, label_value in enumerate(unique_non_nan):
            label_layer = mask_3d_da[index]
            label_layer = xr.where(label_layer == 1.0, label_value, 0.0)
            if index == 0:
                mask_2d_da = label_layer
            else:
                mask_2d_da = label_layer + mask_2d_da
        mask_2d_da = xr.where(mask_2d_da == 0.0, np.nan, mask_2d_da)
    else:
        # In the case where unique_non_nan is empty, create all zeroes DataArray
        mask_2d_da = xr.full_like(mask_3d_da, np.nan)
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
