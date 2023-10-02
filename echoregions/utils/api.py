import warnings
from typing import List, Union

import numpy as np
import xarray as xr
from xarray import Dataset


def convert_mask_2d_to_3d(mask_2d_ds: Dataset) -> Union[Dataset, None]:
    """
    Convert 2D multi-labeled mask data into its 3D one-hot encoded form.

    Parameters
    ----------
    mask_2d_ds : Dataset
        A dataset with the following:
            DataArray with the data_var masked by a specified region. Individual data
            points will be in the form of integers, demarking region_id of masked regions,
            and nan values, demarking non-masked areas.
            DataArray with mask labels corresponding to region_id values.

    Returns
    -------
    mask_3d_ds : Dataset
        A dataset with the following:
            A DataArray 3D mask where each layer of the mask will contain a 1s/0s mask for
            each unique label in the 2D mask. The layers will be labeled via region_id
            values extracted from 2d values.
            DataArray with mask labels corresponding to region_id values.
    """
    # Check if 'mask_2d' exists as a data variable
    if "mask_2d" not in mask_2d_ds:
        raise ValueError("The variable 'mask_2d' does not exist in the input dataset.")

    # Get unique non nan values from the 2d mask
    region_id = list(np.unique(mask_2d_ds.mask_2d.data[~np.isnan(mask_2d_ds.mask_2d.data)]))

    # Create a list of mask objects from one-hot encoding M.data non-nan values
    # and a dictionary to remember said values from one-hot encoded data arrays.
    # If unique_non_nan is None, make mask_dictionary None.
    if len(region_id) > 0:
        mask_list = []
        for _, value in enumerate(region_id):
            # Create new 1d mask
            new_mask_data = xr.where(mask_2d_ds.mask_2d == value, 1.0, 0.0)
            # Append data to mask_list
            mask_list.append(new_mask_data)
        # Concat mask list together to make 3d mask
        mask_3d_da = xr.concat(mask_list, dim=region_id)
        mask_3d_da = mask_3d_da.rename({"concat_dim": "region_id"})
        # Drop mask_2d
        mask_2d_ds = mask_2d_ds.drop_vars("mask_2d")
        # Set mask to mask_3d_da
        mask_2d_ds["mask_3d"] = mask_3d_da
        mask_3d_ds = mask_2d_ds
        return mask_3d_ds
    else:
        warnings.warn(
            "Returning No Mask. Empty 3D Mask cannot be converted to 2D Mask.",
            UserWarning,
        )
        return None


def convert_mask_3d_to_2d(mask_3d_ds: Dataset) -> Union[Dataset, None]:
    """
    Convert 3D one-hot encoded mask data into its 2D multi-labeled form.

    Parameters
    ----------
    mask_3d_ds : Dataset
        A dataset with the following:
            A DataArray 3D mask where each layer of the mask will contain a 1s/0s mask for
            each unique label in the 2D mask. The layers will be labeled via region_id
            values extracted from 2d values.
            DataArray with mask labels corresponding to region_id values.

    Returns
    -------
    mask_2d_ds : Dataset
        A dataset with the following:
            DataArray with the data_var masked by a specified region. Individual data
            points will be in the form of integers, demarking region_id of masked regions,
            and nan values, demarking non-masked areas.
            DataArray with mask labels corresponding to region_id values.
    """
    # Check if 'mask_2d' exists as a data variable
    if "mask_3d" not in mask_3d_ds:
        raise ValueError("The variable 'mask_3d' does not exist in the input dataset.")

    # Get region_id from the 3D Mask
    region_id = list(mask_3d_ds.mask_3d.region_id)

    # Check if there is overlap between layers.
    # TODO This code is also extremely slow. It is an O(n^2) operation that
    # can be parallelized due to the index operations being independent to
    # one another.
    if len(region_id) > 1:
        non_zero_indices_list = [
            np.transpose(np.nonzero(np_mask)) for np_mask in mask_3d_ds.mask_3d.data
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

    if len(region_id) > 0:
        # Iterate through 3D array layers and set 1.0 to associated label values
        # dependent on which layer is being worked on and create append layers to
        # form 2D mask array.
        for index, label_value in enumerate(region_id):
            label_layer = mask_3d_ds.mask_3d[index]
            label_layer = xr.where(label_layer == 1.0, label_value, 0.0)
            if index == 0:
                mask_2d_da = label_layer
            else:
                mask_2d_da = label_layer + mask_2d_da
        mask_2d_da = xr.where(mask_2d_da == 0.0, np.nan, mask_2d_da)

        # Setup mask_2d_ds
        mask_2d_ds = mask_3d_ds
        # Drop mask_2d
        mask_2d_ds = mask_2d_ds.drop_vars("mask_3d")
        # Set mask to mask_3d_da
        mask_2d_ds["mask_2d"] = mask_2d_da
        # Drop region_id coordinate if it exists
        if "region_id" in mask_2d_ds.mask_2d.coords:
            mask_2d_ds.mask_2d = mask_2d_ds.mask_2d.drop_vars(["region_id"])
        return mask_2d_ds
    else:
        warnings.warn(
            "Returning No Mask. Empty 3D Mask cannot be converted to 2D Mask.",
            UserWarning,
        )
        return None


def merge(objects: List, reindex_ids: bool = False):  # -> Regions2D:
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
    """
    if isinstance(objects, list):
        if len(objects) == 0:
            raise ValueError("objects must contain elements. objects sent in is empty.")
        if not all(isinstance(o, Regions2D) for o in objects):
            raise TypeError("Invalid elements in objects. Must be of type Regions2D")
    else:
        raise TypeError(f"Invalid objects Type: {type(objects)}. Must be of type List[DataFrame]")

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
                for r in regions.region_id
            ]
    # Attach region information to region ids
    merged = dict(zip(merged_idx, merged_data))
    # Create new Regions2D object
    merged_obj = Regions2D()
    # Populate metadata
    merged_obj.data["metadata"] = objects[0].data["metadata"]
    # Combine metadata of all Regions2D objects
    for field in objects[0].data["metadata"].keys():
        merged_obj.data["metadata"][field] = [o.data["metadata"][field] for o in objects]
    # Set region data
    merged_obj.data["regions"] = merged
    return merged_obj
    """
