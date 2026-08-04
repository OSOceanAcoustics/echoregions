import warnings
from typing import List, Union

import numpy as np
import pandas as pd
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


def merge(objects: List, reindex_ids: bool = False):
    """Merge a list of echoregion objects.

    Parameters
    ----------
    objects : list
        A list of one or more `Lines` or `Regions2D` objects.
    reindex_ids : bool, default False
        Only used for `Regions2D` merges. If `True`, it renumbers `region_id`
        in the merged result from `0` upward. For `Lines` merges, it raises
        an error.

    Returns
    -------
    merged_obj : Lines or Regions2D
        A merged object of the same class as the inputs.
    """
    # Avoid circular imports by importing here instead of at beginning of file
    from ..lines.lines import Lines
    from ..regions2d.regions2d import Regions2D

    if not isinstance(objects, list):
        raise TypeError(
            f"Invalid objects Type: {type(objects)}. Must be of type List[Lines | Regions2D]"
        )
    if len(objects) == 0:
        raise ValueError("objects must contain elements. objects sent in is empty.")

    if not all(isinstance(obj, (Lines, Regions2D)) for obj in objects):
        raise TypeError("Invalid elements in objects. Must be of type Lines or Regions2D")

    if not all(isinstance(obj, type(objects[0])) for obj in objects):
        raise TypeError(
            "All objects in the list must be the same class: all Lines or all Regions2D"
        )

    # TODO: consider how to record source info for 'mixed' merges. A single
    # output_file name would be unclear when the inputs come from different
    # source types like DataFrame and .evr/.evl case. This gets more complicated
    # if we consider the mask case.

    if reindex_ids and not isinstance(objects[0], Regions2D):
        raise ValueError("reindex_ids=True is only supported for Regions2D merges.")

    if isinstance(objects[0], Lines):
        merged_data = pd.concat([obj.data for obj in objects], ignore_index=True)
        # Build the merged object directly so we do not run parsing
        merged_obj = Lines.__new__(Lines)
        merged_obj.input_file = objects[0].input_file
        merged_obj.data = merged_data
        merged_obj.output_file = []
        merged_obj._nan_depth_value = objects[0]._nan_depth_value
        return merged_obj

    merged_data = pd.concat([obj.data for obj in objects], ignore_index=True)
    if reindex_ids:
        merged_data["region_id"] = range(len(merged_data))

    # Build the merged object directly so we do not run parsing
    merged_obj = Regions2D.__new__(Regions2D)
    merged_obj.input_file = objects[0].input_file
    merged_obj.data = merged_data
    merged_obj.output_file = []
    merged_obj.min_depth = objects[0].min_depth
    merged_obj.max_depth = objects[0].max_depth
    return merged_obj
