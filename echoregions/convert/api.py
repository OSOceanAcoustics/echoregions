from ..formats.lines import Lines
from ..formats.regions2d import Regions2D


def read_evr(
    filepath, offset=0, min_depth=None, max_depth=None, depth=None
) -> "Regions2D":
    """Read an EVR file into a Regions2D object.

    Parameters
    ----------
    filepath : str, Path object
        A valid path to an EVR file
    offset : float, default 0
        Depth offset in meters
    min_depth : float, default ``None``
        Depth value in meters to set -9999.99 depth edges to.
    max_depth : float, default ``None``
        Depth value in meters to set 9999.99 depth edges to.
    depth : array, default ``None``
        Array of range values assumed to be monotonically increasing

    Returns
    -------
    Regions2D
        Object that contains the EVR data and metadata with methods for saving to file.
    """
    return Regions2D(
        input_file=str(filepath),
        parse=True,
        offset=offset,
        min_depth=min_depth,
        max_depth=max_depth,
        depth=depth,
    )


def read_evl(
    filepath,
    nan_depth_value=None,
    offset=0,
) -> "Lines":
    """Read an EVL file into a Lines object.

    Parameters
    ----------
    filepath : str, Path object
        A valid path to an EVL file
    nan_depth_value : float, default ``None``
        Depth in meters to replace -10000.990000 ranges with.
    offset : float, default 0
        Depth offset in meters

    Returns
    -------
    Lines
        Object that contains EVL data and metadata with methods for saving to file.
    """
    return Lines(
        input_file=str(filepath),
        parse=True,
        nan_depth_value=nan_depth_value,
        offset=offset,
    )


def merge(objects, reindex_ids=False):
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
    if not isinstance(objects, list) or not all(
        isinstance(o, Regions2D) for o in objects
    ):
        return ValueError("`merge` takes a list of Regions2D objects")

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
