from ..formats.regions2d import Regions2D


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
    if not isinstance(objects, list) or not all(isinstance(o, Regions2D) for o in objects):
        return ValueError("`merge` takes a list of Regions2D objects")

    merged_idx = []
    merged_data = []
    for regions in objects:
        if not regions.output_data:
            raise ValueError("EVR file has not been parsed. Call `parse_file` first.")
        merged_data += regions.output_data['regions'].values()
        if reindex_ids:
            merged_idx += list(range(len(regions.output_data['regions'])))
        else:
            merged_idx += [f"{regions.output_data['metadata']['file_name']}_{r}"
                           for r in regions.region_ids]
    # Attach region information to region ids
    merged = dict(zip(merged_idx, merged_data))
    # Create new Regons2D object
    merged_obj = Regions2D()
    # Populate metadata
    merged_obj.output_data['metadata'] = objects[0].output_data['metadata']
    # Combine metadata of all Regions2D objects
    for field in objects[0].output_data['metadata'].keys():
        merged_obj.output_data['metadata'][field] = [o.output_data['metadata'][field] for o in objects]
    # Set region data
    merged_obj.output_data['regions'] = merged
    return merged_obj
