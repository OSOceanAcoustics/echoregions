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
    if not isinstance(objects, list):
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
    merged = dict(zip(merged_idx, merged_data))
    return merged
