import numpy as np
import xarray as xr


def write_evr(
    evr_path: str,
    mask: xr.DataArray,
    region_classification: str = "",
    echoview_version: str = "EVRG 7 12.0.341.42620",
):
    """
    Writes region data to an Echoview Region (*.evr) file.

    Parameters
    ----------
    evr_path : str
        Path to the output .evr file.
    mask : xr.DataArray
        A binary 2D mask containing 'ping_time' and 'depth' coordinates for contour mapping.
    mode : str
        Writer mode.
    region_classification : str
        Region classification. Assumes single type of region classification.
        # TODO alongside with the other arguments, make this something that allows multiple region classifications.
    echoview_version : str, optional
        Version information for Echoview, by default "EVRG 7 12.0.341.42620".

    Returns
    -------
    None
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "The 'write_evr' function requires that 'opencv-python' package is installed. "
            "Please install it using 'pip install opencv-python' and try again. Support for "
            "contour finding without using openCV will be added in a future release."
        )

    # Validate the mask
    if not isinstance(mask, xr.DataArray):
        raise TypeError("The 'mask' parameter must be an xarray.DataArray.")
    expected_coords = {"ping_time", "depth"}
    if set(mask.coords) != expected_coords:
        raise ValueError(
            f"The 'mask' must have only 'ping_time' and 'depth' as coordinates, but found {sorted(set(mask.dims))}."
        )
    if mask.isnull().any():
        raise ValueError(
            "The 'mask' contains NaN values. Please remove or fill them before proceeding."
        )
    unique_values = np.unique(mask)
    if (
        not np.array_equal(unique_values, [0.0, 1.0])
        and not np.array_equal(unique_values, [0.0])
        and not np.array_equal(unique_values, [1.0])
    ):
        raise ValueError("The 'mask' must be binary, containing only 0s and 1s.")

    # Compute contours
    binary_image = mask.transpose("ping_time", "depth").data.astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute number of regions
    number_of_regions = str(len(contours))

    # Write header
    with open(evr_path, "w") as f:
        f.write(echoview_version + "\n")
        f.write(number_of_regions + "\n")

    # Write each region
    for idx, contour in enumerate(contours):
        depths = mask.depth[contour[:, 0, 0]]
        times = mask.ping_time[contour[:, 0, 1]]
        region_id = idx + 1

        _write_region(
            evr_path=evr_path,
            depths=depths,
            times=times,
            region_id=region_id,
            region_classification=region_classification,
        )


def _write_region(
    evr_path: str, depths: np.ndarray, times: np.ndarray, region_id: int, region_classification: str
):
    """
    Helper function to write a single region's data to the .evr file.

    Parameters
    ----------
    evr_path : str
        Path to the output .evr file.
    depths : np.ndarray
        Array of depth values corresponding to the region.
    times : np.ndarray
        Array of time values corresponding to the region.
    region_id : int
        Unique identifier for the region.
    region_classification : str
        Region classification. Assumes single type of region classification.

    Returns
    -------
    None
    """
    # TODO some of these can potentially be user-specified
    region_version = "13"
    point_count = str(len(times))
    selected = "0"
    region_creation_type = "2"
    dummy = "-1"
    bounding_rectangle_calculated = "1"
    number_of_lines_of_notes = "0"
    number_of_lines_of_detection_settings = "0"
    region_type = "1"

    with open(evr_path, "a") as f:
        # Calculate bounding box
        left_x = (
            f"{min(times).dt.strftime('%Y%m%d').data} {min(times).dt.strftime('%H%M%S%f').data}"
        )
        top_y = str(max(depths).data)
        right_x = (
            f"{max(times).dt.strftime('%Y%m%d').data} {min(times).dt.strftime('%H%M%S%f').data}"
        )
        bottom_y = str(min(depths).data)
        bbox = f"{left_x} {top_y} {right_x} {bottom_y}"

        # Write region metadata
        f.write(
            f"\n{region_version} {point_count} {region_id} {selected} {region_creation_type} {dummy} {bounding_rectangle_calculated} {bbox}\n"
        )
        f.write(f"{number_of_lines_of_notes}\n")
        f.write(f"{number_of_lines_of_detection_settings}\n")
        f.write(f"{region_classification}\n")

        # Write points
        for date, time, depth in zip(times, times, depths):  # time = date here
            date_str = str(date.dt.strftime("%Y%m%d").data)
            time_str = str(time.dt.strftime("%H%M%S%f").data)[:-2]
            depth_str = str(depth.data)
            point = f"{date_str} {time_str} {depth_str}"
            f.write(f"{point} ")

        # Write region type and name
        f.write(f" {region_type}\n")
        f.write(f"{region_classification}{region_id}\n")
