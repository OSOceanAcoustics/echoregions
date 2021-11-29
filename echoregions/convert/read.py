from ..formats.regions2d import Regions2D
from ..formats.lines import Lines
from ..convert.ecs_parser import CalibrationParser


def read_evr(
    filepath,
    offset=0,
    min_depth=None,
    max_depth=None,
    depth=None
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
        depth=depth
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


def read_ecs(
    filepath,
    ignore_comments=True
) -> "CalibrationParser":
    """Read an ECS file into a CalibrationParser object.

    Parameters
    ----------
    filepath : str, Path object
        A valid path to an EVL file
    ignore_comments : bool
        Skip over lines of the ECS file that are commented out

    Returns
    -------
    CalibrationParser
        Object that contains ECS data with methods for saving to file.
    """
    return CalibrationParser(input_file=filepath, parse=True, ignore_comments=ignore_comments)
