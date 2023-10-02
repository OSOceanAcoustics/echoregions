from .lines.lines import Lines
from .regions2d.regions2d import Regions2D


def read_evr(filepath: str, min_depth: float = 0.0, max_depth: float = 1000.0) -> Regions2D:
    """Read an EVR file into a Regions2D object.

    Parameters
    ----------
    filepath : str, Path object
        A valid path to an EVR file
    min_depth : float, default 0
        Depth value in meters to set -9999.99 depth edges to.
    max_depth : float, default 1000
        Depth value in meters to set 9999.99 depth edges to.

    Returns
    -------
    Regions2D
        Object that contains the EVR data and metadata with methods for saving to file.
    """
    return Regions2D(input_file=str(filepath), min_depth=min_depth, max_depth=max_depth)


def read_evl(filepath: str, nan_depth_value: float = None) -> Lines:
    """Read an EVL file into a Lines object.

    Parameters
    ----------
    filepath : str, Path object
        A valid path to an EVL file
    nan_depth_value : float, default ``None``
        Depth in meters to replace -10000.990000 ranges with.

    Returns
    -------
    Lines
        Object that contains EVL data and metadata with methods for saving to file.
    """
    return Lines(input_file=str(filepath), nan_depth_value=nan_depth_value)
