from pathlib import Path
from typing import Union

import pandas as pd

from .lines.lines import Lines
from .regions2d.regions2d import Regions2D


def read_evr(
    input_file: Union[str, Path], min_depth: float = 0.0, max_depth: float = 1000.0
) -> Regions2D:
    """Read an EVR file into a Regions2D object.

    Parameters
    ----------
    input_file : str, Path
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
    return Regions2D(
        input_file=str(input_file), min_depth=min_depth, max_depth=max_depth, input_type="EVR"
    )


def read_regions_csv(
    input_file: Union[str, Path, pd.DataFrame], min_depth: float = 0.0, max_depth: float = 1000.0
) -> Regions2D:
    """Read a region CSV into a Regions2D object.

    To use `er.read_region_csv`, the input dataframe/CSV must contain (at minimum) columns
    `region_id`, `depth`, and `time` where each `depth` entry is a 1-D float array and
    each `time` entry is a 1-D `datetime64[ns]` array.
    Please see the 'Saving to ".csv" and Reading From ".csv"' section in
    https://echoregions.readthedocs.io/en/latest/Regions2D_functionality.html
    for an example of this formatting.

    Parameters
    ----------
    input_file : str, Path, pd.DataFrame
        A valid path to a region CSV or the DataFrame representation of it.
    min_depth : float, default 0
        Depth value in meters to set -9999.99 depth edges to.
    max_depth : float, default 1000
        Depth value in meters to set 9999.99 depth edges to.

    Returns
    -------
    Regions2D
        Object that contains the EVR data and metadata with methods for saving to file.
    """
    if isinstance(input_file, Path):
        input_file = str(input_file)

    return Regions2D(
        input_file=input_file, min_depth=min_depth, max_depth=max_depth, input_type="CSV"
    )


def read_evl(input_file: Union[str, Path], nan_depth_value: float = None) -> Lines:
    """Read an EVL file into a Lines object.

    Parameters
    ----------
    input_file : str, Path
        A valid path to an EVL file
    nan_depth_value : float, default ``None``
        Depth in meters to replace -10000.990000 ranges with.

    Returns
    -------
    Lines
        Object that contains EVL data and metadata with methods for saving to file.
    """
    return Lines(input_file=str(input_file), nan_depth_value=nan_depth_value, input_type="EVL")


def read_lines_csv(
    input_file: Union[str, Path, pd.DataFrame], nan_depth_value: float = None
) -> Lines:
    """Read a lines CSV into a Lines object.

    To use `er.read_lines_csv`, the input dataframe/CSV must contain (at minimum) columns
    `depth` and `time` where each `depth` entry is a single float value and each `time`
    entry is a single `datetime64[ns]` value.
    Please see the 'Saving to ".csv" and Reading From ".csv"' section in
    https://echoregions.readthedocs.io/en/latest/Lines_functionality.html
    for an example of this formatting.

    Parameters
    ----------
    input_file : str, Path, pd.DataFrame
        A valid path to an lines CSV or the DataFrame representation of it.
    nan_depth_value : float, default ``None``
        Depth in meters to replace -10000.990000 ranges with.

    Returns
    -------
    Lines
        Object that contains EVL data and metadata with methods for saving to file.
    """
    if isinstance(input_file, Path):
        input_file = str(input_file)

    return Lines(input_file=input_file, nan_depth_value=nan_depth_value, input_type="CSV")
