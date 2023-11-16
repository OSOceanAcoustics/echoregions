"""
These core functions are what users will call on
at the beginning of their programming and will produce objects
that contain the majority of the functionality within echoregions.
"""

from _echoregions_version import version as __version__  # noqa

from .core import read_evl, read_evr, read_lines_csv, read_regions_csv
from .utils.api import convert_mask_2d_to_3d, convert_mask_3d_to_2d

__all__ = [
    "read_evl",
    "read_lines_csv",
    "read_evr",
    "read_regions_csv",
    "convert_mask_2d_to_3d",
    "convert_mask_3d_to_2d",
]  # noqa
