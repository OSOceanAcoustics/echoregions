"""
These core functions are what users will call on
at the beginning of their programming and will produce objects
that contain the majority of the functionality within echoregions.
"""

# TODO fix this since it doesn't work with a normal pip install but
# pip installation with editable mode seems to generate it just fine.
# from _echoregions_version import version as __version__

from .core import read_evl, read_evr, read_lines_csv, read_regions_csv
from .regions2d.evr_writer import write_evr
from .utils.api import convert_mask_2d_to_3d, convert_mask_3d_to_2d

__version__ = "0.2.3"

__all__ = [
    "read_evl",
    "read_lines_csv",
    "read_evr",
    "read_regions_csv",
    "convert_mask_2d_to_3d",
    "convert_mask_3d_to_2d",
    "write_evr",
]  # noqa
