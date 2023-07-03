"""
The core functions are what users will call on
at the beginning of their programming and will produce objects
that contain the majority of the functionality within echoregions.
"""

from .core import read_evl, read_evr

__all__ = ["read_evl", "read_evr"]
