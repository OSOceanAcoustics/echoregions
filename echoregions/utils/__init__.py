"""
Utility functions that facilitate file and time processing.
"""

from .io import check_file, from_JSON
from .time import parse_simrad_fname_time, parse_time

__all__ = ["from_JSON", "check_file", "parse_simrad_fname_time", "parse_time"]
