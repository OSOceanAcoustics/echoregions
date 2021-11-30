import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SIMRAD_FILENAME_MATCHER = re.compile(
    r"(?P<survey>.+)?-?D(?P<date>\w{1,8})-T(?P<time>\w{1,6})-?(?P<postfix>\w+)?\..+"
)


def from_JSON(j):
    """Opens a JSON file

    Parameters
    ----------
    j : str
        Valid JSON string or path to JSON file
    """
    if os.path.isfile(j):
        with open(j, "r") as f:
            data_dict = json.load(f)
    else:
        try:
            data_dict = json.loads(j)
        except json.decoder.JSONDecodeError:
            raise ValueError("Invalid JSON string")
    return data_dict


def validate_path(save_path=None, input_file=None, ext=".json"):
    # Check if save_path is specified.
    # If not try to create one with the input_file and ext

    if save_path is None:
        if input_file is None:
            raise ValueError("No paths given")
        elif ext is None:
            raise ValueError("No extension given")
        else:
            input_file = Path(input_file)
            save_path = input_file.parent / (input_file.stem + ext)
    # If save path is specified, check if folders need to be made
    else:
        save_path = Path(save_path)
        # If save path is a directory, use name of input file
        if save_path.suffix == "":
            if input_file is None:
                raise ValueError("No filename given")
            else:
                input_file = Path(input_file)
                save_path = save_path / (input_file.stem + ext)

    # Check if extension of save path matches desired file format
    if save_path.suffix.lower() != ext.lower():
        raise ValueError(f"{save_path} is not a {ext} file")

    # Create directories if they do not exist
    if not save_path.parent.is_dir():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    return str(save_path)


def parse_time(ev_time, datetime_format="%Y%m%d %H%M%S%f", unix=False):
    """Convert EV datetime to a numpy datetime64 object

    Parameters
    ----------
    ev_time : str, list
        EV datetime string or list of these
    datetime_format : str
        Format of datestring to be used with datetime strptime
        in CCYYMMDD HHmmSSssss format
    unix : bool, default False
        Output the time in the unix time format

    Returns
    -------
    np.datetime64 or float
        converted input datetime
    """
    if isinstance(ev_time, np.ndarray) and np.issubdtype(
        ev_time.dtype, "datetime64[ms]"
    ):
        return ev_time
    elif not isinstance(ev_time, str) and not isinstance(ev_time, list):
        raise ValueError("'ev_time' must be type str or list")
    t = pd.to_datetime(ev_time, format=datetime_format)
    if unix:
        t = (t - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
    return t


def parse_simrad_fname_time(filenames):
    """Convert Simrad-style datetime to a numpy datetime64 object

    Parameters
    ----------
    filenames : str, list
        Simrad-style filename

    Returns
    -------
    datetime : np.datetime64
        converted input datetime
    """
    if isinstance(filenames, list):
        f_list = []
        for f in filenames:
            groups = SIMRAD_FILENAME_MATCHER.match(f)
            f_list.append(groups["date"] + " " + groups["time"])
    elif isinstance(filenames, str):
        groups = SIMRAD_FILENAME_MATCHER.match(filenames)
        f_list = [groups["date"] + " " + groups["time"]]
    else:
        raise ValueError("'filenames' must be type str or list")
    return parse_time(f_list, "%Y%m%d %H%M%S")
