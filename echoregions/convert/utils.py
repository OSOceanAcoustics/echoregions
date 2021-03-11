import re
import os
import json
import datetime as dt
import numpy as np
from pathlib import Path
import matplotlib

EK60_fname_pattern = r'(?P<survey>.+)?-?D(?P<date>\w{1,8})-T(?P<time>\w{1,6})-?(?P<postfix>\w+)?\..+'


def from_JSON(j):
    """ Opens a JSON file

    Parameters
    ----------
    j : str
        Valid JSON string or path to JSON file
    """
    if os.path.isfile(j):
        with open(j, 'r') as f:
            data_dict = json.load(f)
    else:
        try:
            data_dict = json.loads(j)
        except json.decoder.JSONDecodeError:
            raise ValueError("Invalid JSON string")
    return data_dict


def validate_path(save_path=None, input_file=None, ext='.json'):
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
        if save_path.suffix == '':
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


def parse_time(ev_time, datetime_format='D%Y%m%dT%H%M%S%f', unix=False):
    """Convert EV datetime to a numpy datetime64 object

    Parameters
    ----------
    ev_time : str
        EV datetime string
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
    if isinstance(ev_time, np.ndarray) and np.issubdtype(ev_time.dtype, np.datetime64):
        return ev_time
    elif not isinstance(ev_time, str):
        raise ValueError("'ev_time' must be type str")
    t = np.array(dt.datetime.strptime(ev_time, datetime_format), dtype=np.datetime64)
    if unix:
        t = matplotlib.dates.date2num(t)
    return t


def parse_filetime(fname):
    """Convert Simrad-style datetime to a numpy datetime64 object

    Parameters
    ----------
    filename : str
        Simrad-style filename

    Returns
    -------
    datetime : np.datetime64
        converted input datetime
    """
    groups = re.match(EK60_fname_pattern, fname).groups()
    return parse_time(f"D{groups[1]}T{groups[2]}", 'D%Y%m%dT%H%M%S')
