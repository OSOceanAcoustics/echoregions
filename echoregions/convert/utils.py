import re
import os
import json
import datetime as dt
import numpy as np

EK60_fname_pattern = r'(?P<survey>.+)?-?D(?P<date>\w{1,8})-T(?P<time>\w{1,6})-?(?P<postfix>\w+)?\.raw'

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

def parse_time(ev_time, datetime_format='D%Y%m%dT%H%M%S%f'):
    """Convert EV datetime to a numpy datetime64 object

    Parameters
    ----------
    ev_time : str
        EV datetime string
    datetime_format : str
        Format of datestring to be used with datetime strptime
        in CCYYMMDD HHmmSSssss format

    Returns
    -------
    datetime : np.datetime64
        converted input datetime

    Raises
    ------
    ValueError
        when ev_time is not a string
    """
    if isinstance(ev_time, np.ndarray) and np.issubdtype(ev_time.dtype, np.datetime64):
        return ev_time
    elif not isinstance(ev_time, str):
        raise ValueError("'ev_time' must be type str")
    return np.array(dt.datetime.strptime(ev_time, datetime_format), dtype=np.datetime64)

def parse_filetime(fname):
    groups = re.match(EK60_fname_pattern, fname).groups()
    return parse_time(f"D{groups[1]}T{groups[2]}", 'D%Y%m%dT%H%M%S')