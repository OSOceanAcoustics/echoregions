import datetime as dt
import os
import numpy as np
import json

EV_DATETIME_FORMAT = 'D%Y%m%dT%H%M%S%f'


def parse_time(ev_time):
    """Convert EV datetime to a numpy datetime64 object

    Parameters
    ----------
    ev_time : str
        EV datetime in CCYYMMDD HHmmSSssss format

    Returns
    -------
    datetime : np.datetime64
        converted input datetime

    Raises
    ------
    ValueError
        when ev_time is not a string
    """
    if not isinstance(ev_time, str):
        raise ValueError("'ev_time' must be type str")
    timestamp = np.array(dt.datetime.strptime(ev_time, EV_DATETIME_FORMAT), dtype=np.datetime64)
    return timestamp


def JSON_to_dict(j, convert_time=True):
    """Convert JSON to dict

    Parameters
    ----------
    j : str
        Valid JSON string or path to JSON file, defaults to True
    convert_time : bool
        Whether to convert EV time to datetime64

    Returns
    -------
    data : dict
        dicationary from JSON data

    Raises
    ------
    ValueError
        when j is not a valid echoregions JSON file or JSON string
    """

    if os.path.isfile(j):
        with open(j, 'r') as f:
            data_dict = json.load(f)
    else:
        try:
            data_dict = json.loads(j)
        except json.decoder.JSONDecodeError:
            raise ValueError("Invalid JSON string")

    if convert_time:
        # EVR format
        if 'regions' in data_dict:
            for rid, region in data_dict['regions'].items():
                for p, point in region['points'].items():
                    point = data_dict['regions'][rid]['points'][p]
                    point[0] = parse_time(point[0])
            return data_dict
        # EVL format
        elif 'points' in data_dict:
            for p, point in data_dict['points'].items():
                data_dict['points'][p]['x'] = parse_time(data_dict['points'][p]['x'])
            return data_dict
        else:
            raise ValueError("Invalid data format")
    else:
        return data_dict
