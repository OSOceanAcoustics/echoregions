import datetime as dt
import numpy as np

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
