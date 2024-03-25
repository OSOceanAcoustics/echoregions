import re
from typing import List, Union

import numpy as np
import pandas as pd
from pandas import Timestamp

SIMRAD_FILENAME_MATCHER = re.compile(
    r"(?P<survey>.+)?-?D(?P<date>\w{1,8})-T(?P<time>\w{1,6})-?(?P<postfix>\w+)?\..+"
)


def parse_time(
    ev_time: Union[List[str], str],
    datetime_format: str = "%Y%m%d %H%M%S%f",
    unix: bool = False,
) -> Timestamp:
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
    if isinstance(ev_time, np.ndarray) and np.issubdtype(ev_time.dtype, "datetime64[ms]"):
        return ev_time
    elif not isinstance(ev_time, str) and not isinstance(ev_time, list):
        raise ValueError("'ev_time' must be type str or list")
    t = pd.to_datetime(ev_time, format=datetime_format)
    if unix:
        t = (t - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
    return t
