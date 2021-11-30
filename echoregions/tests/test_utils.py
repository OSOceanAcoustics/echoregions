import numpy as np

from echoregions.convert.utils import parse_simrad_fname_time, parse_time


def test_parse_time():
    # Test converting EV datetime string to numpy datetime64
    timestamp = "20170625 1539223320"
    assert parse_time(timestamp) == np.datetime64("2017-06-25T15:39:22.3320")


def test_parse_filename_time():
    raw_fname = "Summer2017-D20170625-T124834.raw"
    assert parse_simrad_fname_time(raw_fname) == np.datetime64(
        "2017-06-25T12:48:34.0000"
    )
