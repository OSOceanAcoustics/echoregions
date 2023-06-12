import os
import pytest
from pandas import DataFrame, Series
import echoregions as er

from ..convert.api import merge
from ..convert.ev_parser import EvParserBase

data_dir = "./echoregions/test_data/"
output_csv = data_dir + "output_CSV/"
output_json = data_dir + "output_JSON/"


def test_convert_evl():
    """
    Test converting an Echoview lines files (.EVL).
    """
    evl_path = data_dir + "x1.evl"
    evl = er.read_evl(evl_path)
    evl.to_csv(output_csv)
    evl.to_json(output_json)

    for path in evl.output_file:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


def test_convert_evr():
    """
    Test converting an Echoview 2D Regions files (.EVR).
    """
    evr_path = data_dir + "x1.evr"
    evr = er.read_evr(evr_path)
    evr.to_csv(output_csv)

    assert os.path.exists(evr.output_file)
    os.remove(evr.output_file)

    os.rmdir(output_csv)


def test_ev_to_csv_type_error():
    """
    Test ev to_csv type checking.
    """
    empty_ev = EvParserBase(data_dir + "x1.evl", "EVL")

    with pytest.raises(TypeError):
        empty_series = Series()
        empty_ev.to_csv(empty_series)
    with pytest.raises(TypeError):
        empty_list = []
        empty_ev.to_csv(empty_list)


def test_merge_type_checking():
    """
    Test merge type checking functionality.
    """
    with pytest.raises(ValueError):
        merge([])
    with pytest.raises(TypeError):
        merge([DataFrame()])
    with pytest.raises(TypeError):
        merge([Series()])
