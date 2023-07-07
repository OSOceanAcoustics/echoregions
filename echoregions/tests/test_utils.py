import numpy as np
import pytest
from pandas import DataFrame, Series

from ..utils.api import merge
from ..utils.time import parse_simrad_fname_time, parse_time

# TODO Add tests for check_file, from_JSON, validate_path


@pytest.mark.utils
def test_parse_time() -> None:
    """
    Test converting Echoview datetime string in EVR/EVL to numpy datetime64.
    """
    timestamp = "20170625 1539223320"
    assert parse_time(timestamp) == np.datetime64("2017-06-25T15:39:22.3320")


@pytest.mark.utils
def test_parse_filename_time() -> None:
    """
    Test parsing Simrad-style filename for timestamp.
    """

    raw_fname = "Summer2017-D20170625-T124834.raw"
    assert parse_simrad_fname_time(raw_fname) == np.datetime64(
        "2017-06-25T12:48:34.0000"
    )


@pytest.mark.utils
def test_merge_type_checking() -> None:
    """
    Test merge type checking functionality.
    """

    with pytest.raises(ValueError):
        merge([])
    with pytest.raises(TypeError):
        merge([DataFrame()])
    with pytest.raises(TypeError):
        merge([Series()])
