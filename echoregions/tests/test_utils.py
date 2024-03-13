from pathlib import Path

import numpy as np
import pytest

from echoregions.utils.io import check_file, validate_path
from echoregions.utils.time import parse_time

DATA_DIR = Path("./echoregions/test_data/")
EVR_PATH = DATA_DIR / "transect.evr"
EVL_PATH = DATA_DIR / "transect.evl"
EVR_PATH_DNE = DATA_DIR / "transect_DNE.evr"
EVL_PATH_DNE = DATA_DIR / "transect_DNE.evl"


@pytest.mark.utils
def test_parse_time() -> None:
    """
    Test converting Echoview datetime string in EVR/EVL to numpy datetime64.
    """
    timestamp = "20170625 1539223320"
    assert parse_time(timestamp) == np.datetime64("2017-06-25T15:39:22.3320")


@pytest.mark.utils
def test_check_file_errors():
    """
    Test for check file errors that may arise from improper usage of read_evl and read_evr.
    """

    # Check invalid mismatch EVR path and EVL format
    with pytest.raises(ValueError):
        check_file(EVR_PATH.__str__(), ".EVL")
    # Check invalid mismatch EVL path and EVR format
    with pytest.raises(ValueError):
        check_file(EVL_PATH.__str__(), ".EVR")
    # Check does not exist EVR path
    with pytest.raises(ValueError):
        check_file(EVR_PATH_DNE.__str__(), ".EVR")
    # Check does not exist EVL path
    with pytest.raises(ValueError):
        check_file(EVL_PATH_DNE.__str__(), ".EVL")


@pytest.mark.utils
def test_validate_path_errors():
    """
    Test for check file errors that may arise from improper usage of to_csv and to_json.
    """

    # Check for no path given
    with pytest.raises(ValueError):
        validate_path(ext="EVL")
    # Check for no extension given
    with pytest.raises(ValueError):
        validate_path(save_path=EVL_PATH.__str__(), input_file=EVL_PATH_DNE.__str__())
    # Check for no filename given
    with pytest.raises(ValueError):
        validate_path(save_path=EVL_PATH.__str__(), ext="EVL")
    with pytest.raises(ValueError):
        # Check for mismatch of file name and file format
        validate_path(save_path=EVL_PATH.__str__(), input_file=EVL_PATH_DNE.__str__(), ext="EVR")
