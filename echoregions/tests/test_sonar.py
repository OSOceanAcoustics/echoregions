from pathlib import Path
from xarray import DataArray
import os
import pytest

import echoregions as er

DATA_DIR = Path("./echoregions/test_data/")
NC_FILE = "x1_test.nc"


def test_sonar_data():
    """
    Tests sonar data type.
    """
    sonar = er.read_nc(os.path.join(DATA_DIR, NC_FILE))

    assert type(sonar.data) == DataArray

def test_sonar_errors():
    """
    Tests sonar data errors to make sure they are properly thrown.
    """
    with pytest.raises(ValueError):
        sonar = er.read_nc(os.path.join(DATA_DIR, "x1_test.evl"))

    sonar = er.read_nc(os.path.join(DATA_DIR, NC_FILE))
    empty_dict = {}
    with pytest.raises(TypeError):
        sonar.data = empty_dict
