import os
from pathlib import Path

import pytest
from xarray import DataArray

import echoregions as er

DATA_DIR = Path("./echoregions/test_data/")
NC_FILE = "x1_test.nc"
ZARR_FILE = "x1.zarr"


def test_nc_sonar_data():
    """
    Tests .nc sonar data type.
    """
    sonar = er.read_sonar(os.path.join(DATA_DIR, NC_FILE))

    assert type(sonar.data) == DataArray


def test_zarr_sonar_data():
    """
    Tests .zarr sonar data type.
    """
    sonar = er.read_sonar(os.path.join(DATA_DIR, ZARR_FILE))

    assert type(sonar.data) == DataArray


def test_sonar_errors():
    """
    Tests sonar data errors to make sure they are properly thrown.
    """
    with pytest.raises(ValueError):
        sonar = er.read_sonar(os.path.join(DATA_DIR, "x1_test.evl"))

    sonar = er.read_sonar(os.path.join(DATA_DIR, NC_FILE))
    empty_dict = {}
    with pytest.raises(TypeError):
        sonar.data = empty_dict
