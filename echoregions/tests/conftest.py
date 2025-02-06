"""``pytest`` configuration."""

import pytest

from echoregions.testing import TEST_DATA_FOLDER


@pytest.fixture(scope="session")
def dump_output_dir():
    return TEST_DATA_FOLDER / "dump"


@pytest.fixture(scope="session")
def test_path():
    return {
        "ROOT": TEST_DATA_FOLDER,
    }
