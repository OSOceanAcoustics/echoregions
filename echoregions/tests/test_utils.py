from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import echoregions as er
from echoregions.lines.lines import Lines
from echoregions.regions2d.regions2d import Regions2D
from echoregions.utils.api import merge
from echoregions.utils.io import check_file, validate_path
from echoregions.utils.time import parse_time

DATA_DIR = Path("./echoregions/test_data/")
EVR_DIR = DATA_DIR / "evr"
EVL_DIR = DATA_DIR / "evl"
EVR_PATH = EVR_DIR / "transect.evr"
EVL_PATH = EVL_DIR / "transect.evl"
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


@pytest.mark.utils
def test_merge_lines_objects() -> None:
    """Test that merge keeps the expected line rows for a Lines-only merge."""
    first_line = er.read_evl(EVL_DIR / "transect_first_seafloor_point.evl")
    second_line = er.read_evl(EVL_DIR / "transect_second_seafloor_point.evl")

    merged = merge([first_line, second_line])

    assert isinstance(merged, Lines)
    assert merged.data.shape == (2, 7)
    assert merged.data["file_name"].tolist() == [
        "transect_first_seafloor_point.evl",
        "transect_second_seafloor_point.evl",
    ]
    assert merged.data.iloc[0]["time"] == pd.to_datetime("2019-07-02 18:39:41.321000")
    assert merged.data.iloc[0]["depth"] == pytest.approx(442.996834)
    assert merged.data.iloc[0]["status"] == "3"
    assert merged.data.iloc[1]["time"] == pd.to_datetime("2019-07-02 18:39:42.679000")
    assert merged.data.iloc[1]["depth"] == pytest.approx(437.818405)
    assert merged.data.iloc[1]["status"] == "3"

    with pytest.raises(ValueError):
        merge([first_line, second_line], reindex_ids=True)


@pytest.mark.utils
def test_merge_lines_invalid_input_raises() -> None:
    """Test that passing a direct object or non-list input into merge raises type error."""
    first_line = er.read_evl(EVL_DIR / "transect_first_seafloor_point.evl")
    second_line = er.read_evl(EVL_DIR / "transect_second_seafloor_point.evl")

    with pytest.raises(TypeError):
        merge(first_line)
    with pytest.raises(TypeError):
        merge([first_line, second_line, 123])


@pytest.mark.utils
def test_merge_regions2d_objects() -> None:
    """Test that merge keeps the expected row values for a Regions2D-only merge."""
    first_region = er.read_evr(EVR_DIR / "transect_first_region.evr")
    second_region = er.read_evr(EVR_DIR / "transect_second_subset.evr")

    merged = merge([first_region, second_region])

    assert isinstance(merged, Regions2D)
    assert merged.data.shape == (2, 22)
    assert merged.data["file_name"].tolist() == [
        "transect_first_region.evr",
        "transect_second_subset.evr",
    ]
    assert merged.data["region_id"].tolist() == [1, 2]
    assert merged.data["region_name"].tolist() == ["COM", "Com"]

    reindexed = merge([first_region, second_region], reindex_ids=True)
    assert reindexed.data["region_id"].tolist() == [0, 1]
    assert reindexed.data["region_name"].tolist() == ["COM", "Com"]


@pytest.mark.utils
def test_merge_regions2d_invalid_input_raises() -> None:
    """Test that passing a direct object or non-list input into merge raises type error."""
    first_region = er.read_evr(EVR_DIR / "transect_first_region.evr")
    second_region = er.read_evr(EVR_DIR / "transect_second_subset.evr")

    with pytest.raises(TypeError):
        merge(first_region)
    with pytest.raises(TypeError):
        merge([first_region, second_region, 123])
