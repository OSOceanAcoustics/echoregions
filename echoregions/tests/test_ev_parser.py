import os

import numpy as np

import echoregions as er
from echoregions.convert.utils import parse_time

data_dir = "./echoregions/test_data/"
output_csv = data_dir + "output_CSV/"
output_json = data_dir + "output_JSON/"


def test_parse_time():
    # Test converting EV datetime string to numpy datetime64
    timestamp = "20170625 1539223320"
    assert parse_time(timestamp) == np.datetime64("2017-06-25T15:39:22.3320")


def test_convert_ecs():
    # Test converting an EV calibration file (ECS)
    ecs_path = data_dir + "Summer2017_JuneCal_3freq.ecs"

    ecs = er.read_ecs(ecs_path)
    ecs.parse_file(ignore_comments=True)
    ecs.to_csv(output_csv)
    ecs.to_json(output_json)

    for path in ecs.output_file:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


def test_convert_evl():
    # Test converting an EV lines files (EVL)
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
    # Test converting an EV 2D Regions files (EVR)
    evr_path = data_dir + "x1.evr"
    evr = er.read_evr(evr_path)
    evr.to_csv(output_csv)

    assert os.path.exists(evr.output_file)
    os.remove(evr.output_file)

    os.rmdir(output_csv)
