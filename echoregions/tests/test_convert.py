import os

import echoregions as er

data_dir = "./echoregions/test_data/"
output_csv = data_dir + "output_CSV/"
output_json = data_dir + "output_JSON/"


def test_convert_ecs():
    """
    Test converting an Echoview calibration file (.ECS).
    """
    ecs_path = data_dir + "Summer2017_JuneCal_3freq.ecs"

    ecs = er.read_ecs(ecs_path)
    ecs.to_csv(output_csv)
    ecs.to_json(output_json)

    for path in ecs.output_file:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


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
