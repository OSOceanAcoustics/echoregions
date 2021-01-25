import os
from ..convert.evr_parser import Region2DParser
from ..convert.ecs_parser import CalibrationParser
from ..convert.evl_parser import LineParser

data_dir = './echoregions/test_data/ek60/'
output_csv = data_dir + 'output_CSV/'
output_json = data_dir + 'output_JSON/'


def test_convert_evr():
    evr_paths = [data_dir + 'x1.evr',
                 data_dir + 'x3.evr']
    parser = Region2DParser(evr_paths)
    parser.to_json(output_json)
    parser.to_csv(output_csv)
    points = parser.get_points_from_region(parser.output_path[0], 4)
    assert points[0] == ('D20170625T1539223320', '9.2447583998')

    for path in parser.output_path:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


def test_convert_ecs():
    evr_path = data_dir + 'Summer2017_JuneCal_3freq.ecs'

    parser = CalibrationParser(evr_path)
    parser.parse_files()
    parser.to_csv(output_csv)
    parser.to_json(output_json)

    for path in parser.output_path:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


def test_convert_evl():
    evl_paths = [data_dir + 'x1.bottom.evl',
                 data_dir + 'x3.bottom.evl']
    parser = LineParser(evl_paths)
    parser.parse_files()
    parser.to_csv(output_csv)
    parser.to_json(output_json)

    for path in parser.output_path:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)
