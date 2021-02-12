import os
import numpy as np
from ..convert.utils import parse_time
from ..convert.ecs_parser import CalibrationParser
from ..convert.evl_parser import LineParser

data_dir = './echoregions/test_data/ek60/'
output_csv = data_dir + 'output_CSV/'
output_json = data_dir + 'output_JSON/'


def test_parse_time():
    timestamp = 'D20170625T1539223320'
    assert parse_time(timestamp) == np.datetime64('2017-06-25T15:39:22.3320')

def test_plotting_points():
    evl_paths = data_dir + 'x1.bottom.evl'
    l_parser = LineParser(evl_paths)
    l_parser.to_json(output_json)
    evl = l_parser.convert_points(l_parser.output_data['x1.bottom']['points'])
    evl_points = np.array(l_parser.points_dict_to_list(evl))
    x = np.array(evl_points[:, 0], dtype=np.datetime64)
    y = evl_points[:, 1]
    assert len(x) == 13764
    assert len(y) == 13764

    os.remove(l_parser.output_path)
    os.rmdir(output_json)

def test_convert_ecs():
    ecs_path = data_dir + 'Summer2017_JuneCal_3freq.ecs'

    parser = CalibrationParser(ecs_path)
    parser.parse_file(ignore_comments=True)
    parser.to_csv(output_csv)
    parser.to_json(output_json)

    for path in parser.output_path:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


def test_convert_evl():
    evl_path = data_dir + 'x1.bottom.evl'
    parser = LineParser(evl_path)
    parser.parse_file()
    parser.to_csv(output_csv)
    parser.to_json(output_json)

    for path in parser.output_path:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)
