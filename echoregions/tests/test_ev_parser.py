import os
import numpy as np
from ..convert import utils
from ..convert.evr_parser import Region2DParser
from ..convert.ecs_parser import CalibrationParser
from ..convert.evl_parser import LineParser

data_dir = './echoregions/test_data/ek60/'
output_csv = data_dir + 'output_CSV/'
output_json = data_dir + 'output_JSON/'


def test_plotting_points():
    evr_paths = data_dir + 'x1.evr'
    r_parser = Region2DParser(evr_paths)
    r_parser.to_json(output_json)
    r_parser.set_range_edge_from_raw(data_dir + 'Summer2017-D20170625-T161209.raw')
    points = r_parser.get_points_from_region(r_parser.output_data['x1']['regions']['1'])
    evr_points = np.array(r_parser.convert_points(points, convert_time=True, convert_range_edges=True))
    x = np.array(evr_points[:, 0], dtype=np.datetime64)
    y = evr_points[:, 1]
    assert all(y == [r_parser.min_depth, r_parser.max_depth, r_parser.max_depth, r_parser.min_depth])
    # Plotting example
    # import matplotlib.pyplot as plt
    # plt.plot(x, y)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    evl_paths = data_dir + 'x1.bottom.evl'
    l_parser = LineParser(evl_paths)
    l_parser.to_json(output_json)
    evl = l_parser.convert_points(l_parser.output_data['x1.bottom']['points'])
    evl_points = np.array(l_parser.points_dict_to_list(evl))
    x = np.array(evl_points[:, 0], dtype=np.datetime64)
    y = evl_points[:, 1]
    assert len(x) == 13764
    assert len(y) == 13764
    # Plotting example
    # plt.plot(x, y)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()


def test_parse_time():
    timestamp = 'D20170625T1539223320'
    assert utils.parse_time(timestamp) == np.datetime64('2017-06-25T15:39:22.3320')


def test_convert_evr():
    evr_paths = [data_dir + 'x1.evr',
                 data_dir + 'x3.evr']
    parser = Region2DParser(evr_paths)
    parser.set_range_edge_from_raw(data_dir + 'hake_2017/Summer2017-D20170624-T001210.raw')
    parser.to_json(output_json)
    parser.to_csv(output_csv)
    points = parser.get_points_from_region(4, 'x1', convert_time=True)
    points = parser.get_points_from_region(4, parser.output_path[0])
    assert points[0] == ['D20170625T1539223320', '9.2447583998']

    for path in parser.output_path:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


def test_convert_ecs():
    evr_path = data_dir + 'Summer2017_JuneCal_3freq.ecs'

    parser = CalibrationParser(evr_path)
    parser.parse_files(ignore_comments=True)
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
