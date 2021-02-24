import os
import numpy as np
from ..formats.regions2d import Regions2D

data_dir = './echoregions/test_data/ek60/'
output_csv = data_dir + 'output_CSV/'
output_json = data_dir + 'output_JSON/'


def test_convert_evr():
    # Test converting EV regions file (EVR)
    evr_path = data_dir + 'x1.evr'
    r2d = Regions2D(evr_path)
    # parser.set_range_edge_from_raw(data_dir + 'hake_2017/Summer2017-D20170624-T001210.raw')
    r2d.to_json(output_json)
    r2d.to_csv(output_csv)
    points = r2d.get_points_from_region(4)
    points = r2d.get_points_from_region(4, r2d.output_file[0])
    assert points[0] == ['D20170625T1539223320', '9.2447583998']

    for path in r2d.output_file:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)


def test_plotting_points():
    # Test converting points in EV format to plottable values (datetime64 and float)
    evr_paths = data_dir + 'x1.evr'
    r_parser = Regions2D(evr_paths)
    r_parser.to_json(output_json)
    r_parser.set_range_edge_from_raw(data_dir + 'hake_raw/Summer2017-D20170625-T161209.raw')
    points = r_parser.get_points_from_region(r_parser.output_data['regions']['1'])
    evr_points = np.array(r_parser.convert_points(points, convert_time=True, convert_range_edges=True))
    x = np.array(evr_points[:, 0], dtype=np.datetime64)
    y = evr_points[:, 1]
    assert all(y == [r_parser.min_depth, r_parser.max_depth, r_parser.max_depth, r_parser.min_depth])

    os.remove(r_parser.output_file)
    os.rmdir(output_json)


def test_file_select():
    # Test file selection based on region bounds
    raw_files = os.listdir(data_dir + 'hake_raw')

    # Parse region file
    evr_paths = data_dir + 'x1.evr'
    regions = Regions2D(evr_paths)
    regions.parse_file(convert_time=True, convert_range_edges=True)
    raw = regions.select_raw(raw_files, 11)
    assert raw == 'Summer2017-D20170625-T195927.raw'