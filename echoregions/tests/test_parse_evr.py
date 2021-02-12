import os
import numpy as np
from ..formats.region2d import Region2D

data_dir = './echoregions/test_data/ek60/'
output_csv = data_dir + 'output_CSV/'
output_json = data_dir + 'output_JSON/'

def test_convert_evr():
    evr_path = data_dir + 'x1.evr'
    r2d = Region2D(evr_path)
    # parser.set_range_edge_from_raw(data_dir + 'hake_2017/Summer2017-D20170624-T001210.raw')
    r2d.to_json(output_json)
    r2d.to_csv(output_csv)
    points = r2d.get_points_from_region(4)
    points = r2d.get_points_from_region(4, r2d.output_path[0])
    assert points[0] == ['D20170625T1539223320', '9.2447583998']

    for path in r2d.output_path:
        assert os.path.exists(path)
        os.remove(path)

    os.rmdir(output_csv)
    os.rmdir(output_json)

def test_plotting_points():
    evr_paths = data_dir + 'x1.evr'
    r_parser = Region2D(evr_paths)
    r_parser.to_json(output_json)
    r_parser.set_range_edge_from_raw(data_dir + 'Summer2017-D20170625-T161209.raw')
    points = r_parser.get_points_from_region(r_parser.output_data['x1']['regions']['1'])
    evr_points = np.array(r_parser.convert_points(points, convert_time=True, convert_range_edges=True))
    x = np.array(evr_points[:, 0], dtype=np.datetime64)
    y = evr_points[:, 1]
    assert all(y == [r_parser.min_depth, r_parser.max_depth, r_parser.max_depth, r_parser.min_depth])

    os.remove(r_parser.output_path)
    os.rmdir(output_json)