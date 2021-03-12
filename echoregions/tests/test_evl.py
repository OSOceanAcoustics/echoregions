import echoregions as er
from pathlib import Path
import matplotlib.pyplot as plt


data_dir = Path('./echoregions/test_data/ek60/')
evl_path = data_dir / 'x1.bottom.evl'


def test_convert_evl():
    lines = er.read_evl(evl_path)
    lines.plot()
    plt.show()