import numpy as np
import matplotlib.pyplot as plt
from ..convert import Region2DParser

class Region2DPlotter():
    def __init__(self, Region2D):
        self.Region2D = Region2D


    def plot_region(self, region, offset=0):
        points = np.array(self.Region2D.convert_points(
            self.Region2D.get_points_from_region(region),
            convert_time=True,
            convert_range_edges=True
        ))

        x = np.array(points[:, 0], dtype=np.datetime64)
        y = [p - offset for p in points[:, 1]]
        plt.plot_date(x, y, marker='o', linestyle='dashed', color='r')

        return x, y