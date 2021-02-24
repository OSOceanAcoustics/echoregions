import numpy as np
import matplotlib.pyplot as plt


class Region2DPlotter():
    def __init__(self, Regions2D):
        self.Regions2D = Regions2D

    def plot_region(self, region, offset=0):
        points = np.array(self.Regions2D.convert_points(
            self.Regions2D.get_points_from_region(region),
            convert_time=True,
            convert_range_edges=True
        ))

        x = np.array(points[:, 0], dtype=np.datetime64)
        y = [p - offset for p in points[:, 1]]
        plt.plot_date(x, y, marker='o', linestyle='dashed', color='r')

        return x, y
