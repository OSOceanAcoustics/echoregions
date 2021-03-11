import numpy as np
import pandas as pd
import os
from ..convert.utils import from_JSON
import matplotlib.pyplot as plt


class Regions2DPlotter():
    """Class for plotting Regions. Should only be used by `Regions2D`"""
    def __init__(self, Regions2D):
        self.Regions2D = Regions2D

    def plot_region(self, region, offset=0):
        """Plot a region.

        Parameters
        ----------
        region : str
            id of region to be plotted
        offset : float
            meters to offset the region depth by

        Returns
        -------
        numpy arrays for the x and y points plotted
        """
        points = np.array(self.Regions2D.convert_points(
            self.get_points_from_region(region),
            convert_time=True,
            convert_range_edges=True,
            offset=offset
        ))
        points = self.close_region(points)

        x = np.array(points[:, 0], dtype=np.datetime64)
        y = points[:, 1]
        plt.plot_date(x, y, marker='o', linestyle='dashed', color='r')

        return x, y

    def get_points_from_region(self, region, file=None):
        """Get a list of points from a given region.

        Parameters
        ----------
        region : str
            id of region to be plotted
        file : float
            CSV or JSON file. If `None`, use `output_data`

        Returns
        -------
        list of points from the given region
        """
        if file is not None:
            if file.upper().endswith('.CSV'):
                if not os.path.isfile(file):
                    raise ValueError(f"{file} is not a valid CSV file.")
                data = pd.read_csv(file)
                region = data.loc[data['region_id'] == int(region)]
                # Combine x and y points to get a list of points
                return list(zip(region.x, region.y))
            elif file.upper().endswith('.JSON'):
                data = from_JSON(file)
                points = list(data['regions'][str(region)]['points'].values())
            else:
                raise ValueError(f"{file} is not a CSV or JSON file")

        # Pull region points from passed region dict
        if isinstance(region, dict):
            if 'points' in region:
                points = list(region['points'].values())
            else:
                raise ValueError("Invalid region dictionary")
        # Pull region points from parsed data
        else:
            region = str(region)
            if region in self.Regions2D.output_data['regions']:
                points = list(self.Regions2D.output_data['regions'][region]['points'].values())
            else:
                raise ValueError("{region} is not a valid region")
        return [list(p) for p in points]

    def close_region(self, points):
        """Close a region by appending the first point to end of the list of points.

        Parameters
        ----------
        points : list
            list of points

        Returns
        -------
        list of points where the first point is appended to the end
        """
        is_array = True if isinstance(points, np.ndarray) else False
        points = list(points)
        points.append(points[0])
        if is_array:
            points = np.array(points)
        return points
