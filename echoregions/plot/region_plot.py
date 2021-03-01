import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO add in first point to last in order to close the shape
class Region2DPlotter():
    def __init__(self, Regions2D):
        self.Regions2D = Regions2D

    def plot_region(self, region, offset=0):
        points = np.array(self.Regions2D.convert_points(
            self.get_points_from_region(region),
            convert_time=True,
            convert_range_edges=True
        ))

        x = np.array(points[:, 0], dtype=np.datetime64)
        y = [p - offset for p in points[:, 1]]
        plt.plot_date(x, y, marker='o', linestyle='dashed', color='r')

        return x, y

    def get_points_from_region(self, region, file=None):
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
        return [list(l) for l in points]