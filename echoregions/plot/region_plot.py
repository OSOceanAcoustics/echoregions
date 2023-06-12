from typing import Any, NewType
import matplotlib.pyplot as plt
from pandas import DataFrame

# Regions 2D Type Place Holder
Regions2DType = NewType("Regions2D", Any)


class Regions2DPlotter:
    """Class for plotting Regions. Should only be used by `Regions2D`"""

    def __init__(self, Regions2D: Regions2DType):
        self.Regions2D = Regions2D

    def plot(self, region: DataFrame, close_region=False, **kwargs) -> None:
        """Plot one or more regions.

        Parameters
        ----------
        region : DataFrame
            A region id provided as a number, string, list of these,
            or a DataFrame/Series containing the region_id column name.
        close_region : bool
            Plot the region as a closed polygon
        kwargs : keyword arguments
            Additional arguments passed to matplotlib plot
        """
        if close_region:
            region = self.Regions2D.close_region(region)
        for idx, row in region.iterrows():
            plt.plot(row["time"], row["depth"], **kwargs)
