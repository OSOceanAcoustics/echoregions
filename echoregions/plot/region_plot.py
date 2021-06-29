import matplotlib.pyplot as plt


class Regions2DPlotter():
    """Class for plotting Regions. Should only be used by `Regions2D`"""
    def __init__(self, Regions2D):
        self.Regions2D = Regions2D

    def plot(self, region, **kwargs):
        """Plot one or more regions.

        Parameters
        ----------
        region : str
            DataFrame containing the region(s) to be plotted
        kwargs : keyword arguments
            Additional arguments passed to matplotlib plot
        """
        for idx, row in region.iterrows():
            plt.plot(row.ping_time, row.depth, **kwargs)
