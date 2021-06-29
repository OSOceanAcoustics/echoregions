import numpy as np
import matplotlib.pyplot as plt


class LinesPlotter():
    """Class for plotting Regions. Should only be used by `Lines`"""
    def __init__(self, Lines):
        self.Lines = Lines

    def plot(
        self,
        calibrated_dataset=None,
        min_ping_time=None,
        max_ping_time=None,
        fill_between=True,
        max_depth=0,
        alpha=0.5,
        **kwargs
    ):
        if calibrated_dataset is not None:
            if min_ping_time is None:
                min_ping_time = calibrated_dataset.ping_time.min().values
            if max_ping_time is None:
                max_ping_time = calibrated_dataset.ping_time.max().values
            if max_depth is None:
                max_depth = calibrated_dataset.range.max().values
        x = self.Lines.data.ping_time
        y = self.Lines.data.depth

        # Apply ping_time upper and lower bounds
        if min_ping_time is not None or max_ping_time is not None:
            if min_ping_time is not None and max_ping_time is not None:
                indices = np.where((x >= min_ping_time) & (x <= max_ping_time))
            elif min_ping_time is not None:
                indices = np.where((x >= min_ping_time))
            elif max_ping_time is not None:
                indices = np.where((x <= max_ping_time))

            x = x[indices]
            y = y[indices]

        if fill_between:
            plt.fill_between(x, y, max_depth, alpha=alpha, **kwargs)
        else:
            plt.plot(x, y, alpha=alpha, **kwargs)
