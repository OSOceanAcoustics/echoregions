import matplotlib.pyplot as plt
import pandas as pd

class LinesPlotter:
    """Class for plotting Regions. Should only be used by `Lines`"""

    def __init__(self, Lines):
        self.Lines = Lines

    def plot(
        self,
        fmt="",
        start_time=None,
        end_time=None,
        fill_between=False,
        max_depth=0,
        **kwargs
    ):
        df = self.Lines.data
        if start_time is not None:
            df = df[df["time"] > pd.to_datetime(start_time)]
        if end_time is not None:
            df = df[df["time"] < pd.to_datetime(end_time)]

        if fill_between:
            plt.fill_between(df.time, df.depth, max_depth, **kwargs)
        else:
            plt.plot(df.time, df.depth, fmt, **kwargs)
