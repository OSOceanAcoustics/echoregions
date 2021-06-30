import numpy as np
import matplotlib.pyplot as plt


class LinesPlotter():
    """Class for plotting Regions. Should only be used by `Lines`"""
    def __init__(self, Lines):
        self.Lines = Lines

    def plot(
        self,
        start_ping_time=None,
        end_ping_time=None,
        fill_between=True,
        max_depth=0,
        **kwargs
    ):
        df = self.Lines.data

        if start_ping_time is not None:
            df = df[df['ping_time'] > start_ping_time]
        if end_ping_time is not None:
            df = df[df['ping_time'] < end_ping_time]

        if fill_between:
            plt.fill_between(df.ping_time, df.depth, max_depth, **kwargs)
        else:
            plt.plot(df.ping_time, df.depth, **kwargs)
