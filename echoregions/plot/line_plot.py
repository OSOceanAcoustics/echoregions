import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union

class LinesPlotter:
    """Class for plotting Regions. Should only be used by `Lines`"""

    def __init__(self, Lines: 'Lines'):
        self.Lines = Lines

    def plot(self, fmt: str="", start_time: datetime=None, end_time: datetime=None,
        fill_between: bool=False, max_depth: Union[int, float]=0, **kwargs) -> None:
        df = self.Lines.data
        if start_time is not None:
            df = df[df["time"] > start_time]
        if end_time is not None:
            df = df[df["time"] < end_time]

        if fill_between:
            plt.fill_between(df.time, df.depth, max_depth, **kwargs)
        else:
            plt.plot(df.time, df.depth, fmt, **kwargs)
