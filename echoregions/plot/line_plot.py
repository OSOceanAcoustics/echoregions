import numpy as np
import pandas as pd
import os
from ..convert.utils import from_JSON
import matplotlib.pyplot as plt


class LinesPlotter():
    """Class for plotting Regions. Should only be used by `Lines`"""
    def __init__(self, Lines):
        self.Lines = Lines
