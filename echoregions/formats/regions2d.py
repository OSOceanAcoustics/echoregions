from numpy.lib.arraysetops import isin
from ..convert import Region2DParser, utils
from ..plot.region_plot import Region2DPlotter
import numpy as np

class Regions2D():
    def __init__(self, input_file=None):
        self.parser = Region2DParser(input_file)
        self._plotter = None

    def __iter__(self):
        return iter(self.output_data['regions'].values())

    def __getitem__(self, key):
        key = str(key)
        if key not in self.output_data['regions']:
            raise KeyError(f"{key} is not a valid region")
        return self.output_data['regions'][key]

    @property
    def output_data(self):
        return self.parser.output_data

    @property
    def output_path(self):
        return self.parser.output_path

    @property
    def raw_range(self):
        return self.parser.raw_range

    @raw_range.setter
    def raw_range(self, val):
        self.parser.raw_range = val

    @property
    def max_depth(self):
        if self.parser.max_depth is None and self.raw_range is not None:
            self.max_depth = self.raw_range.max()
        return self.parser.max_depth

    @property
    def min_depth(self):
        if self.parser.min_depth is None and self.raw_range is not None:
            self.min_depth = self.raw_range.min()
        return self.parser.min_depth

    @max_depth.setter
    def max_depth(self, val):
        if self.min_depth is not None:
            if val <= self.min_depth:
                raise ValueError("max_depth cannot be less than min_depth")
        self.parser.max_depth = float(val)

    @min_depth.setter
    def min_depth(self, val):
        if self.max_depth is not None:
            if val >= self.max_depth:
                raise ValueError("min_depth cannot be greater than max_depth")
        self.parser.min_depth = float(val)

    @property
    def plotter(self):
        if self._plotter is None:
            if not self.output_data:
                raise ValueError("Input file has not been parsed; call `parse_file` to parse.")
            self._plotter = Region2DPlotter(self)
        return self._plotter

    @property
    def regions(self):
        return list(self.output_data['regions'].keys())

    def parse_file(self, convert_time=False, convert_range_edges=True):
        """Parse the EVR file into a `Regions2D.output_data`

        Parameters
        ----------
        convert_time : bool
            Whether or not to convert times in the EV datetime format to numpy datetime64.
            Numpy datetime64 objects cannot be saved to JSON. Default to `False`.
        convert_range_edges : bool
            Whether or not to convert -9999.99 and -9999.99 range edges to real values for EVR files.
            Set the values by assigning range values to `min_range` and `max_range`
            or by passing a file into `set_range_edge_from_raw`. Defaults to `True`
        """
        self.parser.parse_file(convert_time=convert_time, convert_range_edges=convert_range_edges)

    def to_csv(self, save_dir=None, convert_time=False, convert_range_edges=True):
        """Convert EVR to CSV

        Parameters
        ----------
        convert_time : bool
            Whether or not to convert times in the EV datetime format to numpy datetime64.
            Default to `False`.
        convert_range_edges : bool
            Whether or not to convert -9999.99 and -9999.99 range edges to real values for EVR files.
            Set the values by assigning range values to `min_range` and `max_range`
            or by passing a file into `set_range_edge_from_raw`. Defaults to True
        """
        self.parser.to_csv(save_dir=save_dir, convert_time=False, convert_range_edges=convert_range_edges)

    def to_json(self, save_dir=None, convert_range_edges=True, pretty=False):
        """Convert EVR to JSON

        Parameters
        ----------
        convert_range_edges : bool
            Whether or not to convert -9999.99 and -9999.99 range edges to real values for EVR files.
            Set the values by assigning range values to `min_range` and `max_range`
            or by passing a file into `set_range_edge_from_raw`. Defaults to True
        pretty : bool
            Whether or not to output more human readable JSON
        """
        self.parser.to_json(save_dir=save_dir, pretty=pretty, convert_range_edges=convert_range_edges)

    def get_points_from_region(self, region, file=None):
        """Get points from specified region from a JSON or CSV file
        or from the parsed data.

        Parameters
        ----------
        region : int, str, or dict
            ID of the region to extract points from or region dictionary
        file : str
            path to JSON or CSV file. Use parsed data if None

        Returns
        -------
        points : list
            list of x, y points
        """
        return self.parser.get_points_from_region(region, file)

    def convert_points(self, points, convert_time=True, convert_range_edges=True):
        """Convert x and y values of points from the EV format.
        Returns a copy of points.

        Parameters
        ----------
        points : list, dict
            point in [x, y] format or list/dict of these
        convert_time : bool
            Whether to convert EV time to datetime64, defaults `True`
        convert_range_edges : bool
            Whether to convert -9999.99 edges to real range values.
            Min and max ranges must be set manually or by calling `set_range_edge_from_raw`

        Returns
        -------
        points : list or dict
            single converted point or list/dict of converted points depending on input
        """
        return self.parser.convert_points(points, convert_time, convert_range_edges)

    def set_range_edge_from_raw(self, raw, model='EK60'):
        """Calculate the sonar range from a raw file using Echopype.
        Used to replace EVR range edges -9999.99 and 9999.99 with real values

        Parameters
        ----------
        raw : str
            Path to raw file
        model : str
            The sonar model that created the raw file, defaults to `EK60`.
            See echopype for list of supported sonar models.
            Echoregions is only tested with EK60
        """
        self.parser.set_range_edge_from_raw(raw, model=model)

    def convert_output(self, convert_time=True, convert_range_edges=True):
        """Convert x and y values of points from the EV format.
        Modifies Regions2d.output_data
        """
        self.parser.convert_output(convert_time=convert_time, convert_range_edges=convert_range_edges)

    def plot_region(self, region, offset=0):
        """Plot a region from output_data

        Parameters
        ----------
        region : str
            region id
        offset : float
            range offset that region is plotted with
        """
        self.plotter.plot_region(region, offset=offset)

    def select_raw(self, files, region_id=None, t1=None, t2=None):
        """Finds raw files in the time domain that encompasses region or list of regions

        Parameters
        ----------
        files : list
            raw filenames
        region_id : str or list
            region(s) to select raw files with
            If none, select all regions. Defaults to `None`
        t1 : str, numpy datetime64
            lower bound to select files from.
            either EV time string or datetime64 object
        t2 : str, numpy datetime64
            upper bound to select files from
            either EV time string or datetime64 object

        Returns
        -------
        raw : str, list
            raw file as a string if a single raw file is selected.
            list of raw files if multiple are selected.
        """
        files.sort()
        filetimes = np.array([utils.parse_filetime(fname) for fname in files])

        if region_id is not None:
            if not isinstance(region_id, list):
                region_id = [region_id]
        else:
            if t1 is None and t2 is None:
                region_id = list(self.output_data['regions'].keys())
            elif (t1 is not None and t2 is None) or (t1 is None and t2 is not None):
                raise ValueError("Both an upper and lower bound must be provided")
            else:
                t1 = utils.parse_time(t1)
                t2 = utils.parse_time(t2)

        if t1 is None:
            if not all(str(r) in self.output_data['regions'] for r in region_id):
                raise ValueError(f"Invalid region id in {region_id}")
            regions = [self.convert_points(list(self.output_data['regions'][str(r)]['points'].values()))
                       for r in region_id]
            t1 = []
            t2 = []
            for region in regions:
                points = [p[0] for p in region]
                t1.append(min(points))
                t2.append(max(points))
            t1 = min(t1)
            t2 = max(t2)
        lower_idx = np.searchsorted(filetimes, t1) - 1
        upper_idx = np.searchsorted(filetimes, t2)

        if lower_idx == -1:
            lower_idx = 0

        files = files[lower_idx:upper_idx]
        if len(files) == 1:
            return files[0]
        else:
            return files
