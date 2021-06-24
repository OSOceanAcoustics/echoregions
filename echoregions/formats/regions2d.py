from ..convert import utils
from ..convert.evr_parser import Region2DParser
from pathlib import Path
import numpy as np


class Regions2D():
    def __init__(self, input_file=None, parse=True,
                 offset=0, min_depth=None, max_depth=None, raw_range=None):
        self._parser = Region2DParser(input_file)
        self._plotter = None
        self._masker = None
        self._region_ids = None

        self.raw_range = raw_range
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.offset = offset
        if parse:
            self.parse_file()

    def __iter__(self):
        return iter(self.output_data['regions'].values())

    def __getitem__(self, key):
        key = str(key)
        if key not in self.output_data['regions']:
            raise KeyError(f"{key} is not a valid region")
        return self.output_data['regions'][key]

    @property
    def output_data(self):
        """Dictionary containing region data and metadata for the EVR file"""
        return self._parser.output_data

    @property
    def output_file(self):
        """Path(s) to the list of files saved.
        String if a single file. LIst of strings if multiple.
        """
        return self._parser.output_file

    @property
    def input_file(self):
        """String path to the EVR file"""
        return self._parser.input_file

    @property
    def raw_range(self):
        """Get the range vector that provides the min_depth and max_depth"""
        return self._parser.raw_range

    @raw_range.setter
    def raw_range(self, val):
        """Set the range vector that provides the min_depth and max_depth"""
        self._parser.raw_range = val
        self.max_depth
        self.min_depth

    @property
    def max_depth(self):
        """Get the depth value that the 9999.99 edge will be set to"""
        if self._parser.max_depth is None and self.raw_range is not None:
            self.max_depth = self.raw_range.max()
        return self._parser.max_depth

    @property
    def min_depth(self):
        """Get the depth value that the -9999.99 edge will be set to"""
        if self._parser.min_depth is None and self.raw_range is not None:
            self.min_depth = self.raw_range.min()
        return self._parser.min_depth

    @property
    def offset(self):
        """Get the depth offset to apply to y values"""
        return self._parser.offset

    @max_depth.setter
    def max_depth(self, val):
        """Set the depth value that the 9999.99 edge will be set to"""
        if self._parser.min_depth is not None:
            if val <= self._parser.min_depth:
                raise ValueError("max_depth cannot be less than min_depth")
        self._parser.max_depth = float(val) if val is not None else val

    @min_depth.setter
    def min_depth(self, val):
        """Set the depth value that the -9999.99 edge will be set to"""
        if self._parser.max_depth is not None:
            if val >= self._parser.max_depth:
                raise ValueError("min_depth cannot be greater than max_depth")
        self._parser.min_depth = float(val) if val is not None else val

    @offset.setter
    def offset(self, val):
        """Set the depth offset to apply to y values"""
        self._parser.offset = float(val)

    def parse_file(self, offset=0):
        """Parse the EVR file as a DataFrame into `Regions2D.output_data`
        """
        self._parser.parse_file()

    def to_csv(self, save_path=None, **kwargs):
        """Convert an EVR file to a CSV file

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        convert_time : bool, default False
          Convert times in the EV datetime format to numpy datetime64.
        kwargs : keyword arguments
            Additional arguments passed to `Regions2D.parse_file`
        """
        self._parser.to_csv(save_path=save_path, **kwargs)

    def to_json(self, save_path=None, **kwargs):
        """Convert EVR to a JSON file

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        pretty : bool, default False
            Output more human readable JSON
        kwargs : keyword arguments
            Additional arguments passed to `Regions2D.parse_file`
        """
        self._parser.to_json(save_path=save_path, **kwargs)

    def close_region(self, points):
        """Closes a region by appending the first point to the end

        Parameters
        ----------
        points : list or np.ndarray
            List of points

        Returns
        -------
        points : list or np.ndarray
            List of points for closed region or numpy array depending on input type
        """
        self._init_plotter()
        return self._plotter.close_region(points)

    def set_range_edge_from_raw(self, raw, model='EK60'):
        """Calculate the sonar range from a raw file using Echopype.
        Used to replace EVR depth edges -9999.99 and 9999.99 with real values

        Parameters
        ----------
        raw : str
            Path to raw file
        model : str
            The sonar model that created the raw file, defaults to `EK60`.
            See echopype for list of supported sonar models.
            Echoregions is only tested with EK60
        """
        self._parser.set_range_edge_from_raw(raw, model=model)

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
        filetimes = np.array([utils.parse_filetime(Path(fname).name) for fname in files])

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
            regions = np.array([self.convert_points(list(self.output_data['regions'][str(r)]['points'].values()))
                                for r in region_id])
            t1 = []
            t2 = []
            for region in regions:
                points = region[:, 0].astype('datetime64[ms]')
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

    def _init_plotter(self):
        """Initialize the object used to plot regions"""
        if self._plotter is None:
            if not self.output_data:
                raise ValueError("Input file has not been parsed; call `parse_file` to parse.")
            from ..plot.region_plot import Regions2DPlotter
            self._plotter = Regions2DPlotter(self)

    def plot_region(self, region, offset=0):
        """Plot a region from output_data.
        Automatically convert time and range_edges.

        Parameters
        ---------
        region : str
            region_id to plot

        offset : float
            A depth offset in meters added to the range of the points used for masking

        Returns
        -------
        x : np.ndarray
            x points used by the matplotlib plot function
        y : np.ndarray
            y points used by the matplotlib plot function
        """
        self._init_plotter()
        self._plotter.plot_region(region, offset=offset)

    def _init_masker(self):
        """Initialize the object used to mask regions"""
        if self._masker is None:
            if not self.output_data:
                raise ValueError("Input file has not been parsed; call `parse_file` to parse.")
            from ..mask.region_mask import Regions2DMasker
            self._masker = Regions2DMasker(self)

    def mask_region(self, ds, region, data_var='Sv', offset=0):
        """Mask an xarray dataset

        Parameters
        ----------
        ds : Xarray Dataset
            calibrated data (Sv or Sp) with range

        region : str
            ID of region to mask the dataset with

        data_var : str
            The data variable in the Dataset to mask

        offset : float
            A depth offset in meters added to the range of the points used for masking

        Returns
        -------
        A dataset with the data_var masked by the specified region
        """
        self._init_masker()
        return self._masker.mask_region(ds, region, offset=offset)
