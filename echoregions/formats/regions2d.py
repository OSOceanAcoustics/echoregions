from pathlib import Path

import numpy as np
from numpy import ndarray
from typing import Union, List, Dict, Iterable
from pandas import DataFrame, Series
from xarray import DataArray

from ..convert import utils
from . import Geometry
from ..plot.region_plot import Regions2DPlotter
from ..convert.evr_parser import Regions2DParser

class Regions2D(Geometry):
    def __init__(
        self,
        input_file: str=None,
        parse: bool=True,
        offset: Union[int, float]=0,
        min_depth: Union[int, float]=None,
        max_depth: Union[int, float]=None,
        depth: ndarray=None,
    ):
        super().__init__()
        self._parser = Regions2DParser(input_file)
        self._plotter = None
        self._masker = None

        self.depth = depth
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.offset = offset

        self.data = None
        if parse:
            self.parse_file()

    def __iter__(self) -> Iterable:
        return self.data.iterrows()

    def __getitem__(self, val: int) -> Series:
        return self.data.iloc[val]

    @property
    def output_file(self) -> Union[str, List[str]]:
        """Path(s) to the list of files saved.
        String if a single file. List of strings if multiple.
        """
        return self._parser.output_file

    @property
    def input_file(self) -> str:
        """String path to the EVR file"""
        return self._parser.input_file

    @property
    def max_depth(self) -> Union[int, float]:
        """Get the depth value that the 9999.99 edge will be set to"""
        if self._max_depth is None and self.depth is not None:
            self._max_depth = self.depth.max()
        return self._max_depth

    @max_depth.setter
    def max_depth(self, val: Union[int, float]) -> Union[int, float]:
        """Set the depth value that the 9999.99 edge will be set to"""
        if self.min_depth is not None:
            if val <= self.min_depth:
                raise ValueError("max_depth cannot be less than min_depth")
        self._max_depth = float(val) if val is not None else val

    @property
    def min_depth(self) -> Union[int, float]:
        """Get the depth value that the -9999.99 edge will be set to"""
        if self._min_depth is None and self.depth is not None:
            self._min_depth = self.depth.min()
        return self._min_depth

    @min_depth.setter
    def min_depth(self, val: Union[int, float]) -> None:
        """Set the depth value that the -9999.99 edge will be set to"""
        if self.max_depth is not None:
            if val >= self.max_depth:
                raise ValueError("min_depth cannot be greater than max_depth")
        self._min_depth = float(val) if val is not None else val

    @property
    def offset(self) -> Union[int, float]:
        """Get the depth offset to apply to y values"""
        return self._offset

    @offset.setter
    def offset(self, val: Union[int, float]) -> None:
        """Set the depth offset to apply to y values"""
        self._offset = float(val)

    @property
    def plotter(self) -> Regions2DPlotter:
        if self._plotter is None:
            if not self.data:
                raise ValueError(
                    "Input file has not been parsed; call `parse_file` to parse."
                )
            self._plotter = Regions2DPlotter(self)
        return self._plotter

    def parse_file(self, offset: Union[int, float]=0) -> None:
        # TODO make use of offset.
        """Parse the EVR file as a DataFrame into `Regions2D.data`"""
        self.data = self._parser.parse_file()
        self.replace_nan_depth()
        self.adjust_offset()

    def to_csv(self, save_path: str=None, **kwargs) -> None:
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
        if self.data is None:
            self.parse_file(**kwargs)
        self._parser.to_csv(self.data, save_path=save_path, **kwargs)

    def to_json(self, save_path: str=None, **kwargs) -> None:
        # TODO: Implement this function
        """Convert EVR to a JSON file.

        Parameters
        ----------
        save_path : str
            Path to save csv file to
        pretty : bool, default False
            Output more human readable JSON
        kwargs : keyword arguments
            Additional arguments passed to `Regions2D.parse_file`
        """
        # self._parser.to_json(save_path=save_path, **kwargs)

    def select_region(self, region: Union[float, str, list, Series, DataFrame]=None,
                      copy=False) -> DataFrame:
        """Ensure that region is a DataFrame.

        Parameters
        ----------
        region : float, str, list, Series, DataFrame, ``None``
            A region id provided as a number, string, list of these,
            or a DataFrame/Series containing the region_id column name.
        copy : bool
            Return a copy of the `data` DataFrame
        Returns
        -------
        DataFrame
            A DataFrame subselected from Regions2D.data.
            There is a row for each region id provided by the region parameter.
        """
        if region is not None:
            if isinstance(region, DataFrame):
                region = list(region.region_id)
            elif isinstance(region, Series):
                region = [region.region_id]
            elif (
                isinstance(region, float)
                or isinstance(region, int)
                or isinstance(region, str)
            ):
                region = [region]
            elif not isinstance(region, list):
                raise TypeError(f"Invalid Region Type: {type(region)}. Must be \
                                of type float, str, list, Series, DataFrame, ``None``")
            # Select row by column id
            region = self.data[self.data["region_id"].isin(region)]
        else:
            region = self.data
        if copy:
            return region.copy()
        else:
            return region

    def close_region(self, region: Union[float, str, List, Series, DataFrame]=None) -> DataFrame:
        """Close a region by appending the first point to end of the list of points.

        Parameters
        ----------
        region : float, str, list, Series, DataFrame, ``None``
            region(s) to select raw files with
            If ``None``, select all regions. Defaults to ``None``

        Returns
        -------
        DataFrame
            Returns a new DataFrame with closed regions
        """
        region = self.select_region(region, copy=True)
        region["time"] = region.apply(
            lambda row: np.append(row["time"], row["time"][0]), axis=1
        )
        region["depth"] = region.apply(
            lambda row: np.append(row["depth"], row["depth"][0]), axis=1
        )
        return region

    def select_sonar_file(self, files: List[str], 
                          region: Union[float, str, list, Series, DataFrame]=None) -> List:
        """Finds sonar files in the time domain that encompasses region or list of regions

        Parameters
        ----------
        files : list
            raw filenames
        region : float, str, list, Series, DataFrame, ``None``
            region(s) to select sonar files with
            If ``None``, select all regions. Defaults to ``None``

        Returns
        -------
        files: list
            list of raw file(s) spanning the encompassing region or list of regions.
        """
        files.sort()
        filetimes = utils.parse_simrad_fname_time(
            [Path(fname).name for fname in files]
        ).values

        # Ensure that region is a DataFrame
        region = self.select_region(region)

        times = np.hstack(region["time"].values)
        lower_idx = np.searchsorted(filetimes, times.min()) - 1
        upper_idx = np.searchsorted(filetimes, times.max())

        lower_idx = 0 if lower_idx < 0 else lower_idx

        files = files[lower_idx:upper_idx]
        return files

    def adjust_offset(self, inplace: bool=False) -> DataFrame:
        """Apply a constant depth value to the 'depth' column in the output DataFrame

        Parameters
        ----------
        inplace : bool
            Modify the current `data` inplace

        Returns
        -------
        DataFrame with depth offsetted by the value in Regions2D.offset
        """
        if self.offset is None or self.data is None:
            return

        regions = self.data if inplace else self.data.copy()
        regions["depth"] = regions["depth"] + self.offset
        return regions

    def replace_nan_depth(self, inplace: bool=False) -> DataFrame:
        """Replace 9999.99 or -9999.99 depth values with user-specified min_depth and max_depth

        Parameters
        ----------
        inplace : bool
            Modify the current `data` inplace

        Returns
        -------
        DataFrame with depth edges replaced by Regions2D.min_depth and  Regions2D.max_depth
        """

        def replace_depth(row: Series) -> Series:
            def swap_val(val: Union[int, float]) -> Union[int, float]:
                if val == 9999.99:
                    return self.max_depth
                elif val == -9999.99:
                    return self.min_depth
                else:
                    return val

            row.at["region_bbox_top"] = swap_val(row["region_bbox_top"])
            row.at["region_bbox_bottom"] = swap_val(row["region_bbox_bottom"])
            for idx, val in enumerate(row["depth"]):
                row["depth"][idx] = swap_val(val)
            return row

        if self.min_depth is None and self.max_depth is None:
            return

        regions = self.data if inplace else self.data.copy()
        regions.loc[:] = regions.apply(replace_depth, axis=1)
        return regions

    def convert_points(
        self, points: Union[List, Dict, DataFrame], convert_time: bool=True, 
        convert_depth_edges: bool=True, offset: Union[int, float]=0, unix: bool=False
    ) -> Union[List, Dict]:
        """Convert x and y values of points from the EV format.
        Returns a copy of points.
        Parameters
        ----------
        points : list, dict
            point in [x, y] format or list/dict of these
        convert_time : bool
            Whether to convert EV time to datetime64, defaults `True`
        convert_depth_edges : bool
            Whether to convert -9999.99 edges to real range values.
            Min and max ranges must be set manually or by calling `set_range_edge_from_raw`
        offset : int, float
            depth offset in meters
        unix : bool
            unix : bool
            Whether or not to output the time in the unix time format
        Returns
        -------
        points : list or dict
            single converted point or list/dict of converted points depending on input
        """
        return self._parser.convert_points(
            points,
            convert_time=convert_time,
            convert_depth_edges=convert_depth_edges,
            offset=offset,
            unix=unix,
        )

    def get_points_from_region(self, region_id: Union[int, str, Dict], 
                               file: str=None) -> List:
        """Get points from specified region from a JSON or CSV file
        or from the parsed data.
        Parameters
        ----------
        region_id : int, str, or dict
            ID of the region to extract points from or region dictionary
        file : str
            path to JSON or CSV file. Use parsed data if None
        Returns
        -------
        points : list
            list of x, y points
        """
        return self.plotter.get_points_from_region(region_id, file)

    def _init_plotter(self) -> None:
        """Initialize the object used to plot regions."""
        if self._plotter is None:
            if self.data is None:
                raise ValueError(
                    "Input file has not been parsed; call `parse_file` to parse."
                )

            self._plotter = Regions2DPlotter(self)

    def plot(self, region: Union[str, List, DataFrame]=None, close_region: bool=False, 
             **kwargs) -> None:
        """Plot a region from data.
        Automatically convert time and range_edges.

        Parameters
        ---------
        region : float, str, list, Series, DataFrame, ``None``
            Region(s) to select raw files with
            If ``None``, select all regions. Defaults to ``None``
        close_region : bool
            Plot the region as a closed polygon. Defaults to False
        kwargs : keyword arguments
            Additional arguments passed to matplotlib plot
        """
        self._init_plotter()

        # Ensure that region is a DataFrame
        region = self.select_region(region)

        self._plotter.plot(region, close_region=close_region, **kwargs)

    def _init_masker(self) -> None:
        """Initialize the object used to mask regions"""
        if self._masker is None:
            if self.data is None:
                raise ValueError(
                    "Input file has not been parsed; call `parse_file` to parse."
                )
            from ..mask.region_mask import Regions2DMasker

            self._masker = Regions2DMasker(self)

    def mask(self, ds: DataArray, region_ids: List, mask_var: str=None, mask_labels=None, 
             offset: Union[int, float]=0) -> DataArray:
        # TODO Does not currently work.
        """Mask an xarray DataArray.

        Parameters
        ----------
        ds : Xarray DataArray
            calibrated data (Sv or Sp) with range
        region_ids : list
            list IDs of regions to create mask for
        mask_var : str
            If provided, used to name the output mask array, otherwise `mask`
        mask_labels:
            None: assigns labels automatically 0,1,2,...

            "from_ids": uses the region ids

            list: uses a list of integers as labels

        offset : int, float
            A depth offset in meters added to the range of the points used for masking

        Returns
        -------
        A DataArray with the data_var masked by the specified region
        """

        if isinstance(mask_labels, list) and (len(mask_labels) != len(region_ids)):
            raise ValueError(
                "If mask_labels is a list, it should be of same length as region_ids."
            )

        if not isinstance(ds, DataArray):
            raise TypeError(f"Invalid ds Type: {type(ds)}. Must be of type\
                            float, str, list, Series, DataFrame, ``None``")

        self._init_masker()

        # dataframe containing region information
        region_df = self.select_region(region_ids)
        return self._masker.mask(
            ds, region_df, mask_var=mask_var, mask_labels=mask_labels, offset=offset
        )
