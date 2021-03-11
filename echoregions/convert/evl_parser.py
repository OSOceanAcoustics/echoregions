import pandas as pd
import os
import copy
from .utils import parse_time, validate_path
from .ev_parser import EvParserBase


class LineParser(EvParserBase):
    """Class for parsing EV lines (EVL) files
    """
    def __init__(self, input_file=None):
        super().__init__(input_file, 'EVL')

    def _parse(self, fid, convert_time=False, replace_nan_range_value=None, offset=0):
        # Read header containing metadata about the EVL file
        file_type, file_format_number, ev_version = self.read_line(fid, True)
        file_metadata = {
            'file_name': os.path.splitext(os.path.basename(self.input_file))[0],
            'file_type': file_type,
            'file_format_number': file_format_number,
            'echoview_version': ev_version
        }
        points = []
        n_points = int(self.read_line(fid))
        for i in range(n_points):
            date, time, depth, status = self.read_line(fid, split=True)
            points.append({
                'x': f'D{date}T{time}',           # Format: D{CCYYMMDD}T{HHmmSSssss}
                'y': float(depth) + offset,       # Depth [m]
                'status': status                  # 0 = none, 1 = unverified, 2 = bad, 3 = good
            })
        points = self.convert_points(
            points,
            convert_time=convert_time,
            replace_nan_range_value=replace_nan_range_value,
            offset=offset
        )
        return file_metadata, points

    def to_csv(self, save_path=None):
        """Convert an Echoview lines .evl file to a .csv file

        Parameters
        ----------
        save_path : str
            path to save the CSV file to
        """
        if not self.output_data:
            self.parse_file()

        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext='.csv')

        # Save a row for each point
        df = pd.concat(
            [pd.DataFrame([point], columns=['x', 'y', 'status']) for
                pid, point in self.output_data['points'].items()],
            ignore_index=True
        )
        # Save file metadata for each point
        metadata = pd.Series(self.output_data['metadata'])
        for k, v in metadata.items():
            df[k] = v

        # Reorder columns and export to csv
        df.to_csv(save_path, index=False)
        self._output_file.append(save_path)

    def convert_points(self, points, convert_time=True, replace_nan_range_value=None, offset=0):
        """Convert x and y values of points from the EV format.
        Modifies points in-place.

        Parameters
        ----------
        points : list or dict
            List containing EVL points or a single point in dict form
        convert_time : bool, default True
            Convert EV time to datetime64
        replace_nan_range_value : float, default ``None``
            Value in meters to replace -10000.990000 ranges with.
            Don't replace if ``None``.
        offset : float, default 0
            Depth offset in meters.

        Returns
        -------
        list or dict
            Converted points with type depending on input
        """
        def convert_single(point):
            if convert_time:
                point['x'] = parse_time(point['x'])
            if replace_nan_range_value is not None and float(point['y']) == -10000.99:
                point['y'] = replace_nan_range_value + offset
            return point

        singular = True if isinstance(points, dict) and 'x' in points else False
        if singular:
            points = [points]

        converted_points = [convert_single(point) for point in copy.deepcopy(points)]
        if singular:
            converted_points = converted_points[0]
        return converted_points

    @staticmethod
    def points_dict_to_list(points):
        """Convert a dictionary of points to a list

        Parameters
        ----------
        points : dict
            dict of points from parsing an EVL file

        Returns
        -------
        points : list
            list of points in [x, y] format
        """
        return [[p['x'], p['y']] for p in points.values()]
