import pandas as pd
import os
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
        if convert_time or replace_nan_range_value is not None:
            points = self.convert_points(
                points,
                convert_time=convert_time,
                replace_nan_range_value=replace_nan_range_value,
                offset=offset
            )
        return file_metadata, points

    def to_dataframe(self, **kwargs):
        """Create a pandas DataFrame from an Echoview lines file.

        Parameters
        ----------
        kwargs : keyword arguments
            Additional arguments passed to `Lines.parse_file`
        """
        if not self.data:
            self.parse_file(**kwargs)

        # Save a row for each point
        df = pd.DataFrame(self.data['points'])
        # Save file metadata for each point
        df = df.assign(**self.data['metadata'])
        order = list(self.data['metadata'].keys()) + list(self.data['points'][0].keys())
        return df[order].rename({"x": "ping_time", "y": "depth"}, axis=1)

    def to_csv(self, save_path=None, **kwargs):
        """Convert an Echoview lines .evl file to a .csv file

        Parameters
        ----------
        save_path : str
            path to save the CSV file to
        kwargs : keyword arguments
            Additional arguments passed to `Lines.parse_file`
        """
        if not self.data:
            self.parse_file(**kwargs)
        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext='.csv')
        # Reorder columns and export to csv
        self.to_dataframe().to_csv(save_path, index=False)
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
            converted_point = [0, 0]
            converted_point[0] = parse_time(point[x_label]) if convert_time else point[x_label]
            if replace_nan_range_value is not None and float(point[y_label]) == -10000.99:
                converted_point[1] = float(replace_nan_range_value) + offset
            else:
                converted_point[1] = float(point[y_label]) + offset
            return converted_point

        singular = True if isinstance(points, dict) and 'x' in points else False
        if singular:
            points = [points]

        # Change point indexing label if point is a dict or list
        x_label = 'x' if isinstance(points[0], dict) else 0
        y_label = 'y' if isinstance(points[0], dict) else 1
        converted_points = [convert_single(point) for point in points]
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
