import json
import numpy as np
import datetime as dt
import os

EV_DATETIME_FORMAT = 'D%Y%m%dT%H%M%S%f'

class EvParserBase():
    def __init__(self, input_file, file_format):
        self._input_file = None
        self._filename = None
        self.output_data = {}
        self._output_path = []

        self.format = file_format
        self.input_file = input_file

    @property
    def filename(self):
        if self._filename is None or self._filename not in self.input_file:
            self._filename = os.path.splitext(os.path.basename(self.input_file))[0]
        return self._filename

    @property
    def input_file(self):
        if self._input_file is None:
            raise ValueError("No input file to parse")
        return self._input_file

    @input_file.setter
    def input_file(self, file):
        if file is not None:
            if not file.upper().endswith(self.format):
                raise ValueError(f"Input file {file} is not a {self.format} file")
            if not os.path.isfile(file):
                raise ValueError(f"Input file {file} does not exist")
            self._input_file = file

    @property
    def output_path(self):
        if len(self._output_path) == 1:
            self._output_path = list(set(self._output_path))
            return self._output_path[0]
        else:
            return self._output_path

    @staticmethod
    def read_line(open_file, split=False):
        """Remove the LF at the end of every line.
        Specify split = True to split the line on spaces"""
        if split:
            return open_file.readline().strip().split()
        else:
            return open_file.readline().strip()

    def _validate_path(self, save_dir=None):
        # Checks a path to see if it is a folder that exists.
        # Create the folder if it doesn't
        if save_dir is None:
            save_dir = os.path.dirname(self.input_file[0])
        else:
            if not os.path.isdir(save_dir):
                if os.path.splitext(save_dir)[1] == '':
                    os.mkdir(save_dir)
                else:
                    raise ValueError(f"{save_dir} is not a valid save directory")
        return save_dir

    @staticmethod
    def from_JSON(j):
        """ Opens a JSON file

        Parameters
        ----------
        j : str
            Valid JSON string or path to JSON file
        """
        if os.path.isfile(j):
            with open(j, 'r') as f:
                data_dict = json.load(f)
        else:
            try:
                data_dict = json.loads(j)
            except json.decoder.JSONDecodeError:
                raise ValueError("Invalid JSON string")
        return data_dict

    def parse_file(self, **kwargs):
        """Base method for parsing the file in `input_file`.
        Used for EVR and EVL parsers

        Parameters
        ----------
        kwargs : dict
            keyword arguments

        Other Paramters
        ---------------
        convert_range_edges : bool
            Whether or not to convert -9999.99 and -9999.99 range edges to real values for EVR files.
            Set the values by assigning range values to `min_range` and `max_range`
            or by passing a file into `set_range_edge_from_raw`. Defaults to True

        replace_nan_range_value : float
            Value to replace -10000.990000 ranges with.
            Don't replace if `None`, defaults to `None`
        """
        fid = open(self.input_file, encoding='utf-8-sig')

        metadata, data = self._parse(fid, **kwargs)
        if self.format == 'EVR':
            data_name = 'regions'
        elif self.format == 'EVL':
            data_name = 'points'
        else:
            raise ValueError("Invalid data format")

        self.output_data = {
            'metadata': metadata,
            data_name: data
        }

    def to_json(self, save_dir=None, **kwargs):
        """Convert an Echoview 2D regions .evr file to a .json file

        Parameters
        ----------
        save_dir : str
            directory to save the JSON file to
        kwargs
            keyword arguments passed into `parse_file`
        """
        # Parse EVR file if it hasn't already been done
        if not self.output_data:
            self.parse_file(**kwargs)

        # Check if the save directory is safe
        save_dir = self._validate_path(save_dir)

        # Save the entire parsed EVR dictionary as a JSON file
        output_file_path = os.path.join(save_dir, self.filename) + '.json'
        with open(output_file_path, 'w') as f:
            f.write(json.dumps(self.output_data))
        self._output_path.append(output_file_path)

    @staticmethod
    def parse_time(ev_time):
        """Convert EV datetime to a numpy datetime64 object

        Parameters
        ----------
        ev_time : str
            EV datetime in CCYYMMDD HHmmSSssss format

        Returns
        -------
        datetime : np.datetime64
            converted input datetime

        Raises
        ------
        ValueError
            when ev_time is not a string
        """
        if not isinstance(ev_time, str):
            raise ValueError("'ev_time' must be type str")
        timestamp = np.array(dt.datetime.strptime(ev_time, EV_DATETIME_FORMAT), dtype=np.datetime64)
        return timestamp

    def to_csv(self):
        """Base method for saving to a csv file"""
