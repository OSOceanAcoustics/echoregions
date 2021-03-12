import json
import os
from .utils import validate_path


class EvParserBase():
    def __init__(self, input_file, file_format):
        self._input_file = None
        self._filename = None
        self.output_data = {}
        self._output_file = []

        self.format = file_format
        self.input_file = input_file

    @property
    def filename(self):
        if self._filename is None or self._filename not in self.input_file:
            self._filename = os.path.splitext(os.path.basename(self.input_file))[0]
        return self._filename

    @property
    def input_file(self):
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
    def output_file(self):
        if len(self._output_file) == 1:
            self._output_file = list(set(self._output_file))
            return self._output_file[0]
        else:
            return self._output_file

    @staticmethod
    def read_line(open_file, split=False):
        """Remove the LF at the end of every line.
        Specify split = True to split the line on spaces"""
        if split:
            return open_file.readline().strip().split()
        else:
            return open_file.readline().strip()

    def _parse(fid, **kwargs):
        """Base method for parsing files"""

    def parse_file(self, **kwargs):
        """Base method for parsing the file in `input_file` and constructing `output_data`
        Used for EVR and EVL parsers
        """
        if self.input_file is None:
            return
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

    def to_json(self, save_path=None, pretty=False, **kwargs):
        """Convert an Echoview 2D regions .evr file to a .json file

        Parameters
        ----------
        save_path : str
            path to save the JSON file to
        pretty : bool, default False
            Output more human readable JSON
        kwargs
            keyword arguments passed into `parse_file`
        """
        # Parse EVR file if it hasn't already been done
        if not self.output_data:
            self.parse_file(**kwargs)

        # Check if the save directory is safe
        save_path = validate_path(save_path=save_path, input_file=self.input_file, ext='.json')
        indent = 4 if pretty else None

        # Save the entire parsed EVR dictionary as a JSON file
        with open(save_path, 'w') as f:
            f.write(json.dumps(self.output_data, indent=indent))
        self._output_file.append(str(save_path))

    def to_csv(self):
        """Base method for saving to a csv file"""
