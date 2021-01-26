import json
import os


class EvParserBase():
    def __init__(self):
        self.format = None
        self._input_files = None
        self.output_data = {}
        self._output_path = []

    @property
    def input_files(self):
        if self._input_files is None:
            raise ValueError("No input files to parse")
        return self._input_files

    @input_files.setter
    def input_files(self, files):
        if files is not None:
            if isinstance(files, str):
                files = [files]
            elif not isinstance(files, list):
                raise ValueError(f"Input files must be a string or a list. Got {type(files)} instead")
            for f in files:
                if not f.upper().endswith(self.format):
                    raise ValueError(f'Input file {f} is not a {self.format} file')
                if not os.path.isfile(f):
                    raise ValueError(f"Input file {f} does not exist")
            self._input_files = files

    @property
    def output_path(self):
        if len(self._output_path) == 1:
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
        # Does not create the folder if it doesn't
        # TODO: replace with a general validate path
        if save_dir is None:
            save_dir = os.path.dirname(self.input_files[0])
        else:
            if not os.path.isdir(save_dir):
                if os.path.splitext(save_dir)[1] == '':
                    os.mkdir(save_dir)
                else:
                    raise ValueError(f"{save_dir} is not a valid save directory")
        return save_dir

    def parse_files(self, input_files=None):
        """Base method for parsing the files in `input_files`.
        Used for EVR and EVL parsers"""
        for file in self.input_files:
            fid = open(file, encoding='utf-8-sig')
            fname = os.path.splitext(os.path.basename(file))[0]

            metadata, data = self._parse(fid)
            if self.format == 'EVR':
                data_name = 'regions'
            elif self.format == 'EVL':
                data_name = 'points'

            self.output_data[fname] = {
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
            keyword arguments passed into `parse_files`
        """
        # Parse EVR file if it hasn't already been done
        if not self.output_data:
            self.parse_files(**kwargs)

        # Check if the save directory is safe
        save_dir = self._validate_path(save_dir)

        # Save the entire parsed EVR dictionary as a JSON file
        for file, item in self.output_data.items():
            output_file_path = os.path.join(save_dir, file) + '.json'
            with open(output_file_path, 'w') as f:
                f.write(json.dumps(item))
            self._output_path.append(output_file_path)

    def to_csv(self, save_dir=None):
        """Base method for saving to a csv file"""
