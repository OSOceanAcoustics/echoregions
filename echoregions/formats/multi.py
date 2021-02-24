import pandas as pd
import json
from ..convert.utils import validate_path
from .regions2d import Regions2D


class MultiRegions2D():
    """Container for many Regions2D objects
    """
    def __init__(self, objects):
        """Initialize object

        Parameters
        ----------
        objects : list
            list of parsed Regions2D objects or paths to EVR files.
            The EVR files will be parsed without point conversion.
        """
        if not isinstance(objects, list):
            raise ValueError("Input is not a list")

        self._files = None

        if all(isinstance(f, str) for f in objects):
            self._files = [Regions2D(f) for f in objects]
        elif all(isinstance(f, Regions2D) for f in objects):
            if all([bool(f.output_data) for f in objects]):
                self._files = objects
            else:
                raise ValueError("Not all `Regions2D` objects are parsed")
        else:
            raise ValueError("Input types are not all `str` or `Regions2D` objects.")

        self._output_file = []
        self._output_data = None

    def __getitem__(self, key):
        key = str(key)
        if key not in self.files:
            raise KeyError(f"{key} is not a valid region filename")
        return self.files[key]

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, objects):
        fnames = [r.output_data['metadata']['file_name'] for r in objects]
        self._files = dict(zip(fnames, objects))

    @property
    def output_file(self):
        if len(self._output_file) == 1:
            self._output_file = list(set(self._output_file))
            return self._output_file[0]
        else:
            return self._output_file

    @property
    def output_data(self):
        if self._output_data is None:
            self._output_data = {
                f.output_data['metadata']['file_name']: f.output_data
                for f in self.files
            }
        return self._output_data

    def to_dataframe(self):
        """Concatenates the data of all Region2D files and returns a pandas DataFrame
        """
        df = pd.concat([r.to_dataframe() for r in self.files])
        return df

    def to_csv(self, save_path=None):
        """Save multiple Region2D objects to a single CSV file.

        Parameters
        ----------
        save_path : str
            If save_path is not provided, the file will be saved with the filename of
            the first EVR file at the same location as the EVR file.
        """
        save_path = validate_path(save_path=save_path, input_file=self.files[0].input_file, ext='.csv')
        self.to_dataframe().to_csv(save_path, index=False)
        self._output_file.append(save_path)

    def to_json(self, save_path=None, pretty=False):
        """Save multiple Region2D objects to a single CSV file.

        Parameters
        ----------
        save_path : str
            If save_path is not provided, the file will be saved with the filename of
            the first EVR file at the same location as the EVR file.
        pretty : bool
            Whether or not to format JSON to be more human readable.
            Defaults to `False`
        """
        save_path = validate_path(save_path=save_path, input_file=self.files[0].input_file, ext='.json')
        indent = 4 if pretty else None

        # Save the entire parsed EVR dictionary as a JSON file
        with open(save_path, 'w') as f:
            f.write(json.dumps(self.output_data, indent=indent))
        self._output_file.append(save_path)
