import json
import os

from pandas import DataFrame

from .utils import validate_path


class EvParserBase:
    def __init__(self, input_file: str, file_format: str):
        self._input_file = None
        self._output_file = []

        self.format = file_format
        self.input_file = input_file

    @property
    def input_file(self) -> str:
        return self._input_file

    @input_file.setter
    def input_file(self, file: str) -> None:
        if file is not None:
            if not file.upper().endswith(self.format):
                raise ValueError(f"Input file {file} is not a {self.format} file")
            if not os.path.isfile(file):
                raise ValueError(f"Input file {file} does not exist")
            self._input_file = file
        else:
            raise TypeError("Input file must not be of type None")

    @property
    def output_file(self) -> str:
        if len(self._output_file) == 1:
            self._output_file = list(set(self._output_file))
            return self._output_file[0]
        else:
            return self._output_file

    @staticmethod
    def read_line(open_file, split: bool = False) -> str:
        """Remove the LF at the end of every line.
        Specify split = True to split the line on spaces"""
        if split:
            return open_file.readline().strip().split()
        else:
            return open_file.readline().strip()

    def to_csv(self, data: DataFrame, save_path: bool = None) -> None:
        """Save a Dataframe to a .csv file

        Parameters
        ----------
        data : DataFrame
            DataFrame to save to a CSV
        save_path : str
            path to save the CSV file to
        """
        if not isinstance(data, DataFrame):
            raise TypeError(
                f"Invalid ds Type: {type(data)}. Must be of type DataFrame."
            )

        # Check if the save directory is safe
        save_path = validate_path(
            save_path=save_path, input_file=self.input_file, ext=".csv"
        )
        # Reorder columns and export to csv
        data.to_csv(save_path, index=False)
        self._output_file.append(save_path)

    def to_json(self, save_path: str = None, pretty: bool = True, **kwargs) -> None:
        # TODO Currently only EVL files can be exported to JSON
        """Convert supported formats to .json file.

        Parameters
        ----------
        save_path : str
            path to save the JSON file to
        pretty : bool, default True
            Output more human readable JSON
        kwargs
            keyword arguments passed into `parse_file`
        """
        # Parse file if it hasn't already been done
        if not self._data_dict:
            self.parse_file(**kwargs)

        # Check if the save directory is safe
        save_path = validate_path(
            save_path=save_path, input_file=self.input_file, ext=".json"
        )
        indent = 4 if pretty else None

        # Save the entire parsed EVR dictionary as a JSON file
        with open(save_path, "w") as f:
            f.write(json.dumps(self._data_dict, indent=indent))
        self._output_file.append(str(save_path))
