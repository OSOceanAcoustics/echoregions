from xarray import DataArray

from .sonar_parser import parse_sonar_file


class Sonar:
    """
    Class for holding Sv data.

    Parameters:
        input_file: str; input file for Sv data.
        data: None or DataArray; Sv data.
    """

    def __init__(self, input_file: None = str):
        self.input_file = input_file
        if input_file is None:
            self._data = None
        else:
            self._data = parse_sonar_file(input_file)

    @property
    def data(self) -> DataArray:
        return self._data

    @data.setter
    def data(self, new_da_Sv: DataArray) -> DataArray:
        """Sets data parameter if input is of DataArray type."""
        if type(new_da_Sv) == DataArray:
            self._data = new_da_Sv
        else:
            raise TypeError(
                f"Input is of type {type(new_da_Sv)}. Must be of type DataArray."
            )
