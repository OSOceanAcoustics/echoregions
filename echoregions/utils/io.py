import json
import os
from pathlib import Path
import re
from typing import Dict


def from_JSON(j: str) -> Dict:
    """Opens a JSON file

    Parameters
    ----------
    j : str
        Valid JSON string or path to JSON file
    """
    if os.path.isfile(j):
        with open(j, "r") as f:
            data_dict = json.load(f)
    else:
        try:
            data_dict = json.loads(j)
        except json.decoder.JSONDecodeError:
            raise ValueError("Invalid JSON string")
    return data_dict


def validate_save_path(
    save_path: str = None, input_file: str = None, ext: str = ".json"
) -> str:
    # Check if save_path is specified.
    # If not try to create one with the input_file and ext

    if save_path is None:
        if input_file is None:
            raise ValueError("No paths given")
        elif ext is None:
            raise ValueError("No extension given")
        else:
            input_file = Path(input_file)
            save_path = input_file.parent / (input_file.stem + ext)
    # If save path is specified, check if folders need to be made
    else:
        save_path = Path(save_path)
        # If save path is a directory, use name of input file
        if save_path.suffix == "":
            if input_file is None:
                raise ValueError("No filename given")
            else:
                input_file = Path(input_file)
                save_path = save_path / (input_file.stem + ext)

    # Check if extension of save path matches desired file format
    if save_path.suffix.lower() != ext.lower():
        raise ValueError(f"{save_path} is not a {ext} file")

    # Create directories if they do not exist
    if not save_path.parent.is_dir():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    return str(save_path)


def check_file_extension_existence(file: str, format: str) -> None:
    """
    Checks if file extension is correct and if file exists.

    Arguments:
        file: str; filename for file to be checked.
        format: str; desired value for file extension of input file.
    """
    if file is not None:
        if not re.search(rf".{format}$", file, flags=re.IGNORECASE):
                raise ValueError(f"Input file {file} is not a {format} file")
        if not os.path.isfile(file):
            raise ValueError(f"{file} does not exist as file.")
    else:
        raise TypeError("Input file must not be None")
