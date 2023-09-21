import os
from pathlib import Path
from typing import List, Union


def validate_path(save_path: str = None, input_file: str = None, ext: str = ".json") -> str:
    """
    Checks if save_path is specified. If not try to create one with the input_file and ext.

    Parameters
    ----------
    save_path : str
        Target path to save file to.
    input_file : str
        Input path to find possible location for save path if no inputted save path.
    ext: str
        Save path extension.

    Returns
    -------
    str
        File path for which data was saved to.
    """
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


def check_file(file: str, format: Union[List[str], str]) -> None:
    """
    Checks file name format and file existence.

    Parameters
    ----------
    file : str
        File name to check for format and existence.
    format : str, list
        File format.
    """
    if file is not None:
        if isinstance(format, List):
            within = False
            for str_value in format:
                if file.upper().endswith(str_value):
                    within = True
            if not within:
                raise ValueError(f"Input file {file} is not a {format} file")
        else:
            if not file.upper().endswith(format):
                raise ValueError(f"Input file {file} is not a {format} file")
        if not os.path.isfile(file):
            if not os.path.isdir(file):
                raise ValueError(f"{file} does not exist as file or directory.")
    else:
        raise TypeError("Input file must not be of type None")
