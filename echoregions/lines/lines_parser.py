import os
from typing import Union

import numpy as np
import pandas as pd

from ..utils.io import check_file
from ..utils.time import parse_time


def parse_evl(input_file: str):
    """Parse EVL Line File and place data in Pandas Dataframe.

    Parameters
    ----------
    input_file : str
        Input EVL file to be parsed.

    Returns
    -------
    DataFrame with parsed data from input EVL file.
    """
    # Check for validity of input_file
    check_file(input_file, "EVL")
    # Read file and read all lines
    fid = open(input_file, encoding="utf-8-sig")
    file_lines = fid.readlines()
    # Read header containing metadata about the EVL file
    file_type, file_format_number, ev_version = file_lines[0].strip().split()
    file_metadata = {
        "file_name": os.path.splitext(os.path.basename(input_file))[0]
        + os.path.splitext(os.path.basename(input_file))[1],
        "file_type": file_type,
        "evl_file_format_version": file_format_number,
        "echoview_version": ev_version,
    }
    points = []
    n_points = int(file_lines[1].strip())
    # Check if there is a correct matching of points and file lines.
    if (len(file_lines) - 2) != n_points:
        raise ValueError(
            "There exists a mismatch between the expected number of lines in the file"
            "and the actual number of points. There should be 2 less lines in the file than"
            f"the number of points, however we have {len(file_lines)} number of lines in the file"
            f"and {n_points} number of points."
        )
    for i in range(n_points):
        date, time, depth, status = file_lines[i + 2].strip().split()
        points.append(
            {
                "time": f"{date} {time}",  # Format: CCYYMMDD HHmmSSssss
                "depth": float(depth),  # Depth [m]
                "status": status,  # 0 = none, 1 = unverified, 2 = bad, 3 = good
            }
        )
    # Store JSON serializable data
    data_dict = {"metadata": file_metadata, "points": points}

    # Put data into a DataFrame
    df = pd.DataFrame(data_dict["points"])
    # Save file metadata for each point
    df = df.assign(**data_dict["metadata"])
    df.loc[:, "time"] = df.loc[:, "time"].apply(parse_time)
    order = list(data_dict["metadata"].keys()) + list(data_dict["points"][0].keys())
    data = df[order]

    return data


def parse_lines_df(input_file: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Parses lines dataframe data. This function assumes that the input_file is output
    from lines object's to_csv function or the input_file is bottom_points output
    from lines object's mask function.

    Parameters
    ----------
    input_file : str or pd.DataFrame
        Input lines CSV / DataFrame to be parsed.

    Returns
    -------
    data : pd.DataFrame
        The parsed lines data if all checks pass.

    Raises
    ------
    ValueError
        If the parsed data does not match the expected structure.
    """
    if isinstance(input_file, str):
        # Check for validity of input_file.
        check_file(input_file, "CSV")

        # Read data from CSV file
        data = pd.read_csv(input_file)
    elif isinstance(input_file, pd.DataFrame):
        # Set data as input_file
        data = input_file
    else:
        raise ValueError(
            "Input file must be of type str (string path to file) "
            f"nor pd.DataFrame. It is of type {type(input_file)}."
        )

    # Define the expected columns
    expected_columns = ["time", "depth"]

    # Check if all expected columns are present
    for column in expected_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")

    if not pd.api.types.is_float_dtype(data["depth"]):
        # Convert time to np.float64
        data["depth"] = data["depth"].apply(lambda x: np.float64(x))

    if not pd.api.types.is_datetime64_any_dtype(data["time"]):
        # Convert time to np.datetime64
        data["time"] = data["time"].apply(lambda x: np.datetime64(x))

    return data
