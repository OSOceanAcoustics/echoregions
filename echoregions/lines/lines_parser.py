import os

import pandas as pd

from ..utils.io import check_file
from ..utils.time import parse_time


def parse_line_file(input_file: str):
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
