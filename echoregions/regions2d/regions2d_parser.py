import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray

from ..utils.io import check_file
from ..utils.time import parse_time

COLUMNS = [
    "file_name",
    "file_type",
    "evr_file_format_number",
    "echoview_version",
    "region_id",
    "region_structure_version",
    "region_point_count",
    "region_selected",
    "region_creation_type",
    "dummy",
    "region_bbox_calculated",
    "region_bbox_left",
    "region_bbox_right",
    "region_bbox_top",
    "region_bbox_bottom",
    "region_class",
    "region_type",
    "region_name",
    "time",
    "depth",
    "region_notes",
    "region_detection_settings",
]


def parse_evr(input_file: str):
    """Parse EVR Regions2D File and place data in Pandas Dataframe.

    Parameters
    ----------
    input_file : str
        Input EVR file to be parsed.

    Returns
    -------
    DataFrame with parsed data from input EVR file.
    """
    # Check for validity of input_file.
    check_file(input_file, "EVR")

    # Read file.
    fid = open(input_file, encoding="utf-8-sig")

    def _region_metadata_to_dict(line: List) -> Dict:
        """Assigns a name to each value in the metadata line for each region"""
        bound_calculated = int(line[6])
        if bound_calculated:
            left = parse_time(f"{line[7]} {line[8]}", unix=False)
            right = parse_time(f"{line[10]} {line[11]}", unix=False)
            top = float(line[9])
            bottom = float(line[12])
        else:
            left = None
            right = None
            top = None
            bottom = None
        return {
            "region_id": int(line[2]),
            "region_structure_version": line[0],  # 13 currently
            "region_point_count": line[1],  # Number of points in the region
            "region_selected": line[3],  # Always 0
            "region_creation_type": line[4],  # How the region was created
            "dummy": line[5],  # Always -1
            "region_bbox_calculated": bound_calculated,  # 1 if next 4 fields valid.
            # O otherwise
            # Date encoded as CCYYMMDD and times in HHmmSSssss
            # Where CC=Century, YY=Year, MM=Month, DD=Day, HH=Hour,
            # mm=minute, SS=second, ssss=0.1 milliseconds
            "region_bbox_left": left,  # Time and date of bounding box left x; none if not valid.
            "region_bbox_right": right,  # Time and date of bounding box right x; none if not valid.
            "region_bbox_top": top,  # Top of bounding box; none if not valid.
            "region_bbox_bottom": bottom,  # Bottom of bounding box; none if not valid.
        }

    def _parse_points(line: str) -> Tuple[ndarray]:
        """Takes a line with point information and creates a tuple (x, y) for each point"""
        points_x = parse_time(
            [f"{line[idx]} {line[idx + 1]}" for idx in range(0, len(line), 3)]
        ).values
        points_y = np.array([float(line[idx + 2]) for idx in range(0, len(line), 3)])
        return points_x, points_y

    # Read header containing metadata about the EVR file
    file_type, file_format_number, echoview_version = fid.readline().strip().split()
    file_metadata = pd.Series(
        {
            "file_name": os.path.splitext(os.path.basename(input_file))[0]
            + os.path.splitext(os.path.basename(input_file))[1],
            "file_type": file_type,
            "evr_file_format_number": file_format_number,
            "echoview_version": echoview_version,
        }
    )
    rows = []
    n_regions = int(fid.readline().strip())
    # Loop over all regions in file
    for r in range(n_regions):
        # Unpack region data
        fid.readline()  # blank line separates each region
        # Get region metadata
        r_metadata = _region_metadata_to_dict(fid.readline().strip().split())
        # Add notes to region data
        n_note_lines = int(fid.readline().strip())
        r_notes = [fid.readline().strip() for line in range(n_note_lines)]
        # Add detection settings to region data
        n_detection_setting_lines = int(fid.readline().strip())
        r_detection_settings = [fid.readline().strip() for line in range(n_detection_setting_lines)]
        # Add class to region data
        r_metadata["region_class"] = fid.readline().strip()
        # Add point x and y
        points_line = fid.readline().strip().split()
        # For type: 0=bad (No data), 1=analysis, 3=fishtracks, 4=bad (empty water)
        r_metadata["region_type"] = points_line.pop()
        r_points = _parse_points(points_line)
        r_metadata["region_name"] = fid.readline().strip()

        # Store region data into a Pandas series
        row = pd.concat(
            [
                file_metadata,
                pd.Series(r_metadata)[r_metadata.keys()],
                pd.Series({"time": r_points[0]}),
                pd.Series({"depth": r_points[1]}),
                pd.Series({"region_notes": r_notes}),
                pd.Series({"region_detection_settings": r_detection_settings}),
            ]
        )
        row = row.to_frame().T
        rows.append(row)

    if len(rows) == 0:
        data = pd.DataFrame(columns=COLUMNS)
    else:
        df = pd.concat(rows, ignore_index=True)
        data = df[rows[0].keys()].convert_dtypes()
    return data


def parse_regions_df(input_file: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Parses regions dataframe data. This function assumes that the input_file is output
    from regions2d object's to_csv function or the input_file is region_points output
    from regions2d object's mask function.

    Parameters
    ----------
    input_file : str or pd.DataFrame
        Input regions CSV / DataFrame to be parsed.

    Returns
    -------
    data : pd.DataFrame
        The parsed regions data if all checks pass.

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
    expected_columns = ["region_id", "time", "depth"]

    # Check if all expected columns are present
    for column in expected_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")

    # Check for unique region_id values
    if not data["region_id"].is_unique:
        raise ValueError("Non-unique values found in 'region_id' column.")

    # Check if all time data is in the form of an array that is of type string
    if all(isinstance(value, str) and "[" in value and "]" in value for value in data["depth"]):
        # Strip [] and apply np.fromstring to each element in the "depth" column
        data["depth"] = data["depth"].map(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Check if all time data is in the form of an array that is of type string
    if all(isinstance(value, str) and "[" in value and "]" in value for value in data["time"]):
        # Extract datetime strings enclosed in single quotes and split them
        data["time"] = data["time"].apply(
            lambda x: np.array(
                [dt.strip("'") for dt in x.strip("[]").split()], dtype="datetime64[ns]"
            )
        )

    # Set region_id values to integers
    data["region_id"] = data["region_id"].apply(lambda x: int(x))

    return data
