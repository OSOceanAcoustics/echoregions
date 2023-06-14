import os
import pandas as pd

from ..utils.utils import parse_time, check_file

def parse_line_file(input_file: str):
    # Check for validity of input_file
    check_file(input_file, "EVL")
    fid = open(input_file, encoding="utf-8-sig")
    # Read header containing metadata about the EVL file
    file_type, file_format_number, ev_version = fid.readline().strip().split()
    file_metadata = {
        # TODO: add back the trailing ".evl" in filename for completeness
        "file_name": os.path.splitext(os.path.basename(input_file))[0],
        "file_type": file_type,
        "evl_file_format_version": file_format_number,
        "echoview_version": ev_version,
    }
    # TODO: below is better implemented as reading to EOF
    # and check if the total number of lines read equals to n_points;
    # if the number of lines don't match, return error
    points = []
    n_points = int(fid.readline().strip())
    for i in range(n_points):
        date, time, depth, status = fid.readline().strip().split()
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
    order = list(data_dict["metadata"].keys()) + list(
        data_dict["points"][0].keys()
    )
    data = df[order]

    return data
