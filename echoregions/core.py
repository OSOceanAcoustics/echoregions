import os
from typing import Union

import s3fs

from .lines.lines import Lines
from .regions2d.regions2d import Regions2D


def read_evr(
    filepath: str, min_depth: float = None, max_depth: float = None
) -> Regions2D:
    """Read an EVR file into a Regions2D object.

    Parameters
    ----------
    filepath : str, Path object
        A valid path to an EVR file
    min_depth : float, default ``None``
        Depth value in meters to set -9999.99 depth edges to.
    max_depth : float, default ``None``
        Depth value in meters to set 9999.99 depth edges to.

    Returns
    -------
    Regions2D
        Object that contains the EVR data and metadata with methods for saving to file.
    """
    return Regions2D(input_file=str(filepath), min_depth=min_depth, max_depth=max_depth)


def read_evl(filepath: str, nan_depth_value: float = None) -> Lines:
    """Read an EVL file into a Lines object.

    Parameters
    ----------
    filepath : str, Path object
        A valid path to an EVL file
    nan_depth_value : float, default ``None``
        Depth in meters to replace -10000.990000 ranges with.

    Returns
    -------
    Lines
        Object that contains EVL data and metadata with methods for saving to file.
    """
    return Lines(input_file=str(filepath), nan_depth_value=nan_depth_value)


def read_cloud_evr(
    s3_path: str,
    s3_key: str,
    s3_secret: str,
    target_directory_path: str,
    min_depth: float = None,
    max_depth: float = None,
) -> Regions2D:
    """Read an EVR file from the cloud into a Regions2D object.

    Parameters
    ----------
    s3_path : str
        A valid path to either a evr file on the cloud.
    s3_key: str
        Valid S3 Bucket Key.
    s3_secret: str
        Valid S3 Bucket Secret.
    target_directory_path: str
        Valid relative directory to temporarily place cloud file. Defaults to the
        /echoregions/tmp directory. Must be a non-existent directory.
    min_depth : float, default ``None``
        Depth value in meters to set -9999.99 depth edges to.
    max_depth : float, default ``None``
        Depth value in meters to set 9999.99 depth edges to.

    Returns
    -------
    Regions2d
        Object that contains the either evr data and metadata
        with methods for saving to file.
    """

    return read_cloud(
        file_type="evr",
        s3_path=s3_path,
        s3_key=s3_key,
        s3_secret=s3_secret,
        target_directory_path=target_directory_path,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def read_cloud_evl(
    s3_path: str,
    s3_key: str,
    s3_secret: str,
    target_directory_path: str,
    nan_depth_value: float = None,
) -> Regions2D:
    """Read an EVR file from the cloud into a Regions2D object.

    Parameters
    ----------
    s3_path : str
        A valid path to either a evr file on the cloud.
    s3_key: str
        Valid S3 Bucket Key.
    s3_secret: str
        Valid S3 Bucket Secret.
    target_directory_path: str
        Valid relative directory to temporarily place cloud file. Defaults to the
        /echoregions/tmp directory. Must be a non-existent directory.
    nan_depth_value : float, default ``None``
        Depth in meters to replace -10000.990000 ranges with.

    Returns
    -------
    Regions2d
        Object that contains the either evr data and metadata
        with methods for saving to file.
    """

    return read_cloud(
        file_type="evl",
        s3_path=s3_path,
        s3_key=s3_key,
        s3_secret=s3_secret,
        target_directory_path=target_directory_path,
        nan_depth_value=nan_depth_value,
    )


def read_cloud(
    file_type: str,
    s3_path: str,
    s3_key: str,
    s3_secret: str,
    target_directory_path: str = os.getcwd() + "/echoregions/tmp/",
    min_depth: float = None,
    max_depth: float = None,
    nan_depth_value: float = None,
) -> Union["Regions2D", "Lines"]:
    """Read an EVR file from the cloud into a Regions2D object.

    Parameters
    ----------
    file_type: str

    s3_path : str
        A valid path to either a evr or evl file on the cloud.
    s3_key: str
        Valid S3 Bucket Key.
    s3_secret: str
        Valid S3 Bucket Secret.
    target_directory_path: str
        Valid relative directory to temporarily place cloud file. Defaults to the
        /echoregions/tmp directory. Must be a non-existent directory.
    min_depth : float, default ``None``
        Depth value in meters to set -9999.99 depth edges to.
    max_depth : float, default ``None``
        Depth value in meters to set 9999.99 depth edges to.
    nan_depth_value : float, default ``None``
        Depth in meters to replace -10000.990000 ranges with.

    Returns
    -------
    Regions2D, Lines
        Object that contains the either evr or evl data and metadata
        with methods for saving to file.
    """
    # Check file type. Must be evr or evl.
    if file_type not in ["evr", "evl"]:
        raise ValueError(f"file_type is {file_type}. Must be evl or evr. ")

    # Ensure correct variables are being passed in.
    if file_type == "evl" and (min_depth != None or max_depth != None):
        raise ValueError(
            f"file_type evl does not use min_depth or max_depth values. \
                         Please clear input for mentioned variables."
        )
    elif file_type == "evr" and nan_depth_value != None:
        raise ValueError(
            f"file_type evr does not use nan_depth_values. \
                         Please clear input for nan_depth_values."
        )

    if isinstance(s3_key, str) and isinstance(s3_secret, str):
        try:
            # Get access to S3 bucket filesystem.
            fs = s3fs.S3FileSystem(
                key=s3_key,
                secret=s3_secret,
            )
        except Exception as e:
            print(e)

        # Create directory if not exists. Else, throw value error.
        if not os.path.exists(target_directory_path):
            os.makedirs(target_directory_path)
        else:
            raise ValueError(
                f"Directory {target_directory_path} already exists. Please \
                             choose a path for a directory that does not current exist."
            )

        # Download File
        try:
            fs.download(s3_path, target_directory_path)

            # Check which file it is in.
            file_name = os.listdir(target_directory_path)[
                0
            ]  # Should be only file in directory.
            target_path = target_directory_path + "/" + file_name

            # Check if filetype is evr or evl and create object based off of filetype.
            if file_type == "evr":
                from echoregions import read_evr

                r2d = read_evr(
                    filepath=target_path, min_depth=min_depth, max_depth=max_depth
                )
                return_object = r2d
            else:
                from echoregions import read_evl

                lines = read_evl(filepath=target_path, nan_depth_value=nan_depth_value)
                return_object = lines

            # Remove target path and target_directory path.
            os.remove(target_path)
            os.removedirs(target_directory_path)

            return return_object
        except Exception as e:
            # Remove target directory created prior to download attempt.
            os.removedirs(target_directory_path)
            print(e)
    else:
        raise TypeError("Both s3_key and s3 secret must be of type str.")
