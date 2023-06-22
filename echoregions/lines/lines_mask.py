import pandas as pd
import xarray as xr
from xarray import DataArray

from .lines import Lines


def lines_mask(
    da_Sv: DataArray, lines: Lines, method: str = "nearest", limit_area: str = None
):
    """
    Subsets a bottom dataset to the range of an Sv dataset
    Create a mask of same shape as data found in Sonar object; bottom: False, otherwise: True

    Arguments:
        sonar - Sonar object containing DataArray of shape (ping_time, depth).
        lines - Lines object containing bottom values.
        method - String containing interpolation method.
        limit_area - String for determining filling restriction for NA values.

    Returns:
      bottom_mask - xarray with dimensions: (ping_time, depth) with bottom: False, otherwise: True
    """

    def filter_bottom(bottom, start_date, end_date):
        """
        Selects the values of the bottom between two dates.
        """
        after_start_date = bottom["time"] > start_date
        before_end_date = bottom["time"] < end_date
        between_two_dates = after_start_date & before_end_date
        filtered_bottom = bottom.loc[between_two_dates].set_index("time")
        return filtered_bottom

    if type(lines) != Lines:
        raise TypeError(
            f"lines should be of type Lines. lines is currently of type {type(lines)}."
        )

    lines_df = lines.data

    # new index
    sonar_index = list(da_Sv.ping_time.data)

    # filter bottom within start and end time
    start_time = da_Sv.ping_time.data.min()
    end_time = da_Sv.ping_time.data.max()
    filtered_bottom = filter_bottom(lines_df, start_time, end_time)

    if len(filtered_bottom) > 0:
        # create joint index
        joint_index = list(
            set(list(pd.DataFrame(sonar_index)[0]) + list(filtered_bottom.index))
        )

        # interpolate on the sonar coordinates
        # nearest interpolation has a problem when points are far from each other
        bottom_interpolated = filtered_bottom.reindex(joint_index).loc[
            sonar_index
        ]  # .interpolate('nearest').loc[sonar_index]
        # max_depth to set the NAs to after interpolation
        max_depth = float(da_Sv.depth.max())
        bottom_interpolated = bottom_interpolated.fillna(max_depth)

        # Interpolate on the sonar coordinates. Note that nearest interpolation has a problem when
        # points are far from each other.
        try:
            bottom_interpolated = (
                filtered_bottom.reindex(joint_index)
                .interpolate(method=method, limit_area=limit_area)
                .loc[sonar_index]
            ).fillna(max_depth)
        except:
            print("")
            raise ValueError(
                "Interpolation arguments are invalid. Visit the docs at \
                 https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html \
                 for more information on what can be placed in said arguments."
            )

        # convert to data array for bottom
        bottom_da = bottom_interpolated[
            "depth"
        ].to_xarray()  # .rename({'index':'ping_time'})
        bottom_da = bottom_da.rename({"time": "ping_time"})

        # create a data array of depths
        depth_da = da_Sv["depth"] + xr.zeros_like(da_Sv)

        # create a mask for the bottom:
        # bottom: False, otherwise: True
        bottom_mask = depth_da < bottom_da

    else:
        # set everything to False
        bottom_mask = xr.full_like(da_Sv, False)

    # bottom: False becomes 0, otherwise: True becomes 1
    bottom_mask = bottom_mask.where(True, 1, 0)

    return bottom_mask
