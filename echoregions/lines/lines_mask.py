import xarray as xr
import pandas as pd

from .lines import Lines

def lines_mask(ds_Sv, lines):
    """
    Subsets a bottom dataset to the range of an Sv dataset
    Create a mask of same shape as ds_Sv.Sv, where bottom: False, otherwise: True

    Arguments:
        ds_Sv - Sv dataset containing ds_Sv.Sv xarray of shape (frequency, ping_time, range_bin)
        lines - Lines object containing bottom values.

    Returns:
      bottom_mask - xarray with dimensions: (frequency, ping_time, depth) with bottom: False, otherwise: True
    """

    def filter_bottom(bottom, start_date, end_date):
        """ 
        Selects the values of the bottom between two dates.
        """
        after_start_date = bottom["time"] > start_date
        before_end_date = bottom["time"] < end_date
        between_two_dates = after_start_date & before_end_date
        filtered_bottom = bottom.loc[between_two_dates].set_index('time')
        return(filtered_bottom)
    
    if type(lines) != Lines:
        raise TypeError(f"lines should be of type Lines. lines is currently of type {type(lines)}")

    lines_df = lines.data

    # new index
    sonar_index = list(ds_Sv.Sv.ping_time.data)

    # filter bottom within start and end time
    start_time = ds_Sv.Sv.ping_time.data.min()
    end_time = ds_Sv.Sv.ping_time.data.max()
    filtered_bottom = filter_bottom(lines_df, start_time, end_time)

    if len(filtered_bottom) > 0:

        # create joint index
        joint_index = list(set(list(pd.DataFrame(sonar_index)[0]) + list(filtered_bottom.index)))

        # interpolate on the sonar coordinates
        # nearest interpolation has a problem when points are far from each other
        bottom_interpolated = filtered_bottom.reindex(joint_index).loc[sonar_index]#.interpolate('nearest').loc[sonar_index]
        # max_depth to set the NAs to after interpolation
        max_depth = float(ds_Sv.depth.max())
        bottom_interpolated = bottom_interpolated.fillna(max_depth)

        # convert to data array for bottom
        bottom_da = bottom_interpolated['depth'].to_xarray()#.rename({'index':'ping_time'})
        bottom_da = bottom_da.rename({"time":"ping_time"})

        # create a data array of depths
        depth_da = ds_Sv['depth']+xr.zeros_like(ds_Sv.Sv)
        depth_da = depth_da.drop_vars(['range_sample'])

        # create a mask for the bottom:
        # bottom: False, otherwise: True
        bottom_mask = (depth_da<bottom_da)
  
    else:
        # set everything to False
        bottom_mask = xr.full_like(ds_Sv.Sv, False)

    # bottom: False becomes 0, otherwise: True becomes 1
    bottom_mask = bottom_mask.where(True,1,0)

    return bottom_mask