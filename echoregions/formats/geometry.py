class Geometry():
    def __init__(self):
        self.depth = None       # Single array that can be used to obtain min_depth and max_depth
        self._min_depth = None   # Set to replace -9999.99 depth values which are EVR min range
        self._max_depth = None   # Set to replace 9999.99 depth values which are EVR max range
        self._offset = 0         # Set to apply depth offset (meters)

        self.nan_depth_value = None     # Set to replace -10000.99 depth values with (EVL only)

        # self.data = None

    def adjust_offset(self, inplace=False):
        """Apply a constant depth value to the 'depth' column in the output DataFrame

        Parameters
        ----------
        inplace : bool
            Modify the current `data` inplace

        Returns
        -------
        DataFrame with depth offsetted by the value in Lines.offset
        """
        if self.offset is None or self.data is None:
            return

        regions = self.data if inplace else self.data.copy()
        regions['depth'] = regions['depth'] + self.offset
        return regions

    def replace_nan_depth(self):
        """Base method for replacing nan depth values with user-specified replacement value
        """
