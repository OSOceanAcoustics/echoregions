from pandas import DataFrame


class Geometry:
    def __init__(self):
        self.depth = (
            None  # Single array that can be used to obtain min_depth and max_depth
        )
        self._min_depth = (
            None  # Set to replace -9999.99 depth values which are EVR min range
        )
        self._max_depth = (
            None  # Set to replace 9999.99 depth values which are EVR max range
        )

        self._nan_depth_value = (
            None  # Set to replace -10000.99 depth values with (EVL only)
        )

        # self.data = None

    def replace_nan_depth(self) -> None:
        """Base method for replacing nan depth values with user-specified replacement value"""
