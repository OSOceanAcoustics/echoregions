from ..convert import Region2DParser
from ..plot.region_plot import Region2DPlotter

class Region2D(Region2DParser):
    def __init__(self, input_file=None):
        self.output_data = {}
        self._plotter = None

        super().__init__(input_file)


    @property
    def plotter(self):
        if self._plotter is None:
            if not self.output_data:
                raise ValueError("Input file has not been parsed; call `parse_file` to parse.")
            self._plotter = Region2DPlotter(self)
        return self._plotter

    def plot_region(self, region, offset=0):
        self.plotter.plot_region(region, offset=offset)




