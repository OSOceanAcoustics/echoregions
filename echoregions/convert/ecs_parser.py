import numpy as np
import pandas as pd

from .ev_parser import EvParserBase
from .utils import validate_path


# TODO: move this to under echopype
#       don't need to inherit EvParserBase since most parent methods are not used
class CalibrationParser(EvParserBase):
    """Class for parsing EV calibration (ECS) files"""

    def __init__(self, input_file=None, parse=False, ignore_comments=True):
        super().__init__(input_file, "ECS")

        self.data = None

        if parse:
            self.parse_file(ignore_comments=ignore_comments)

    def _parse_settings(self, fid, ignore_comments):
        """Reads lines from an open file.
        The function expects the lines to be in the format <field> = <value>.
        There may be hash marks (#) before the field and after the value.
        Collects these fields and values into a dictionary until a blank line is encountered

        # TODO: add docstring
        """
        # TODO: parse values after the # to give units and range of valid values
        # TODO: use regex to parse lines to make this more robust

        settings = {}
        while True:
            line = fid.readline().strip().split()
            # Exit loop if no more fields in section
            if len(line) == 0:
                break

            # Check if field is commented out
            if line[0] == "#":
                if ignore_comments:
                    continue
                else:
                    idx = 1
            else:
                idx = 0

            field = line[idx]
            val = line[idx + 2]
            # If no value is recorded for the field, save a nan
            val = np.nan if val == "#" else val
            settings[field] = val
        return settings

    def _parse_sourcecal(self, fid, ignore_comments):
        """Parses the 'SOURCECAL SETTTINGS' section.
        Returns a dictionary with keys being the name of the sourcecal
        and values being a key value dictionary parsed by _parse_settings
        """
        sourcecal = {}
        # Parse all 'SourceCal' sections. Return when all have been parsed
        while True:
            cal_name = fid.readline().strip().split()
            if len(cal_name) > 0 and cal_name[0] == "SourceCal":
                sourcecal["_".join(cal_name)] = self._parse_settings(
                    fid, ignore_comments=ignore_comments
                )
            else:
                return sourcecal

    def parse_file(self, ignore_comments=True):
        # TODO: according to Echoview docs, a leading # means that the values are not used.
        #       rather than completely ignoring the key-value pair, let's store them
        #       but have a flag to indicate usage status
        # TODO: create a mechanism to allow the right-overwriting-left convention
        #       perhaps a class storing each of the settings sections
        #       and then a summary object giving the final settings?
        # TODO: add a header reader and record instrument, time created, and version nunber
        #       currently header is just skipped

        def advance_to_section(fid, section):
            # Function for skipping lines that do not contain the variables to save
            cont = True
            # Read lines
            while cont:
                line = fid.readline()
                if section in line:
                    cont = False
            fid.readline()  # Bottom of heading box
            fid.readline()  # Blank line

        fid = open(self.input_file, encoding="utf-8-sig")

        # TODO: consolidate to use the same _parse_settings,
        #        with the new settings class as output
        advance_to_section(fid, "FILESET SETTINGS")
        fileset_settings = self._parse_settings(fid, ignore_comments=ignore_comments)
        advance_to_section(fid, "SOURCECAL SETTINGS")
        sourcecal_settings = self._parse_sourcecal(fid, ignore_comments=ignore_comments)
        advance_to_section(fid, "LOCALCAL SETTINGS")
        localcal_settings = self._parse_settings(fid, ignore_comments=ignore_comments)

        # TODO: re-write below to use new class and associated methods
        #       ._to_DataFrame hides the use of self._data_dict: not a good practice

        self._data_dict = {
            "fileset_settings": fileset_settings,
            "sourcecal_settings": sourcecal_settings,
            "localcal_settings": localcal_settings,
        }
        self.data = self._to_DataFrame()

    def _to_DataFrame(self):
        """Convert the parsed data from a dictionary to a Pandas DataFrame"""

        def get_row_from_source(row_dict, source_dict, **kw):
            source_dict.update(kw)
            for k, v in source_dict.items():
                row_dict[k] = v
            return pd.Series(row_dict)

        df = pd.DataFrame()
        id_keys = ["value_source", "channel"]
        fileset_keys = list(self._data_dict["fileset_settings"].keys())
        sourcecal_keys = list(
            list(self._data_dict["sourcecal_settings"].values())[0].keys()
        )
        localset_keys = list(self._data_dict["localcal_settings"].keys())

        # Combine keys from the different sections and remove duplicates
        row_dict = dict.fromkeys(
            id_keys + fileset_keys + sourcecal_keys + localset_keys, np.nan
        )

        # [WJ commented] sloppy to just use sourcecal_settings and loop through all dicts
        # there could be info for other transducers not present in that section
        for cal, cal_settings in self._data_dict["sourcecal_settings"].items():
            row_fileset = get_row_from_source(
                row_dict=row_dict.copy(),
                source_dict=self._data_dict["fileset_settings"],
                value_source="FILESET",
                channel=cal,
            )
            row_sourcecal = get_row_from_source(
                row_dict=row_dict.copy(),
                source_dict=cal_settings,
                value_source="SOURCECAL",
                channel=cal,
            )
            row_localset = get_row_from_source(
                row_dict=row_dict.copy(),
                source_dict=self._data_dict["localcal_settings"],
                value_source="LOCALSET",
                channel=cal,
            )
            df = df.append(
                [row_fileset, row_sourcecal, row_localset], ignore_index=True
            )

        return df

    def to_csv(self, save_path=None, **kwargs):
        """Convert an Echoview calibration .ecs file to a .csv file

        Parameters
        ----------
        save_path : str
            path to save the CSV file to
        kwargs
            keyword arguments passed into `parse_file`
        """
        # Parse ECS file if it hasn't already been done
        if self.data is None:
            self.parse_file(**kwargs)

        # Check if the save directory is safe
        save_path = validate_path(
            save_path=save_path, input_file=self.input_file, ext=".csv"
        )
        # Export to csv
        self.data.to_csv(save_path, index=False)
        self._output_file.append(save_path)
