import pandas as pd
import json
from collections import defaultdict
from .ev_parser import EvParserBase
import os


class Region2DParser(EvParserBase):
    def __init__(self, input_files=None):
        super().__init__()
        self.format = 'EVR'
        self.input_files = input_files

    def _parse(self, fid):
        """Reads an open file and returns the file metadata and region information"""
        def _region_metadata_to_dict(line):
            """Assigns a name to each value in the metadata line for each region"""
            return {
                'structure_version': line[0],                               # 13 currently
                'point_count': line[1],                                     # Number of points in the region
                'selected': line[3],                                        # Always 0
                'creation_type': line[4],                                   # Described here: https://support.echoview.com/WebHelp/Reference/File_formats/Export_file_formats/2D_Region_definition_file_format.htm#Data_formats
                'dummy': line[5],                                           # Always -1
                'bounding_rectangle_calculated': line[6],                   # 1 if next 4 fields valid. O otherwise
                # Date encoded as CCYYMMDD and times in HHmmSSssss
                # Where CC=Century, YY=Year, MM=Month, DD=Day, HH=Hour, mm=minute, SS=second, ssss=0.1 milliseconds
                'bounding_rectangle_left_x': f'D{line[7]}T{line[8]}',       # Time and date of bounding box left x
                'bounding_rectangle_top_y': line[9],                       # Top of bounding box
                'bounding_rectangle_right_x': f'D{line[10]}T{line[11]}',    # Time and date of bounding box right x
                'bounding_rectangle_bottom_y': line[12],                    # Bottom of bounding box
            }

        def _points_to_dict(line):
            """Takes a line with point information and creates a tuple (x, y) for each point"""
            points = {}
            for point_num, idx in enumerate(range(0, len(line), 3)):
                x = f'D{line[idx]}T{line[idx + 1]}'
                y = line[idx + 2]
                points[point_num] = (x, y)
            return points

        # Read header containing metadata about the EVR file
        filetype, file_format_number, echoview_version = self.read_line(fid, True)
        file_metadata = {
            'filetype': filetype,
            'file_format_number': file_format_number,
            'echoview_version': echoview_version
        }

        regions = defaultdict(dict)
        n_regions = int(self.read_line(fid))
        # Loop over all regions in file
        for r in range(n_regions):
            fid.readline()    # blank line separates each region
            region_metadata = self.read_line(fid, True)
            rid = region_metadata[2]        # Region ID (unique for each region)

            regions[rid]['metadata'] = _region_metadata_to_dict(region_metadata)
            # Add notes to region data
            n_note_lines = int(self.read_line(fid))
            regions[rid]['notes'] = [self.read_line(fid) for l in range(n_note_lines)]
            # Add detection settings to region data
            n_detection_setting_lines = int(self.read_line(fid))
            regions[rid]['detection_settings'] = [self.read_line(fid) for l in range(n_detection_setting_lines)]
            # Add classification to region data
            regions[rid]['metadata']['region_classification'] = self.read_line(fid)
            # Add point x and y
            points_line = self.read_line(fid, True)
            # For type: 0=bad (No data), 1=analysis, 3=fishtracks, 4=bad (empty water)
            regions[rid]['metadata']['type'] = points_line.pop()
            regions[rid]['points'] = _points_to_dict(points_line)
            regions[rid]['metadata']['name'] = self.read_line(fid)

        return file_metadata, regions

    def to_csv(self, save_dir=None):
        """Convert an Echoview 2D regions .evr file to a .csv file

        Parameters
        ----------
        save_dir : str
            directory to save the CSV file to
        """
        # Parse EVR file if it hasn't already been done
        if not self.output_data:
            self.parse_files()
        # Check if the save directory is safe
        save_dir = self._validate_path(save_dir)

        # Loop over each file. 1 EVR file is saved to 1 CSV file
        for file, data, in self.output_data.items():
            df = pd.DataFrame()
            # Save file metadata for each point
            metadata = pd.Series(data['metadata'])
            # Loop over each region
            for rid, region in data['regions'].items():
                # Save region information for each point
                region_metadata = pd.Series(region['metadata'])
                region_notes = pd.Series({'notes': region['notes']})
                detection_settings = pd.Series({'detection_settings': region['detection_settings']})
                region_id = pd.Series({'region_id': rid})
                # Loop over each point in each region. One row of the dataframe corresponds to one point
                for p, point in enumerate(region['points'].values()):
                    point = pd.Series({
                        'point_idx': p,
                        'x': point[0],
                        'y': point[1],
                    })
                    row = pd.concat([region_id, point, metadata, region_metadata, region_notes, detection_settings])
                    df = df.append(row, ignore_index=True)
            # Reorder columns and export to csv
            output_file_path = os.path.join(save_dir, file) + '.csv'
            df[row.keys()].to_csv(output_file_path, index=False)
            self._output_path.append(output_file_path)

    def get_points_from_region(self, file_path, region):
        """Get points from specified region from a JSON or CSV file
        or from the parsed data.

        Parameters
        ----------
        file_path : str
            path to JSON or CSV file.
            If None, data must be parsed
        region : int
            ID of the region to extract points from

        Returns
        -------
        points : list
            list of x, y points
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"{file_path} is not a valid JSON or CSV file.")
        if file_path.upper().endswith('.JSON'):
            with open(file_path) as f:
                data = json.load(f)
                # Navigate the tree structure and get the points as a list of lists
                points = list(data['regions'][str(region)]['points'].values())
                # Convert to a list of tuples for consistency with CSV
                return [tuple(l) for l in points]
        elif file_path.upper().endswith('.CSV'):
            data = pd.read_csv(file_path)
            region = data.loc[data['region_id'] == int(region)]
            # Combine x and y points to get a list of tuples
            return list(zip(region.x, region.y))
