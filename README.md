# EchoRegions
<a href="https://echoregions.readthedocs.io/en/latest/?badge=latest">
<img src="https://readthedocs.org/projects/echoregions/badge/?version=latest"/>
</a>

![example workflow](https://github.com/OSOceanAcoustics/echoregions/actions/workflows/pytest.yml/badge.svg)


EchoRegions is a tool used for parsing and utilizing region files created in EchoView.

EchoRegions is designed with [Echopype](https://github.com/OSOceanAcoustics/echopype) in mind, and parsed region files will be able to mask or be overlaid on top of Echopype echograms. However, the EVR and EVL parsers do not require Echopype to be installed.

## Functionality and Usage
There are functions for reading EVR and EVL files.

```python
import echoregions as er

evr = er.read_evr('data/x1.evr')  # Read an EVR file
evl = er.read_evl('data/data/x1.bottom.evl')  # Read an EVL file
```

All three of these functions return an object specific to the filetype being parsed, but
they all store the result of the reading into `data`, such as evr.data using the above example.
The data is stored as a Pandas DataFrame, which allows users to leverage the powerful indexing
and computational tools that Pandas provides.

Please see the [API documentation](https://echoregions.readthedocs.io/en/latest/api.html) for all of the classes and functions available in echoregions.
