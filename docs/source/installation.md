# Installation

Echoregions is available and tested for Python 3.12–3.14. You can install the latest release with:

```console
$ pip install echoregions
```

To run in development mode, fork and clone the repository at
[Echoregions](https://github.com/OSOceanAcoustics/echoregions)
and create a conda environment using the conda-forge channel:

```console
# Clone your fork
git clone https://github.com/YOUR_GITHUB_USERNAME/echoregions.git

# Go into the cloned repo folder
cd echoregions

# Add the Echostack repository as upstream
git remote add upstream https://github.com/echostack-org/echoregions.git

# Create a conda environment and install pip
conda create -c conda-forge -n echoregions-dev --yes python=3.12

# Switch to the newly built environment
conda activate echoregions-dev

# Upgrade pip to support dependency groups
python -m pip install --upgrade pip

# We recommend installing ipykernel to use JupyterLab and IPython for development
conda install -c conda-forge ipykernel

# Install echoregions in editable mode with development and testing dependencies
python -m pip install -e . --group dev --group test
```
