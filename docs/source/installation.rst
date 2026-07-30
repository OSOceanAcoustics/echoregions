Installation
============


Echoregions is available and tested for Python 3.12-3.14. The latest branch can be pip installed via the following:

.. code-block:: console

   $ pip install echoregions

To run in development mode, fork and clone the repository at `Echoregions <https://github.com/OSOceanAcoustics/echoregions>`_
and create a conda environment using the conda-forge channel via the following steps:

.. code-block:: console

   $ # Clone your fork
   $ git clone https://github.com/YOUR_GITHUB_USERNAME/echoregions.git

   $ # Go into the cloned repo folder
   $ cd echoregions

   $ # Add the Echostack repository as upstream
   $ git remote add upstream https://github.com/echostack-org/echoregions.git

   $ # Create a conda environment and install pip
   $ conda create -c conda-forge -n echoregions-dev --yes python=3.12

   $ # Switch to the newly built environment
   $ conda activate echoregions-dev

   $ # Upgrade pip to support dependency groups
   $ python -m pip install --upgrade pip

   $ # We recommend installing ipykernel in order to use with JupyterLab and IPython for development
   $ conda install -c conda-forge ipykernel

   $ # Install echopype in editable mode with development dependencies
   $ python -m pip install -e . --group dev
