Installation
============


Echoregions is available and tested for Python>=3.10. The latest branch can be pip installed via the following:

.. code-block:: console

   $ pip install echoregions

To run in development mode, fork and clone the repository at `Echoregions <https://github.com/OSOceanAcoustics/echoregions>`_
and create a conda environment using the conda-forge channel:

.. code-block:: console
   $ # Clone your fork
   $ git clone https://github.com/YOUR_GITHUB_USERNAME/echoregions.git

   $ # Go into the cloned repo folder
   $ cd echoregions

   $ # Add the OSOceanAcoustics repository as upstream
   $ git remote add upstream https://github.com/OSOceanAcoustics/echoregions.git

   $ # Create a conda environment using the supplied requirements files
   $ conda create -c conda-forge -n echoregions --yes python=3.12 --file requirements.txt --file requirements-dev.txt

   $ # Switch to the newly built environment
   $ conda activate echoregions

   $ # We recomment to install ipykernel in order to use with JupyterLab and IPython for development
   $ conda install -c conda-forge ipykernel

   $ # Install echoregions in editable mode (setuptools "develop mode")
   $ # the command will install all the dependencies
   $ pip install -e .
