Installation
============


Echoregions is available and tested for Python>=3.10. The latest branch can be installed via the following:

.. code-block:: console

   $ pip install git+https://github.com/OSOceanAcoustics/echoregions.git

To run in development mode, fork and clone the repository at `Echoregions <https://github.com/OSOceanAcoustics/echoregions>`_.

.. code-block:: console
   # Create a conda environment using the supplied requirements files
   mamba create -c conda-forge -n echoregions --yes python=3.12 --file requirements.txt --file requirements-dev.txt

   # Switch to the newly built environment
   mamba activate echoregions

   # We recomment to install ipykernel in order to use with JupyterLab and IPython for development
   conda install -c conda-forge ipykernel

   # Optionally install opencv for generating contours from masks
   conda install -c conda-forge opencv

   # Install echoregions in editable mode (setuptools "develop mode")
   # the command will install all the dependencies
   pip install -e .
