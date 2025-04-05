Installation
============


Echoregions is available and tested for Python>=3.10. The latest branch can be pip installed via the following:

.. code-block:: console

   $ pip install echoregions

To run in development mode, fork and clone the repository at `Echoregions <https://github.com/OSOceanAcoustics/echoregions>`_
and create a conda environment using the conda-forge channel via the following steps:

Clone your fork:

.. code-block:: console

   $ git clone https://github.com/YOUR_GITHUB_USERNAME/echoregions.git

Navigate into the cloned repo:

.. code-block:: console

   $ cd echoregions

Add the upstream repository:

.. code-block:: console

   $ git remote add upstream https://github.com/OSOceanAcoustics/echoregions.git

Create a conda environment:

.. code-block:: console

   $ conda create -c conda-forge -n echoregions --yes python=3.12 --file requirements.txt --file requirements-dev.txt

Activate the environment:

.. code-block:: console

   $ conda activate echoregions

Install IPython kernel for JupyterLab:

.. code-block:: console

   $ conda install -c conda-forge ipykernel

Install in editable mode:

.. code-block:: console

   $ pip install -e .
