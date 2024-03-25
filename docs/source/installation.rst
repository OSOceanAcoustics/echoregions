Installation
============


Echoregions is available and tested for Python>=3.9. The latest branch can be installed via the following:

.. code-block:: console

   $ pip install git+https://github.com/OSOceanAcoustics/echoregions.git

To run in development mode, fork and clone the repository at `Echoregions <https://github.com/OSOceanAcoustics/echoregions>`_.

.. code-block:: console

   $ mamba create -c conda-forge -n er-dev --yes python=3.10 --file requirements.txt --file requirements-dev.txt
   $ pip install -e
