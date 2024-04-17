# SimStock Paper

Notebooks and data for the SimStock paper.

The full list of dependencies can be viewed in the ``pyproject.toml`` file. These can optionally be installed automatically with poetry using the command  ``poetry install``. You must also have a version of EnergyPlus installed. 

## Organisation

This repository contains Jupyter notebooks and python scripts, together with data files, to recreate some simulation results for Croydon. 

- ``imputation.ipynb`` should be run first. This takes the raw data and fills in missing values.
- ``processing_script.py`` then assigns construction settings.
-  Next, ``sims.ipynb`` uses SimStock to simulate the area. **Note:** this notebook requires you to specify your path to EnergyPlus. Refer to the [SimStock documentation](https://simstock.readthedocs.io/en/latest/index.html) for more details.
- Finally, ``post_processing.ipynb`` plots some of the results of the simulation.

The ``datamethods/`` folder contains some useful functions that are used in the above scripts, while the ``data/`` folder contains the raw input data. These scripts may take several hours to run (since, for simplicity, this example is not set up for parallel computation).
