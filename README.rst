===============
Summer extremes
===============

This repo contains the code required to perform the data analysis and create the figures for McKinnon, Simpson, and Williams, under review in PNAS.

The data processing to assess ranks in different extreme metrics (e.g. Figure 1) is performed in scripts/rank_trends_summer_extremes.py. This script can be run with daily maximum temperature data from:

* ERA5
* GHCND
* CMIP6 models

The data processing to analze the surface energy budget, and its predictions for temperature, is performed in scripts/surface_energy_balance.py. This script requires daily values of shortwave, longwave, and latent heat fluxes, as well as daily maximum temperature, from ERA5.

The figures (main and supplement) are all made within the notebook found at notebooks/make_figs.ipynb.

ERA5 data is downloaded at the hourly, 0.25 degree resolution, and then converted to daily using the hourly maxima (for temperature) or average (for fluxes), as well as bilinearly interpolated to 1x1 degrees using the scripts in the proc_ERA5 folder in helpful_utilities (https://github.com/karenamckinnon/helpful_utilities/)

GHCND is downloaded and then saved to netcdf files using scripts/update_ghcnd_data.sh

Please contact Karen McKinnon (kmckinnon@ucla.edu) for suggestions, comments, improvements, etc.


* Free software: MIT license



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
