===============
Summer extremes
===============

This repo contains the code required to perform the data analysis and create the figures for McKinnon, Simpson, and Williams, under review in PNAS.

There are four scripts to run to perform the analysis before making figures:

(1) The data processing to assess ranks in different extreme metrics (e.g. Figure 1) is performed in scripts/rank_trends_summer_extremes.py. This script can be run with daily maximum temperature data from ERA5, GHCND, CHIRTSdaily, and CMIP6 models

(2) The data processing to analze the surface energy budget, and its predictions for temperature, is performed in scripts/surface_energy_balance.py. This script requires daily values of shortwave, longwave, and latent heat fluxes, as well as daily maximum temperature, from ERA5.

(3) The data processing to analyze the relative trends in precipitation is performed in scripts/relative_precip_trends.py. This script can be run with observational dataset and CMIP6 simulations. Observational datasets should be regridded to the 1x1 ERA5 grid, and use the hot/cold/median day definitions from the code in (2). For CMIP6, hot/cold/median days are calculated in the code. Observational dataset used here are: ERA5, CPC, CHIRPS, PERSIANN, MSWEP, GPCC, and GHCND. 

(4) The synthetic data used in Fig. S2 is created in scripts/synthetic_data_and_dof.py

The main figures are all made within the notebook found at notebooks/make_figs.ipynb. The supplemental figures are made in notebooks/make_supp_figs.ipynb.

ERA5 data is downloaded at the hourly, 0.25 degree resolution, and then converted to daily using the hourly maxima (for temperature) or average (for fluxes), as well as bilinearly interpolated to 1x1 degrees using the scripts in the proc_ERA5 folder in helpful_utilities (https://github.com/karenamckinnon/helpful_utilities/)

GHCND is downloaded and then saved to netcdf files using scripts/update_ghcnd_data.sh

Please contact Karen McKinnon (kmckinnon@ucla.edu) for suggestions, comments, improvements, etc.


* Free software: MIT license



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
