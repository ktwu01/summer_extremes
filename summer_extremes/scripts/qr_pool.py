import time
import multiprocessing
from summer_extremes import utils as summer_utils
import numpy as np
import xarray as xr
import os
import warnings
warnings.filterwarnings("ignore")


procdir = '/home/data/projects/summer_extremes/proc'
qr_dir = '/home/data/projects/summer_extremes/proc/station_qr'
qr_vars = ['TMAX_residual', 'TMIN_residual']
months = np.hstack([608, np.arange(1, 13)])
nboot = 100
qs_to_fit = np.array([0.5, 0.95])


def pull_data_and_run(count):
    ds_ghcnd_seasonal_anoms = xr.open_dataset('%s/ghcnd_data_seasonal_anoms.nc' % procdir)
    to_run = ds_ghcnd_seasonal_anoms.isel(station=count)
    summer_utils.fit_qr_residual_boot(to_run, months, qr_vars, qs_to_fit, nboot, savedir=qr_dir)


if __name__ == '__main__':

    starttime = time.time()
    pool = multiprocessing.Pool()
    ds_ghcnd_seasonal_anoms = xr.open_dataset('%s/ghcnd_data_seasonal_anoms.nc' % procdir)
    # get indices of stations we haven't run
    stations = ds_ghcnd_seasonal_anoms.station.values
    nstations = len(stations)
    to_analyze = []
    for ct in range(len(stations)):
        if not os.path.isfile('%s/%s_qr.nc' % (qr_dir, stations[ct])):
            to_analyze.append(ct)
    to_analyze = np.array(to_analyze)

    pool.map(pull_data_and_run, to_analyze)
    pool.close()
