from summer_extremes import utils as summer_utils
import numpy as np
import xarray as xr
import argparse


procdir = '/glade/work/mckinnon/projects/summer_extremes/proc'
qr_dir = '%s/station_qr' % procdir
qr_vars = ['TMAX_residual', 'TMIN_residual']
months = np.hstack([608, np.arange(1, 13)])
nboot = 100
qs_to_fit = np.array([0.5, 0.95])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proc_number', type=int, help='Job number')
    parser.add_argument('num_per_proc', type=int, help='Number of stations per job')

    args = parser.parse_args()
    proc_number = args.proc_number
    num_per_proc = args.num_per_proc

    ds_ghcnd_seasonal_anoms = xr.open_dataset('%s/ghcnd_data_seasonal_anoms.nc' % procdir)
    stations = ds_ghcnd_seasonal_anoms.station.values
    nstations = len(stations)
    these_idx = np.arange(num_per_proc*proc_number, num_per_proc*(proc_number + 1))
    these_idx = these_idx[these_idx < nstations]

    for kk in these_idx:
        this_da = ds_ghcnd_seasonal_anoms.isel(station=kk)
        summer_utils.fit_qr_residual_boot(this_da, months, qr_vars, qs_to_fit, nboot, savedir=qr_dir,
                                          gmt_fname='/glade/work/mckinnon/BEST/Land_and_Ocean_complete.txt')
