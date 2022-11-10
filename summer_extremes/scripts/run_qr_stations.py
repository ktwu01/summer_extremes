from record_breaking_heat import utils as heat_utils
from summer_extremes import utils as summer_utils

import numpy as np
import pandas as pd
import xarray as xr
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('this_month', type=int, help='Month to run')
    parser.add_argument('procdir', type=str, default='/home/data/projects/summer_extremes/proc',
                        help='Directory to save output')
    parser.add_argument('ghcnd_dir', type=str, default='/home/data/GHCND',
                        help='Directory containing GHCND files that span relevant period')
    parser.add_argument('start_year', type=int, default=1959, help='First year of GHCND files')
    parser.add_argument('end_year', type=int, default=2022, help='Last year of GHCND files')
    parser.add_argument('nboot', type=int, default=100, help='Number of bootstrap samples')
    parser.add_argument('varname', type=str, help='GHCND variable name')

    args = parser.parse_args()
    this_month = args.this_month
    procdir = args.procdir
    ghcnd_dir = args.ghcnd_dir
    start_year = args.start_year
    end_year = args.end_year
    nboot = args.nboot
    variable_to_use = args.varname

    lat_range = -90, 90
    lon_range = -180, 180
    missing_data_cutoff = 0.2  # max fraction of missing data during period
    GHCND_var_names = ['TMAX', 'TMIN']
    nseasonal = 5  # number of seasonal harmonics
    ninteract = 1  # number of seasonal/trend interaction terms

    end_date = '09/2022'

    labelsize = 12
    fontsize = 14

    # Control for FDR
    alpha_fdr = 0.1

    qs_to_fit = np.array([0.05, 0.5, 0.95])
    # Load and process GHCND data if needed
    savenames = ['%s/ghcnd_data_raw.nc' % procdir,
                 '%s/ghcnd_data_seasonal_anoms.nc' % procdir,
                 '%s/ghcnd_data_trend_anoms.nc' % procdir]

    if np.array([os.path.isfile(s) for s in savenames]).all():
        # Load whatever is needed
        ds_ghcnd_seasonal_anoms = xr.open_dataset('%s/ghcnd_data_seasonal_anoms.nc' % procdir)
    else:

        f_station_list = '%s/ghcnd-stations.txt' % ghcnd_dir
        f_inventory = '%s/ghcnd-inventory.txt' % ghcnd_dir

        datadir_ghcnd = '%s/%04i-%04i' % (ghcnd_dir, start_year, end_year)

        inventory_dict = heat_utils.get_ghcnd_inventory_dict(GHCND_var_names, f_inventory)

        station_list, lats, lons = heat_utils.get_ghcnd_station_list(GHCND_var_names, inventory_dict,
                                                                     lat_range, lon_range,
                                                                     start_year, end_year)

        ds_ghcnd = heat_utils.get_ghcnd_ds(station_list, GHCND_var_names, datadir_ghcnd,
                                           start_year, end_year, subset_years=True)

        ds_ghcnd = ds_ghcnd.sel(time=slice(end_date))

        # Remove Feb 29
        ds_ghcnd = ds_ghcnd.sel(time=~((ds_ghcnd['time.month'] == 2) & (ds_ghcnd['time.day'] == 29)))

        # For each month, if missing > missing_data_cutoff of data after 1900, mask all values
        is_missing = np.isnan(ds_ghcnd)
        is_missing = is_missing.groupby('time.month').mean()
        ds_ghcnd = ds_ghcnd.groupby('time.month').where(is_missing < missing_data_cutoff)

        # Also demand that there is less than missing_data_cutoff fraction of missing data in first and last ten years
        is_missing_early = np.isnan(ds_ghcnd.sel(time=slice('%04i' % start_year, '%04i' % (start_year + 9))))
        is_missing_early = is_missing_early.groupby('time.month').mean()

        is_missing_late = np.isnan(ds_ghcnd.sel(time=slice('%04i' % (end_year - 9), '%04i' % end_year)))
        is_missing_late = is_missing_late.groupby('time.month').mean()

        ds_ghcnd = ds_ghcnd.groupby('time.month').where((is_missing_early < missing_data_cutoff) &
                                                        (is_missing_late < missing_data_cutoff))

        no_data = (((~np.isnan(ds_ghcnd['TMAX'])).sum('time') == 0) &
                   ((~np.isnan(ds_ghcnd['TMAX'])).sum('time') == 0))

        ds_ghcnd = ds_ghcnd.where(~no_data)

        # Remove seasonal cycle
        ds_ghcnd_seasonal_anoms = []
        for this_var in GHCND_var_names:
            tmp = summer_utils.fit_seasonal_cycle(ds_ghcnd[this_var], this_var, nseasonal)
            ds_ghcnd_seasonal_anoms.append(tmp)

        ds_ghcnd_seasonal_anoms = xr.merge(ds_ghcnd_seasonal_anoms)

        # Remove seasonal cycle and trend
        ds_ghcnd_seasonal_trend_anoms = []
        for this_var in GHCND_var_names:
            tmp = heat_utils.fit_seasonal_trend(ds_ghcnd[this_var], this_var, nseasonal, ninteract, lastyear=2021)
            ds_ghcnd_seasonal_trend_anoms.append(tmp)

        ds_ghcnd_seasonal_trend_anoms = xr.merge(ds_ghcnd_seasonal_trend_anoms)

        # save and work with this data
        ds_ghcnd.to_netcdf('%s/ghcnd_data_raw.nc' % procdir)
        ds_ghcnd_seasonal_anoms.to_netcdf('%s/ghcnd_data_seasonal_anoms.nc' % procdir)
        ds_ghcnd_seasonal_trend_anoms.to_netcdf('%s/ghcnd_data_trend_anoms.nc' % procdir)

    # ## Quantile regression on station data
    #
    # Perform for each month separately
    t = pd.date_range(start='1950/01/01', periods=365, freq='D')

    qr_savename = '%s/qr_station_data_%s_month_%02i_boot_%04i.nc' % (procdir, variable_to_use,
                                                                     this_month, nboot)
    if os.path.isfile(qr_savename):
        ds_QR = xr.open_dataset(qr_savename)
    else:
        doys = t[t.month == this_month].dayofyear
        # Fit QR trends for each month of data
        ds_QR = heat_utils.fit_qr_trend(ds_ghcnd_seasonal_anoms['%s_residual' % variable_to_use],
                                        doys[0],
                                        doys[-1],
                                        qs_to_fit,
                                        nboot,
                                        lastyear=2021,
                                        gmt_fname='/glade/work/mckinnon/BEST/Land_and_Ocean_complete.txt')

        ds_QR.to_netcdf(qr_savename)
