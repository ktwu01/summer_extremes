import numpy as np
import pandas as pd
import xarray as xr
import os
from subprocess import check_call
from scipy import stats
from helpful_utilities.general import lowpass_butter


def process_ghcnd(yr_start, yr_end, ghcnd_dir='/home/data/GHCND', var_names=['TMAX'], country_list=None):
    """This function will subset GHNCD dly files to ones that have sufficient coverage and, if desired, are in a
    specific set of countries.

    To update GHCND data (first), run summer_extremes/scripts/update_ghcnd_data.sh

    Parameters
    ----------
    yr_start : int
        Latest year at which a station should have data
    yr_end : int
        Earliest year in which a station can no longer have data
    ghcnd_dir : str
        Directory containing dly files
    var_names : list
        List of variable names to keep. Standard 5 vars in GHCND: PRCP, SNOW, SNWD, TMAX, TMIN
    country_list : list or None
        List of countries to save data from (FIPS country code), or None if all countries desired

    """

    f_inventory = '%s/ghcnd-inventory.txt' % ghcnd_dir
    outdir = '%s/%04i-%04i' % (ghcnd_dir, yr_start, yr_end)
    cmd = 'mkdir -p %s' % outdir
    check_call(cmd.split())

    # Pull information from inventory
    namestr = [0, 11]
    latstr = [12, 20]
    lonstr = [21, 30]
    varstr = [31, 35]
    startstr = [36, 40]
    endstr = [41, 45]
    for ct_v, this_var in enumerate(var_names):

        with open(f_inventory, 'r') as f:
            name = []
            lon = []
            lat = []
            start = []
            end = []

            for line in f:
                var = line[varstr[0]:varstr[1]]
                if (var == this_var):
                    name.append(line[namestr[0]:namestr[1]])  # station name
                    lat.append(line[latstr[0]:latstr[1]])  # station latitude
                    lon.append(line[lonstr[0]:lonstr[1]])  # station longitude
                    start.append(line[startstr[0]:startstr[1]])  # start year of station data
                    end.append(line[endstr[0]:endstr[1]])  # end year of station data

            this_dict = [{'name': name, 'lat': lat, 'lon': lon, 'start': start, 'end': end}
                         for name, lat, lon, start, end in zip(name, lat, lon, start, end)]

            if ct_v == 0:
                inventory_dict = {this_var: this_dict}
            else:
                inventory_dict[this_var] = this_dict

    for ct_v, this_var in enumerate(var_names):
        station_list = []
        lons = []
        lats = []

        for key in inventory_dict[this_var]:
            this_name = key['name']
            this_start = float(key['start'])
            this_end = float(key['end'])

            in_region = True
            if country_list is not None:
                # if subsetting to countries, set to False, then change to true if match
                in_region = False
                for c in country_list:
                    in_region = this_name[:2] == c
                    if in_region:
                        break

            if (in_region & (this_start <= yr_start) & (this_end >= yr_end)):

                # Add info for each station if not already added to the list
                if this_name not in station_list:

                    station_list.append(this_name)
                    lons.append(float(key['lon']))
                    lats.append(float(key['lat']))

    # Get data for each station
    # ------------------------------
    # Variable   Columns   Type
    # ------------------------------
    # ID            1-11   Character
    # YEAR         12-15   Integer
    # MONTH        16-17   Integer
    # ELEMENT      18-21   Character
    # VALUE1       22-26   Integer
    # MFLAG1       27-27   Character
    # QFLAG1       28-28   Character
    # SFLAG1       29-29   Character
    # VALUE2       30-34   Integer
    # MFLAG2       35-35   Character
    # QFLAG2       36-36   Character
    # SFLAG2       37-37   Character
    #   .           .          .
    #   .           .          .
    #   .           .          .
    # VALUE31    262-266   Integer
    # MFLAG31    267-267   Character
    # QFLAG31    268-268   Character
    # SFLAG31    269-269   Character
    # ------------------------------

    # These variables have the following definitions:

    # ID         is the station identification code.  Please see "ghcnd-stations.txt"
    #            for a complete list of stations and their metadata.
    # YEAR       is the year of the record.

    # MONTH      is the month of the record.

    # ELEMENT    is the element type.   There are five core elements as well as a number
    #            of addition elements.

    #            The five core elements are:

    #            PRCP = Precipitation (tenths of mm)
    #            SNOW = Snowfall (mm)
    #            SNWD = Snow depth (mm)
    #            TMAX = Maximum temperature (tenths of degrees C)
    #            TMIN = Minimum temperature (tenths of degrees C)

    date_str = pd.date_range(start='1850-01-01', end='%04i-12-31' % yr_end, freq='D')

    yearstr = [11, 15]
    monstr = [15, 17]
    varstr = [17, 21]
    datastr = [21, 269]

    for counter, this_station in enumerate(station_list):
        print(this_station)
        print('%i/%i' % (counter, len(station_list)))
        this_file = '%s/ghcnd_all/%s.dly' % (ghcnd_dir, this_station)

        if os.path.isfile(this_file):

            for this_var in var_names:
                savename = '%s/%s_%s.nc' % (outdir, this_station, this_var)
                data_vec = np.nan*np.ones(len(date_str))
                if os.path.isfile(savename):
                    continue
                with open(this_file, 'r') as f:
                    for line in f:
                        if this_var == line[varstr[0]: varstr[1]]:
                            this_year = line[yearstr[0]: yearstr[1]]

                            if float(this_year) >= 1850:  # only keeping data back to 1850
                                mon = line[monstr[0]: monstr[1]]  # the month of data

                                data = line[datastr[0]: datastr[1]]  # the data

                                days = [data[i*8:i*8+8] for i in np.arange(0, 31, 1)]
                                mflag = [days[i][5] for i in np.arange(31)]  # getting the mflag
                                qflag = [days[i][6] for i in np.arange(31)]  # getting the qflag
                                values = [days[i][:5] for i in np.arange(31)]  # getting the data values
                                values_np = np.array(values).astype(int)  # converting to a numpy array

                                # set missing to NaN
                                is_missing = (values_np == -9999)
                                values_np = values_np.astype(float)
                                values_np[is_missing] = np.nan

                                # removing any that fail the quality control flag or have
                                # L = temperature appears to be lagged with respect to reported hour of observation
                                is_bad = (np.array(qflag) != ' ') | (np.array(mflag) == 'L')
                                values_np[is_bad] = np.nan

                                date_idx = (date_str.month == int(mon)) & (date_str.year == int(this_year))
                                data_vec[date_idx] = values_np[:np.sum(date_idx)]/10  # change to degrees Celsius

                # Remove starting and ending NaNs
                start_idx = np.where(~np.isnan(data_vec))[0][0]
                end_idx = np.where(~np.isnan(data_vec))[0][-1] + 1

                new_date_str = date_str[start_idx:end_idx]
                data_vec = data_vec[start_idx:end_idx]

                # Save data
                this_da = xr.DataArray(data_vec, dims='time', coords={'time': new_date_str})
                this_da['lat'] = lats[counter]
                this_da['lon'] = lons[counter]
                this_da.to_netcdf(savename)


def get_pvalue(x, y):
    """
    Calculate a p-value from OLS regression between x and y

    Parameters
    ----------
    x : numpy.ndarray
        A 1D array of the x values for the regression
    y : numpy.ndarray
        A 1D array of the y values for the regression

    Returns
    -------
    pvalue : float
        The p-value estimated for the regression using linregress in scipy
    """

    if np.isnan(y).any():
        return np.nan
    else:
        out = stats.linregress(x, y)
        return out.pvalue


def get_FDR_cutoff(da_pval, alpha_fdr=0.1):
    """
    Calculate the pvalue for significance using a false discovery rate approach.

    Parameters
    ----------
    da_pval : xarray.DataArray
        Contains pvalues for a field (can included nan values)
    alpha_fdr : float
        The false discovery rate control

    Returns
    -------
    cutoff_pval : float
        The highest pvalue for significance
    """

    nx, ny = da_pval.shape
    pval_vec = da_pval.data.flatten()
    has_data = ~np.isnan(pval_vec)
    pval_vec = pval_vec[has_data]

    a = np.arange(len(pval_vec)) + 1
    # find last index where the sorted p-values are equal to or below a line with slope alpha_fdr
    cutoff_idx = np.where(np.sort(pval_vec) <= alpha_fdr*a/len(a))[0][-1]
    cutoff_pval = np.sort(pval_vec)[cutoff_idx]

    return cutoff_pval


def calc_heat_metrics(residual, names, hot_cutoff=95, cold_cutoff=5):
    """
    Calculate various reasonable metrics of hot and cold extremes, after removing the median,
    or some other estimate of the middle of the distribution

    Parameters
    ----------
    residual : xarray.DataArray
        Temperature deviations from the median
    names : list
        List of names of extreme metrics to calculate
    hot_cutoff : int or float
        The percentile cutoff for the defintion of a hot day in cum_excess_hot, avg_excess_hot, and
        ndays_excess_hot
    cold_cutoff : int or float
        The percentile cutoff for the defintion of a cold day in cum_excess_cold, avg_excess_cold, and
        ndays_excess_cold

    Returns
    -------
    ds_metrics : xarray.Dataset
        Contains time series (annual) of heat metrics across domain

    """

    Thot = residual.quantile(hot_cutoff/100, dim='time').drop('quantile')
    Tcold = residual.quantile(cold_cutoff/100, dim='time').drop('quantile')

    ds_metrics = []
    # Maximum value for the year
    if 'seasonal_max' in names:
        metric = residual.groupby('time.year').max().rename('seasonal_max')
        ds_metrics.append(metric)
    # Minimum value for the year
    if 'seasonal_min' in names:
        metric = residual.groupby('time.year').min().rename('seasonal_min')
        ds_metrics.append(metric)

    # Sum of temperature across days that exceed the local hot cutoff (e.g. 95th percentile)
    if 'cum_excess_hot' in names:
        metric = (residual.where(residual > Thot)).groupby('time.year').sum().rename('cum_excess_hot')
        ds_metrics.append(metric)
    # Average temperature on days that exceed the local hot cutoff (e.g. 95th percentile)
    if 'avg_excess_hot' in names:
        metric = (residual.where(residual > Thot)).groupby('time.year').mean().rename('avg_excess_hot')
        metric = metric.fillna(0)
        ds_metrics.append(metric)
    # Number of days that exceed the local hot cutoff (e.g. 95th percentile)
    if 'ndays_excess_hot' in names:
        metric = (residual.where(residual > Thot) > Thot).groupby('time.year').sum().rename('ndays_excess_hot')
        ds_metrics.append(metric)

    #  Sum of degree-days below the cold cutoff
    if 'cum_excess_cold' in names:
        metric = (residual.where(residual < Tcold)).groupby('time.year').sum().rename('cum_excess_cold')
        ds_metrics.append(metric)
    # Average number of degrees below the cold threshold
    if 'avg_excess_cold' in names:
        metric = (residual.where(residual < Tcold)).groupby('time.year').mean().rename('avg_excess_cold')
        metric = metric.fillna(0)
        ds_metrics.append(metric)
    # Number of days below the cold threshold
    if 'ndays_excess_cold' in names:
        metric = (residual.where(residual < Tcold) <
                  Tcold).groupby('time.year').sum().rename('ndays_excess_cold')
        ds_metrics.append(metric)

    # empirical lag-1 AR coefficient (within a season)
    if 'AR1' in names:
        da_rho = []
        for yy in np.unique(residual['time.year']):  # loop through each year
            tmp = residual.sel(time=slice('%04i' % yy, '%04i' % yy))
            tmp_lag1 = tmp.shift({'time': 1})
            rho = xr.corr(tmp, tmp_lag1, dim='time')
            da_rho.append(rho)
        yrs = np.unique(residual['time.year'])
        metric = xr.concat(da_rho, dim='year').rename('AR1')
        metric['year'] = yrs
        ds_metrics.append(metric)

    ds_metrics = xr.merge(ds_metrics)

    # remask
    # need to do mean because a single time step does not have all gridboxes (only warm season)
    is_ocean = np.isnan(residual.mean('time'))
    ds_metrics = ds_metrics.where(~is_ocean)

    return ds_metrics


def rank_and_sort_heat_metrics(ds_metrics):
    """
    Turn each metric time series into a time series of ranks.

    For heat metrics: rank #1 is hottest year (after removing median)
    For cold metrics: rank #1 is coldest year (after removing median)
    For AR(1): rank #1 is highest autocorrelation (not affected by removing median)

    Parameters
    ----------
    ds_metrics : xarray.Dataset
        Contains time series (annual) of heat metrics across domain

    Returns
    -------
    ranks_all : xarray.Dataset
        Contains ranks associated with each heat metric

    """
    # metrics for which the largest values are defined as #1
    # all hot metrics, also number of cold days, and AR(1)
    ranks_hot = (-ds_metrics[['seasonal_max', 'cum_excess_hot',
                              'avg_excess_hot', 'ndays_excess_hot',
                              'ndays_excess_cold', 'AR1']]).rank('year')

    # metrics for which the smallest values (generally negative) are defined as #1
    # values based on minima of temperature
    ranks_cold = (ds_metrics[['seasonal_min', 'cum_excess_cold',
                              'avg_excess_cold']]).rank('year')

    # move ndays_excess_cold into the cold list, and drop from hot
    ranks_cold['ndays_excess_cold'] = ranks_hot['ndays_excess_cold']
    ranks_hot = ranks_hot.drop('ndays_excess_cold')

    # get average across hot day metrics (exclude AR(1))
    mean_hot = ranks_hot[['seasonal_max', 'cum_excess_hot',
                          'avg_excess_hot', 'ndays_excess_hot']].to_array(dim='new').mean('new')
    ranks_hot = ranks_hot.assign(avg_across_metrics_hot=mean_hot)

    # get average across cold day metrics
    mean_cold = ranks_cold.to_array(dim='new').mean('new')
    ranks_cold = ranks_cold.assign(avg_across_metrics_cold=mean_cold)

    # merge again
    ranks_all = xr.merge((ranks_hot, ranks_cold))

    # reorder for plotting
    ranks_all = ranks_all[['seasonal_max', 'cum_excess_hot', 'avg_excess_hot', 'ndays_excess_hot',
                           'avg_across_metrics_hot', 'AR1', 'seasonal_min', 'cum_excess_cold',
                           'avg_excess_cold', 'ndays_excess_cold', 'avg_across_metrics_cold']]

    return ranks_all


def get_slope(x, y):
    """
    Regress two data arrays against each other
    """
    pl = (~np.isnan(x)) & (~np.isnan(y))
    if pl.any():
        out = stats.linregress(x[pl], y[pl])
        return out.slope
    else:
        return np.nan


def get_residual(x, y):
    """
    Get the residual after regressing out x from y
    """
    pl = (~np.isnan(x)) & (~np.isnan(y))
    if pl.any():
        out = stats.linregress(x[pl], y[pl])
        yhat = out.intercept + out.slope*x
        residual = y - yhat
        return residual
    else:
        return np.nan*np.ones((len(x), ))


def get_smooth_clim(data):
    """
    Estimate a smoothed climatology using a lowpass Butterworth filter with a frequency of 1/30d

    The data is mirrored on either side to address edge issues with the filter.
    """
    idx_data = ~np.isnan(data)
    if idx_data.any():
        vals = data[idx_data]
        nt = len(vals)
        tmp = np.hstack((vals[::-1], vals, vals[::-1]))
        filtered = lowpass_butter(1, 1/30, 3, tmp)
        smooth_data = data.copy()
        smooth_data[idx_data] = filtered[nt:2*nt]

        return smooth_data
    else:
        return data
