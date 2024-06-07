import numpy as np
import xarray as xr


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
