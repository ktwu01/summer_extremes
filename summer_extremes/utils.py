import numpy as np
import xarray as xr
import os
from scipy import stats
import matplotlib.pyplot as plt
from helpful_utilities.data_proc import get_trend_array
import helpful_utilities.stats as hu_stats
import string
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker


# For plots
labelsize = 12
fontsize = 14
datacrs = ccrs.PlateCarree()
plotcrs = ccrs.PlateCarree(central_longitude=0)
letters = list(string.ascii_lowercase)

# Some other shared variables
regions = 'Global', 'NH', 'tropics', 'SH'
ERA5_tname = 't2m_x'
GHCND_tname = 'TMAX'
CMIP_tname = 'tasmax'
scenarios = 'hist', 'ssp370'
trend_normalizer = 65  # all trends shown as per 65 years
hist_year1 = 1958
hist_year2 = 2023
ssp370_year1 = 2024
ssp370_year2 = 2099
lower_lat = -60
upper_lat = 80
tropics_bound = 10


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


def count_sig(da, is_sig):
    """
    Print out metrics of significance of trends in either direction
    """

    for case in 'positive', 'negative':
        if case == 'positive':
            sig_sign = (da > 0) & is_sig.astype(bool)
        elif case == 'negative':
            sig_sign = (da < 0) & is_sig.astype(bool)

        # mask ocean
        has_data = ~np.isnan(da)
        sig_sign = sig_sign.where(has_data)

        weights = np.cos(np.deg2rad(sig_sign.lat))
        area_frac = sig_sign.weighted(weights=weights).mean(['lat', 'lon'])
        print('%0.1f percent of land has sig %s trend' % (area_frac*100, case))


def get_dof(da_ranks):
    """
    For the time x space array of ranks, estimate the degrees of freedom of the spatial field
    using the eigenvalue formula method of Bretherton et al (1999).

    Because estimating the covariance matrix is slow, save the results if already done

    Parameters
    ----------
    da_ranks : xr.DataArray
        An array of ranks that by definition span 1-ntime. Should be time x lat x lon
    """

    # Get dimensions
    nt, nlat, nlon = da_ranks.shape
    # Get latitude-based weights
    weights = np.cos(np.deg2rad(da_ranks.lat))
    weight_vec = np.repeat(weights.data, nlon, axis=-1).flatten()
    rank_vec = da_ranks.values.reshape((nt, nlat*nlon))

    # subset to relevant land domain
    has_data = ~np.isnan(rank_vec[0, :])
    weight_vec = weight_vec[has_data]
    rank_vec = rank_vec[:, has_data]

    # Calculate covariance using weights
    C = np.cov(rank_vec, aweights=weight_vec)

    # Bretherton estimate of dof
    dof = np.trace(C)**2/(np.linalg.norm(C, ord='fro')**2)
    dof = int(np.round(dof))

    return dof


def create_null_samples(dof, ntime, nsamples, name, procdir):
    """
    Create null samples of ranks that span 1-nt.

    The null hypothesis is based on an uncorrelated sampling of ranks.

    The null samples are saved for future loading to speed figure creation.

    Parameters
    ----------
    dof : int
        Spatial degrees of freedom of the field
    ntime : int
        The number of time steps for the ranks (e.g. number of years in the analysis)
    nsamples : int
        The number of samples to produce
    name : str
        A unique identified for saving the null, e.g. ERA5_tropics_seasonal_max
    procdir : str
        Directory to save samples

    Returns
    -------
    null_samples : xr.DataArray
        An array of the samples following the null hypothesis with metadata

    """

    savename = '%s/%s_%i_null_samples.nc' % (procdir, name, nsamples)
    if os.path.isfile(savename):
        null_samples = xr.open_dataarray(savename)
    else:

        # Will save slope and intercept of regression
        ts_params = np.empty((nsamples, 2))

        # X for OLS regression
        X = (np.arange(ntime)).astype(float)
        X -= np.mean(X)

        for kk in range(nsamples):
            all_ts = []
            for i in range(dof):
                # Each time series is a random choice of ranks
                this_ts = np.random.choice(np.arange(ntime) + 1, ntime, replace=False)
                all_ts.append(this_ts)

            # We are comparing to averages over many time series
            avg_ts = np.mean(np.array(all_ts), axis=0)

            # Get intercept and slope
            linfit = stats.linregress(X, avg_ts)

            # Save parameters for this sample
            ts_params[kk, :] = (linfit.slope, linfit.intercept)

        # Organize into a DataArray
        null_samples = xr.DataArray(ts_params, name='trend_samples', dims=('sample', 'degree'),
                                    coords={'degree': np.array([1, 0]), 'sample': np.arange(nsamples)})

        # Add metadata
        null_samples.attrs['dof'] = dof
        null_samples.attrs['ntime'] = ntime

        null_samples.to_netcdf(savename)

    return null_samples


def make_rank_plots(ds_ranks1, rank1name, metrics_to_plot, long_names, figname, nsamples,
                    alpha_fdr, is_land, procdir, figdir,
                    ds_ranks2=None, rank2name=None, make_map_rank2=False):
    """
    Create trend maps and time series of average ranks.

    All plots will show trends in ds_ranks1.
    Optionally, time series of ds_ranks2 can also be shown
    Optionally, a second row of maps showing the trends in ds_ranks2 can be shown

    The code also prints out relevant metrics for significance

    Parameters
    ----------
    ds_ranks1 : xr.Dataset
        Dataset of ranks to plot (year x lat x lon)
    rank1name : str
        Unique relevant keyword or name for ranks (e.g. 'ERA5', 'CHIRTS', 'ERA5-satellite-era')
    metrics_to_plot : list
        List of names of metrics to plot (should be length 2 for standard plots)
    long_names : list
        Long names for display of metrics
    figname : str
        Name to save figure
    nsamples : int
        Number of null samples for the regional trends
    alpha_fdr : float
        The false discovery rate control level e.g. 0.05
    is_land : xr.DataArray
        Binary indicator of land mask
    procdir : str
        Where processed data are stored
    figdir : str
        Directory for figures to be saved
    ds_ranks2 : xr.Dataset
        Optionally, a second set of rank metrics to plot
    rank2name : str
        Optionally, the unique relevant keyword or name for ranks
    make_map_rank2 : bool
        Whether to make a second row of maps with the second metric

    Returns
    -------
    Nothing, figure is saved
    """

    if not make_map_rank2:  # One row of maps
        fig = plt.figure(figsize=(20, 15))
        heights = [2, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1]
    else:  # Two rows of maps
        fig = plt.figure(figsize=(20, 18))
        heights = [2, 0.2, 2, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1]

    widths = [8, 0.2, 1, 8, 0.2]

    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)
    spec.update(wspace=0.1, hspace=0.1)

    # Plotting parameters for maps
    cmap = plt.cm.RdBu_r
    bounds = np.arange(-15, 16, 3)

    # Plotting parameters for time series
    markers = 's', 'o'
    linecolors = 'k', 'gray'
    slope_colors = 'tab:purple', 'tab:blue'

    # Initialize matrices to collect pvalues
    pvals = []

    letter_ct = 0
    ds_ranks_all = [ds_ranks1, ds_ranks2]
    pvals_ts = []  # time series p-values only assessed for first rank array

    for ct, metric in enumerate(metrics_to_plot):

        # Save maps to compare
        maps_for_corr = []

        # First, plot maps (one or two rows)
        col_ct = 3*ct
        cbar_label = 'Trend in %s\n ranks (/%i yr)' % (long_names[ct], trend_normalizer)
        row_ct = 0
        for d_ct, this_ds in enumerate(ds_ranks_all):
            if (this_ds is None):  # if no second array
                continue

            # Calculate trends
            rank_trends = get_trend_array(this_ds, metric, trend_normalizer=trend_normalizer)
            # Save to compare patterns
            maps_for_corr.append(rank_trends)

            if (d_ct == 1) & (not make_map_rank2):  # if we don't want a map of the second array
                continue

            # Calculate p-values
            pvals = hu_stats.calc_p_for_ds(this_ds[metric])

            # Calculate FDR cutoff for this map
            pcut = hu_stats.get_FDR_cutoff(pvals, alpha_fdr=alpha_fdr)
            is_sig = pvals <= pcut
            is_sig = is_sig.where(is_land)

            # Print out some metrics of significance (locally)
            print(metric)
            count_sig(rank_trends, is_sig)

            # Plot map
            ax_map = fig.add_subplot(spec[row_ct, col_ct], projection=plotcrs)

            ax_map, pc = make_standard_map(rank_trends, ax_map, cmap, bounds, is_sig)

            ax_map.set_title('')
            ax_map.text(0.01, 0.05, '(%s)' % letters[letter_ct],
                        transform=ax_map.transAxes, fontsize=fontsize)

            letter_ct += 1
            row_ct += 2  # update row count for next map

        if len(maps_for_corr) == 2:
            rho_maps = xr.corr(maps_for_corr[0], maps_for_corr[1])
            print('Pearson correlation between trend maps: %0.2f' % rho_maps)

        # Add colorbar (shared across two map rows, if present)
        cax = fig.add_subplot(spec[:(row_ct - 1), col_ct + 1])
        cb = plt.colorbar(pc, cax=cax, orientation='vertical', extend='both')
        cb.ax.tick_params(labelsize=labelsize)
        cb.set_label(cbar_label, fontsize=fontsize)

        for r_ct, region in enumerate(regions):  # Different subplot for each region
            ax = fig.add_subplot(spec[row_ct, col_ct])
            ax.text(0.01, 0.85, '(%s)' % letters[letter_ct], transform=ax.transAxes, fontsize=fontsize)

            # If there are two time series, store each to put the correlation in the title
            ts_for_corr = []
            for d_ct, this_ds in enumerate(ds_ranks_all):
                if this_ds is None:  # if no second array
                    continue

                # Calculate average ranks per year for the domain
                domain_avg = calc_regional_averages(this_ds[metric], [region])
                this_nyrs = len(domain_avg.year)
                this_avg_rank = (this_nyrs + 1)/2
                # Get variable for regression
                X = (np.arange(this_nyrs)).astype(float)
                X -= np.mean(X)

                # Get info for null once, using the primary rank dataset (typically ERA5)
                if d_ct == 0:
                    # Estimate spatial degrees of freedom and get null samples
                    # Null hypothesis: ranks are independent year-to-year
                    dof = get_dof(this_ds[metric].sel(lat=get_lats_for_region(region)))
                    null_name = '%s_%s_%s' % (eval('rank%iname' % (d_ct + 1)), region, metric)
                    null_samples = create_null_samples(dof, this_nyrs, nsamples, null_name, procdir)

                    # Plot null range
                    null_yhat = (null_samples.sel(degree=0).data[np.newaxis, :] +
                                 X[:, np.newaxis]*null_samples.sel(degree=1).data[np.newaxis, :])
                    ax.fill_between(domain_avg.year, np.percentile(null_yhat - this_avg_rank, 2.5, axis=1),
                                    np.percentile(null_yhat - this_avg_rank, 97.5, axis=1), color='gray',
                                    alpha=0.5)
                    zorder = 4
                else:
                    zorder = 2

                # Plot time series of average ranks
                (domain_avg - this_avg_rank).plot(ax=ax, color=linecolors[d_ct],
                                                  marker=markers[d_ct], zorder=(zorder - 1))

                # Calclulate and plot slope of average ranks
                linfit = stats.linregress(X, domain_avg)
                this_yhat = linfit.intercept + linfit.slope*X
                ax.plot(domain_avg.year, (this_yhat - this_avg_rank),
                        color=slope_colors[d_ct], lw=3, zorder=zorder)

                # Estimate two-sided p-value based on where the observed slope falls within the null slopes
                if d_ct == 0:
                    this_pval = hu_stats.p_value_from_synthetic_null_data(linfit.slope, null_samples.sel(degree=1))
                    pvals_ts.append(this_pval)

                # Save for correlation if two time series are plotted
                ts_for_corr.append(domain_avg)

            # Set titles, fontsizes, etc.
            if len(ts_for_corr) == 2:
                rho_ts = xr.corr(ts_for_corr[0], ts_for_corr[1])
                ax.set_title('%s (%s) (r$_{%s}$=%0.2f)' % (long_names[ct], region, rank2name, rho_ts),
                             fontsize=fontsize)
            else:
                ax.set_title('%s (%s)' % (long_names[ct], region), fontsize=fontsize)

            ax.tick_params(labelsize=labelsize)
            if region != 'SH':
                ax.set_xticks([])
            ax.set_ylabel('Rank anomaly', fontsize=fontsize)
            ax.set_xlabel('')
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)

            # Move counts forward for next panel
            row_ct += 2
            letter_ct += 1

    # Perform FDR control across time series
    pvals_ts = np.array(pvals_ts)
    pcut_ts = hu_stats.get_FDR_cutoff(xr.DataArray(pvals_ts), alpha_fdr=alpha_fdr)
    is_sig_ts = pvals_ts <= pcut_ts
    counter = 0
    for ct, metric in enumerate(metrics_to_plot):
        for r_ct, region in enumerate(regions):
            print('pval = %0.3f, sig = %s (%s %s)' % (pvals_ts[counter], is_sig_ts[counter].astype(bool),
                                                      region, metric))
            counter += 1

    # Save figure
    plt.savefig('%s/%s' % (figdir, figname), dpi=200, bbox_inches='tight')


def make_standard_map(da, ax, cmap, bounds, is_sig=None):
    """
    Plot the lat/lon dataframe on the axis ax

    Parameters
    ----------
    da : xr.DataArray
        Values to plot in lat/lon space
    ax : GeoAxesSubplot
        Axis to plot in
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap for map
    bounds : numpy.array
        Colorbar intervals
    is_sig : None or xr.DataArray
        0/1 indicator of whether a gridbox is significant (= 1)

    Returns
    -------
    ax : GeoAxesSubplot
        Updated axes
    pc :
    """

    ax.fill_betweenx([-tropics_bound, tropics_bound], -180, 180, color='gray', alpha=0.5, transform=ccrs.PlateCarree())

    pc = da.plot.pcolormesh(ax=ax,
                            cmap=cmap,
                            levels=bounds,
                            transform=datacrs,
                            zorder=1,
                            add_colorbar=False,
                            extend='both')

    if is_sig is not None:
        cs = is_sig.fillna(0).plot.contourf(ax=ax,
                                            levels=[-1, 0, 2],
                                            colors='none', hatches=[None, '...'],
                                            transform=datacrs,
                                            zorder=1,
                                            add_colorbar=False)

        cs.collections[-1].set_edgecolor('w')
        cs.collections[-1].set_linewidth(1)

    ax.add_feature(cfeature.COASTLINE, color='black', zorder=2)
    ax.add_feature(cfeature.LAKES, color='gray', zorder=2)
    ax.set_title('')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2,
                      color='gray', alpha=0.2, linestyle='-', draw_labels=True, zorder=3)
    gl.top_labels = False
    gl.left_labels = True
    gl.right_labels = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 80, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'fontsize': fontsize - 2}
    gl.ylabel_style = {'color': 'black', 'fontsize': fontsize - 2}

    return ax, pc


def calc_regional_averages(da, regions):
    """
    Calculate average values over prespecified regions

    Parameters
    ----------
    da : xr.DataArray
        Values to plot in lat/lon space
    domains : list
        Names of domains (bound are predefined, except the width of the tropics)

    Returns
    -------
    avg_vals : xr.Dataset or xr.DataArray
        Contains the weighted average across latitude/longitude for the specific domains
    """

    avg_vals = []
    for ct_d, region in enumerate(regions):
        lat_slice = get_lats_for_region(region)

        to_avg = da.sel(lat=lat_slice)
        weights = np.cos(np.deg2rad(to_avg.lat))
        avg_val = to_avg.weighted(weights=weights).mean(['lat', 'lon'])
        avg_vals.append(avg_val)

    avg_vals = xr.concat(avg_vals, dim='region')
    avg_vals['region'] = list(regions)

    return avg_vals


def get_lats_for_region(region):
    """
    Given a region name, get the relevant latitude slice
    """
    if region == 'Global':
        lat_slice = slice(-90, 90)
    elif region == 'NH':
        lat_slice = slice(-tropics_bound, 90)
    elif region == 'tropics':
        lat_slice = slice(-tropics_bound, tropics_bound)
    elif region == 'SH':
        lat_slice = slice(-90, -tropics_bound)
    else:
        raise Exception('help!')

    return lat_slice


def add_text_regional_averages(da, ax, regions, fontsize=12):
    """
    Add text to plots showing regional average values for pre-defined domains

    Parameters
    ----------
    da : xr.DataArray
        Values to plot in lat/lon space
    ax : GeoAxesSubplot
        Axis to plot in
    domains : list
        Names of domains (bound are predefined, except the width of the tropics)
    fontsize : int
        Fontsize for added text. Default is 12

    Returns
    -------
    ax : GeoAxesSubplot
        Updated axes
    """

    avg_vals = calc_regional_averages(da, regions)

    for ct_d, region in enumerate(regions):
        ax.text(0.01, 0.5 - 0.1*ct_d, '%s: %0.1f' % (region, avg_vals[ct_d]),
                transform=ax.transAxes, fontsize=fontsize)

    return ax


def swap_hot_ranks(ds):
    """
    In all initial calculations (including all CMIP models), hot ranks were defined such that #1
    was the largest difference between hot and median. In all plots, hot ranks need to be defined
    such that an increase in ranks indicates an increase in the difference between hot and median.

    Parameters
    ----------
    ds : xr.Dataset
        Contains the originally calculated ranks

    Returns
    -------
    ds : xr.Dataset
        Contains the adjusted ranks
    """
    this_nyrs = len(ds.year)
    for v in list(ds.data_vars):
        if ('hot' in v) | ('max' in v):
            multiplier = -1
            offset = this_nyrs + 1
            ds[v] *= multiplier
            ds[v] += offset

    return ds


def get_mask_land_Greenland(land_cutoff=0.5, era5_ls_fname='/home/data/ERA5/fx/era5_lsmask_1x1.nc',
                            country_folder='/home/data/geom/ne_110m_admin_0_countries/'):
    """
    Get a binary land/no land mask, additionally masking Greenland

    Parameters
    ----------
    land_cutoff : float [0, 1]
        Required land fraction to be considered land
    era5_ls_fname : str
        Filename (including path) of location of ERA5 land mask
    country_folder : str
        Path to shape files with country outlines

    Returns
    -------
    is_land : xr.DataArray
        Boolean array indicating land (1) or not (0)
    """
    from helpful_utilities.geom import get_regrid_country

    lsmask_ERA5 = xr.open_dataarray(era5_ls_fname).squeeze()
    is_land_ERA5 = lsmask_ERA5 > land_cutoff

    da_greenland = get_regrid_country('Greenland', country_folder,
                                      is_land_ERA5.lat, is_land_ERA5.lon,
                                      dilate=True)

    is_land = is_land_ERA5 & ~da_greenland

    return is_land


def combine_pr_datasets(pr_datasources, procdir, percentile_width, pr_start_year):
    """
    Average across precipitation datasets in terms of their trends on extreme vs avg days.

    Parameters
    ----------
    pr_datasources : list
        Names of precipitation datasets to average. Trends are pre-calculated.
    procdir : str
        Directory holding pre-processed trends
    percentile_width : int
        Size of percentile bins for hot, cold, avg
    pr_start_year : int
        Starting year for precipitation trends. Note that some datasets start later.

    Returns
    -------
    avg_pr_maps : xr.Dataset
        The average across datasets in terms of different trends in tails versus middle
    """

    tails = 'hot', 'cold', 'avg'
    avg_pr_maps = []
    for ct2, tail in enumerate(tails):
        all_maps = []
        for ct, datasource in enumerate(pr_datasources):
            # Load relative precipitation trends
            fname = '%s/%s_precip_relative_trends_p%02i_%04i-2023.nc' % (procdir, datasource,
                                                                         percentile_width, pr_start_year)
            ds_relative_trends = xr.open_dataset(fname)
            to_save = ds_relative_trends['precip_%s' % (tail)]
            all_maps.append(to_save.rename('%s' % datasource))
        all_maps = xr.concat(all_maps, dim='datasource')
        avg_map = (all_maps.mean('datasource')).rename('precip_%s' % tail)
        avg_pr_maps.append(avg_map)
    avg_pr_maps = xr.merge(avg_pr_maps)

    return avg_pr_maps


def make_SEB_plots(SEB_rel_trends, precip_rel_trends, terms, figname, figdir):
    """
    Make plots of relative trends in SEB terms and precipitation.

    Trends are the difference between those on hot/cold days and average days

    Parameters
    ----------
    SEB_rel_trends : xr.Dataset
        Dataset of trends in SEB terms conditional on temperature
    precip_rel_trends : xr.Dataset
        Dataset of trends in precipitation conditional on temperature
    terms : list
        List of terms to plot. Options: T1, T2, sum, precip
    figname : str
        Figure name for saving
    figdir : str
        Directory to save figure

    Returns
    -------
    Nothing, figure is saved
    """

    term_opts = 'T1', 'T2', 'sum', 'precip'
    idx_to_plot = np.isin(term_opts, terms)
    longnames = np.array((r'$R^{\prime}(1-\overline{EF})$', r'$-EF^{\prime}\overline{R_n}$',
                          'Forcing+EF', 'precipitation'))[idx_to_plot]
    caption_names = np.array(('Forcing term', 'EF term', 'Forcing+EF', 'Precipitation'))[idx_to_plot]
    tails = 'hot', 'cold'

    fig = plt.figure(figsize=(20, 4*len(terms)))
    widths = [8, 0.3, 2, 8, 0.3]
    heights = 2*np.ones((len(terms),))
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)

    spec.update(wspace=0.1, hspace=0.1)

    letter_ct = 0
    for ct, term in enumerate(terms):
        for ct2, tail in enumerate(tails):
            if term == 'precip':
                to_plot = (precip_rel_trends['precip_%s' % (tail)] -
                           precip_rel_trends['precip_avg'])
            elif term == 'sum':
                to_plot = ((SEB_rel_trends['T1_%s' % tail] + SEB_rel_trends['T2_%s' % tail]) -
                           (SEB_rel_trends['T1_avg'] + SEB_rel_trends['T2_avg']))
            else:
                to_plot = (SEB_rel_trends['%s_%s' % (term, tail)] -
                           SEB_rel_trends['%s_avg' % (term)])

            cmap = plt.cm.RdBu_r
            bounds = np.arange(-16, 17, 4)
            units = 'W/m$^{2}$'
            if term == 'precip':
                bounds = bounds/4
                units = 'mm'
                cmap = plt.cm.BrBG

            cbar_label = 'Difference in %s\n trends (%s/%iy)' % (longnames[ct], units, trend_normalizer)

            col_ct = 3*ct2

            # Make map
            ax_map = fig.add_subplot(spec[ct, col_ct], projection=plotcrs)
            ax_map, pc = make_standard_map(to_plot, ax_map, cmap, bounds)

            # Add colorbar
            cax = fig.add_subplot(spec[ct, col_ct + 1])
            cb = plt.colorbar(pc, cax=cax, orientation='vertical', extend='both')
            cb.ax.tick_params(labelsize=labelsize)
            cb.set_label(cbar_label, fontsize=fontsize)
            ax_map.set_title('')
            ax_map.text(0.01, 0.03, '(%s) %s,\n%s minus average' % (letters[letter_ct],
                                                                    caption_names[ct], tail),
                        transform=ax_map.transAxes, fontsize=12)

            letter_ct += 1

            # Get average values for each region to put on map
            ax_map = add_text_regional_averages(to_plot, ax_map, regions)

    plt.savefig('%s/%s' % (figdir, figname), dpi=200, bbox_inches='tight')
