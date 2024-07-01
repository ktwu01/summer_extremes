"""
This script performs the main analysis in terms of calculating the ranks of the difference between the warm season
extremes and the seasonal median, including sensitivity tests for comparing to a 5-year running average of the seasonal
median, or using the mean instead.

The general process is:
- Load data
- Subset to land, except Greenland
- Estimate seasonal cycle
- Identify hottest day by latitude band
- Remove seasonal cycle
- Subset to hottest subseason
- For hemisphere that is split across the calendar year, roll forward by 1/2 year
- Remove the seasonal median
- Calculate various metrics relevant to extremes
- Transform to ranks
"""

from record_breaking_heat import utils as heat_utils
from summer_extremes import utils as summer_utils
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
import os
from helpful_utilities.geom import get_regrid_country


# Local (KM) directories for data
era5_dir = '/home/data/ERA5/day'
era5_ls = '/home/data/ERA5/fx/era5_lsmask_1x1.nc'
procdir = '/home/data/projects/summer_extremes/proc'
figdir = '/home/kmckinnon/summer_extremes/figs'
country_folder = '/home/data/geom/ne_110m_admin_0_countries/'  # outlines of countries

# Args for main text
dataname = 'ERA5'
start_year = 1958  # only using 1958 for SH
end_year = 2023
lower_lat = -60
upper_lat = 80

##### Alternatives #####
# for comparison to satellite era alone
# start_year = 1979

# for comparison to CHIRTS
# start_year = 1983
# end_year = 2016

# for analysis of CMIP
# start_year = 2024
# end_year = 2099

# for analysis of station data
# dataname = 'GHCND'

# for analysis of CHIRTS
# dataname = 'CHIRTS'

# for analysis of CMIP6 data, form is MODEL/VARIANT
# dataname = 'ACCESS-ESM1-5/r11i1p1f1'
##### End alternatives #####

if dataname == 'ERA5':
    tvar = 't2m_x'
    t_data_dir = '/home/data/ERA5/day'
    is_gridded = True
elif dataname == 'GHCND':
    tvar = 'TMAX'
    t_data_dir = '/home/data/GHCND'
    is_gridded = False
elif '/' in dataname:  # CMIP sims
    tvar = 'tasmax'
    t_data_dir = '/home/data/CMIP6'
    model_name = dataname.split('/')[0]
    variant_number = dataname.split('/')[1]
    is_gridded = True
elif dataname == 'CHIRTS':
    tvar = 'Tmax'
    t_data_dir = '/home/data/CHIRTS'
    start_year = 1983
    end_year = 2016
    is_gridded = True

season = 'warm'  # type of season (currently 'warm' is only option)
halfwidth = 45  # half-length of warm season
land_cutoff = 0.5  # land is anything with a land fraction greater than this

# types of extremes metrics to calculate
extreme_names = ('seasonal_max', 'seasonal_min',
                 'cum_excess_hot', 'avg_excess_hot', 'ndays_excess_hot',
                 'cum_excess_cold', 'avg_excess_cold', 'ndays_excess_cold',
                 'AR1')

# GHCND specs
cutoff_years = 0.75
cutoff_intraseason = 1
ghcnd_to_ERA5_dict = {'TMAX': 't2m_x', 'TMIN': 't2m_n'}
ERA5_to_ghcnd_dict = {'t2m_x': 'TMAX', 't2m_n': 'TMIN'}

# Strings for data saving
savestr = '%s_%s_years-%04i-%04i' % (dataname.replace('/', '-'), tvar, start_year, end_year)
savestr2 = '_%s-season_2x%02i-days_lat%02i-%02i' % (season, halfwidth, lower_lat, upper_lat)
savestr = savestr + savestr2

# Load correct land mask for gridded datasets
if is_gridded:
    lsmask = xr.open_dataarray(era5_ls).squeeze()

# if CMIP, overwrite
if '/' in dataname:  # CMIP sims
    mask_name = glob('%s/fx/sftlf_fx_%s*.nc' % (t_data_dir, model_name))[0]
    lsmask = xr.open_dataset(mask_name)['sftlf']
    if lsmask.max() > 1:
        lsmask /= 100

da_greenland = get_regrid_country('Greenland', country_folder, lsmask.lat, lsmask.lon, dilate=True)
is_land = (lsmask > land_cutoff) & ~da_greenland

if (dataname == 'ERA5') | (dataname == 'CHIRTS'):
    if dataname == 'ERA5':
        files = sorted(glob('%s/%s/1x1/%s_????_1x1.nc' % (t_data_dir, tvar, tvar)))
        years = np.array([int(f.split('/')[-1].split('_')[-2]) for f in files])
    else:
        files = sorted(glob('%s/1x1/%s.????_1x1.nc' % (t_data_dir, tvar)))
        years = np.array([int(f.split('/')[-1].split('.')[-2].split('_')[0]) for f in files])

    use_files = np.isin(years, np.arange(start_year, end_year + 1))
    files = np.array(files)[np.where(use_files)[0].astype(int)]
    da = xr.open_mfdataset(files).convert_calendar('365_day')[tvar]

elif dataname == 'GHCND':

    # need longer period for SH stations, so load data separately
    f_station_list = '%s/ghcnd-stations.txt' % t_data_dir
    f_inventory = '%s/ghcnd-inventory.txt' % t_data_dir
    inventory_dict = heat_utils.get_ghcnd_inventory_dict([tvar], f_inventory)

    # Get SH stations
    lat_range = lower_lat, 0
    lon_range = -180, 180
    datadir_ghcnd = '%s/%04i-%04i' % (t_data_dir, start_year, end_year)
    station_list, lats, lons = heat_utils.get_ghcnd_station_list([tvar], inventory_dict,
                                                                 lat_range, lon_range,
                                                                 start_year, end_year)

    ds_SH = heat_utils.get_ghcnd_ds(station_list, [tvar], datadir_ghcnd,
                                    start_year, end_year, subset_years=True)

    # Get NH stations
    if start_year == 1958:
        NH_start = 1959
    else:
        NH_start = start_year
    lat_range = 0, upper_lat
    datadir_ghcnd = '%s/%04i-%04i' % (t_data_dir, NH_start, end_year)
    station_list, lats, lons = heat_utils.get_ghcnd_station_list([tvar], inventory_dict,
                                                                 lat_range, lon_range,
                                                                 start_year + 1, end_year)

    ds_NH = heat_utils.get_ghcnd_ds(station_list, [tvar], datadir_ghcnd,
                                    start_year + 1, end_year, subset_years=True)

    # merge
    ds = xr.concat((ds_NH, ds_SH), dim='station')

    # remove leap days, and subset to desired years
    da = ds.convert_calendar('365_day')[tvar]
    da = da.sel(time=slice('%04i' % (start_year), '%04i' % (end_year)))

    # want time x station
    if da.shape[0] != len(da.time):
        da = da.T

    da_subset = da.load()

elif '/' in dataname:  # CMIP sims

    hist_files = sorted(glob('%s/historical/day/tasmax/%s/%s/g*/*.nc' % (t_data_dir,
                                                                         model_name, variant_number)))
    ssp_files = sorted(glob('%s/ssp370/day/tasmax/%s/%s/g*/*.nc' % (t_data_dir, model_name, variant_number)))
    da = xr.open_mfdataset(hist_files + ssp_files).convert_calendar('365_day')[tvar]

else:
    raise Exception('TODO')

if is_gridded:  # gridded products

    da = da.sel(time=slice('%04i' % (start_year), '%04i' % (end_year)))

    # round lat for matching between landmask and data -- sometimes an issue in CMIP models
    da = da.assign_coords(lat=np.round(da.lat, 3))
    is_land = is_land.assign_coords(lat=np.round(is_land.lat, 3))

    # pull out desired domain
    da_subset = da.sel(lat=slice(lower_lat, upper_lat)).load()

    # mask to land
    da_subset = da_subset.where(is_land)

# create basis functions to remove seasonal cycle
time_vec = pd.date_range(start='1950-01-01', periods=365, freq='D')
doy = xr.DataArray(np.arange(1, 366), coords={'time': time_vec}, dims='time')
t_basis = (doy/365).values
nbases = 5
nt = len(t_basis)
bases = np.empty((nbases, nt), dtype=complex)
for counter in range(nbases):
    bases[counter, :] = np.exp(2*(counter + 1)*np.pi*1j*t_basis)

# get empirical average for the doy
empirical_sc = da_subset.groupby('time.dayofyear').mean()
mu = empirical_sc.mean(dim='dayofyear')

if len(empirical_sc.shape) == 3:  # gridded
    nday, nlat, nlon = empirical_sc.shape
    loc_len = nlat*nlon
elif len(empirical_sc.shape) == 2:  # in situ
    nday, nstations = empirical_sc.shape
    loc_len = nstations

# project zero-mean data onto basis functions
data = (empirical_sc - mu).data

coeff = 2/nt*(np.dot(bases, data.reshape((nday, loc_len))))

# reconstruct seasonal cycle
rec = np.real(np.dot(bases.T, np.conj(coeff)))
if len(empirical_sc.shape) == 3:
    rec = rec.reshape((nday, nlat, nlon))

da_rec = empirical_sc.copy(data=rec) + mu

# get correlation to confirm that we've done it correctly. This should be very high!
r2_ann = xr.corr(da_rec, empirical_sc, dim='dayofyear')

# Identify hottest day of the year for each latitude band for gridded products

if dataname == 'GHCND':  # for GHCND, use ERA5 seasonality
    seasonality_savename = '%s/ERA5_hottest_doys_%s.nc' % (procdir, ghcnd_to_ERA5_dict[tvar])
    if os.path.isfile(seasonality_savename):
        da_doy = xr.open_dataarray(seasonality_savename)
    else:
        raise Exception("Need to calculate seasonality with ERA5 first")

else:

    seasonality_savename = '%s/%s_hottest_doys_%s.nc' % (procdir, dataname.replace('/', '-'), tvar)
    if os.path.isfile(seasonality_savename):
        da_doy = xr.open_dataarray(seasonality_savename)
    else:

        # get hottest doy at each location
        hottest_day = da_rec.mean('lon').fillna(-999).argmax(dim='dayofyear')
        hottest_day = hottest_day.where(hottest_day >= 0)

        doy_mat = np.nan*np.ones((365, nlat))

        # for each latitude, perform masking
        # label as 1 if included, 0 otherwise
        doy_array = np.arange(1, 366)
        for ct_lat, this_lat in enumerate(hottest_day.lat):
            middle_day = hottest_day.sel(lat=this_lat).data
            if np.isnan(middle_day):  # all ocean
                continue
            first_day = middle_day - 45
            last_day = middle_day + 45
            if first_day < 0:
                first_day += 365
            if last_day > 365:
                last_day -= 365

            if last_day > first_day:  # all in one year
                keep_idx = (doy_array >= first_day) & (doy_array <= last_day)
            else:  # years are split
                keep_idx = (doy_array <= last_day) | (doy_array >= first_day)
            doy_mat[keep_idx, ct_lat] = 1

        da_doy = xr.DataArray(doy_mat, dims=('dayofyear', 'lat'),
                              coords={'dayofyear': np.arange(1, 366), 'lat': da_subset.lat})

        da_doy.to_netcdf(seasonality_savename)

# Remove the seasonal cycle
da_anom = da_subset.groupby('time.dayofyear') - da_rec

if dataname == 'GHCND':
    # match each station's latitude to the right doy
    da_doy_ghcnd = [da_doy.sel(lat=this_lat, method='nearest') for this_lat in da_anom.lat]
    da_doy = xr.concat(da_doy_ghcnd, dim='station')
    da_doy['lat'] = da_anom.lat  # reset to original latitude values

# Shift the SH 1/2 year forward so that DJF of year X-X+1 is associated with year X+1
# define SH as when seasonal cycle switches by ~180 days
# sometimes happens at e.g. -1.5S
if dataname == 'GHCND':
    end_SH_lat = 0
else:
    ndays_first_half = da_doy.sel(dayofyear=slice(0, 365/2)).sum('dayofyear')
    end_SH_idx = np.where(ndays_first_half == 0)[0][-1]
    end_SH_lat = da_doy.lat[end_SH_idx].data

# Subset to 91 day window around warmest day
da_anom_warm_season = da_anom.groupby('time.dayofyear').where(da_doy == 1)

# roll SH by 365/2 days so that years align across latitudes
idx_SH = da_anom_warm_season.lat <= end_SH_lat
try:
    da_anom_warm_season_SH = da_anom_warm_season.sel(lat=slice(end_SH_lat)).shift({'time': int(365/2)})
except:
    da_anom_warm_season_SH = da_anom_warm_season[:, idx_SH].shift({'time': int(365/2)})

# replace SH data with shifted data
if len(da_anom_warm_season.shape) == 3:  # gridded
    da_anom_warm_season[:, idx_SH, :] = da_anom_warm_season_SH
elif len(da_anom_warm_season.shape) == 2:  # in situ
    da_anom_warm_season[:, idx_SH] = da_anom_warm_season_SH

# cutoff first year, because we've shifted SH forward
da_anom_warm_season = da_anom_warm_season.sel(time=slice('%i' % (start_year + 1), '%i' % end_year))

# Remove stations with insufficient data in GHCND
if dataname == 'GHCND':

    season_length = halfwidth*2 + 1
    days_needed_per_season = cutoff_intraseason*season_length
    has_data = ~np.isnan(da_anom_warm_season)
    n_data = has_data.groupby('time.year').sum()
    frac_data = n_data/season_length

    bad_years = frac_data < cutoff_intraseason
    da_anom_warm_season = da_anom_warm_season.groupby('time.year').where(~bad_years)

    # want cutoff_years% coverage overall, and in the first and last decade
    bad_station = ((bad_years.sum('year') > (1 - cutoff_years)*len(bad_years.year)) |
                   (bad_years[:10, :].sum('year') > (1 - cutoff_years)*10) |
                   (bad_years[-10:, :].sum('year') > (1 - cutoff_years)*10))

    idx_good = np.where(~bad_station)[0]
    da_anom_warm_season = da_anom_warm_season[:, idx_good]

# Calculate ranks for various metrics of extremes minus the median

metric_savename = '%s/metrics_%s.nc' % (procdir, savestr)
rank_savename = '%s/ranks_%s.nc' % (procdir, savestr)

if os.path.isfile(metric_savename) & os.path.isfile(rank_savename):
    ds_metrics = xr.open_dataset(metric_savename)
    ds_ranks = xr.open_dataset(rank_savename)

else:
    sample_median = da_anom_warm_season.groupby('time.year').median()
    anom_from_median = da_anom_warm_season.groupby('time.year') - sample_median

    # calculate heatwave metrics based on these anomalies
    ds_metrics = summer_utils.calc_heat_metrics(anom_from_median, extreme_names)
    ds_ranks = summer_utils.rank_and_sort_heat_metrics(ds_metrics)

    # save
    ds_metrics.to_netcdf(metric_savename)
    ds_ranks.to_netcdf(rank_savename)

# Compare to a 5-year running average of the median
if dataname == 'ERA5':
    metric_savename_5yrmedian = '%s/metrics_5yrmedian_%s.nc' % (procdir, savestr)
    rank_savename_5yrmedian = '%s/ranks_5yrmedian_%s.nc' % (procdir, savestr)

    if os.path.isfile(metric_savename_5yrmedian) & os.path.isfile(rank_savename_5yrmedian):
        ds_metrics_5yrmedian = xr.open_dataset(metric_savename_5yrmedian)
        ds_ranks_5yrmedian = xr.open_dataset(rank_savename_5yrmedian)

    else:
        sample_median = da_anom_warm_season.groupby('time.year').median()
        sample_median_5 = sample_median.rolling(year=5, center=True, min_periods=1).mean()

        anom_from_median_5 = da_anom_warm_season.groupby('time.year') - sample_median_5

        # calculate heatwave metrics based on these anomalies
        ds_metrics_5yrmedian = summer_utils.calc_heat_metrics(anom_from_median_5, extreme_names)
        ds_ranks_5yrmedian = summer_utils.rank_and_sort_heat_metrics(ds_metrics_5yrmedian)

        # save
        ds_metrics_5yrmedian.to_netcdf(metric_savename_5yrmedian)
        ds_ranks_5yrmedian.to_netcdf(rank_savename_5yrmedian)

# Compare to using the mean as estimate of the middle of the distribution
if dataname == 'ERA5':
    metric_savename_mean = '%s/metrics_mean_%s.nc' % (procdir, savestr)
    rank_savename_mean = '%s/ranks_mean_%s.nc' % (procdir, savestr)

    if os.path.isfile(metric_savename_mean) & os.path.isfile(rank_savename_mean):
        ds_metrics_mean = xr.open_dataset(metric_savename_mean)
        ds_ranks_mean = xr.open_dataset(rank_savename_mean)

    else:
        sample_mean = da_anom_warm_season.groupby('time.year').mean()
        anom_from_mean = da_anom_warm_season.groupby('time.year') - sample_mean

        # calculate heatwave metrics based on these anomalies
        ds_metrics_mean = summer_utils.calc_heat_metrics(anom_from_mean, extreme_names)
        ds_ranks_mean = summer_utils.rank_and_sort_heat_metrics(ds_metrics_mean)

        # save
        ds_metrics_mean.to_netcdf(metric_savename_mean)
        ds_ranks_mean.to_netcdf(rank_savename_mean)
