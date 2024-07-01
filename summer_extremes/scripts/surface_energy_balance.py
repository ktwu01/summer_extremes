from summer_extremes.utils import get_mask_land_Greenland, mask_start_end, shift_replace_SH, calc_heat_metrics
from summer_extremes.utils import rank_and_sort_heat_metrics
from helpful_utilities import stats as helpful_stats
from helpful_utilities.data_proc import get_smooth_clim, get_day_idx_temperature_percentiles_doy
import numpy as np
import xarray as xr
from glob import glob
import os

# Default options
start_year = 1958  # only using 1958 for SH
end_year = 2023
land_cutoff = 0.5

# Alternative options
# start_year = 1979  # to confirm results with shorter data record

procdir = '/home/data/projects/summer_extremes/proc'
figdir = '/home/kmckinnon/summer_extremes/figs'
datadir = '/home/data/ERA5/day'
country_folder = '/home/data/geom/ne_110m_admin_0_countries/'

# options for types of extremes metrics to calculate
extreme_names = ('seasonal_max', 'seasonal_min',
                 'cum_excess_hot', 'avg_excess_hot', 'ndays_excess_hot',
                 'cum_excess_cold', 'avg_excess_cold', 'ndays_excess_cold',
                 'AR1')

tvar = 't2m_x'
percentile_width = 5  # half-width for definition of hot, cold, average days
percentile_base = np.array([5, 50, 95])  # middle of windows for temperature percentiles

# Get land mask (excluding Greenland)
is_land = get_mask_land_Greenland()
# Subset to domain
lower_lat = -60
upper_lat = 80
is_land = is_land.sel(lat=slice(lower_lat, upper_lat))

# In all cases, trends are presented as /65years for consistency
# xarray calculates time trends in per nanosecond
nyrs = 65
slope_normalizer = 1e9*60*60*24*365*nyrs

start_date = '%04i-01-01' % start_year
end_date = '%04i-10-31' % end_year
seconds_per_hour = 3600

# load warm season information
da_doy = xr.open_dataarray('%s/ERA5_hottest_doys_%s.nc' % (procdir, tvar))
ndays_first_half = da_doy.sel(dayofyear=slice(0, 365/2)).sum('dayofyear')
end_SH_idx = np.where(ndays_first_half == 0)[0][-1]
end_SH_lat = da_doy.lat[end_SH_idx].data + 0.5
print(end_SH_lat)

# # Data preprocessing
#
# Required variables for SEB equation:
# - temperature (tx)
# - shortwave
# - longwave
# - latent heat (for evaporative fraction)
#
# If already done, proceed to analysis block where variables are loaded

varnames = ('t2m_x', 'surface_latent_heat_flux',
            'surface_net_solar_radiation', 'surface_net_thermal_radiation')
# All ERA5 fluxes are, by default, positive downwards
# We want LW and LH to be positive upwards
# Fluxes were averaged from daily values, so to convert from Joules to W/m2, multiply by seconds per hour
unit_multiplier = 1, (-1/seconds_per_hour), (1/seconds_per_hour), (-1/seconds_per_hour)

savename = '%s/warm_season_for_SEB_%04i-%04i.nc' % (procdir, start_year, end_year)
clim_savename = '%s/SEB_clims_%04i-%04i.nc' % (procdir, start_year, end_year)

if os.path.isfile(savename):
    ds_anom_warm_season = xr.open_dataset(savename).load()
else:

    ds = []
    for ct_v, v in enumerate(varnames):
        print('Loading %s' % v)
        files = sorted(glob('%s/%s/1x1/*.nc' % (datadir, v)))
        da = xr.open_mfdataset(files).convert_calendar('365_day')[v]
        # Multiply units if needed
        da *= unit_multiplier[ct_v]

        # Mask to land
        da = da.where(is_land)
        # Mask to our desired time span
        da = da.sel(time=slice(start_date, end_date)).load()
        ds.append(da)

    ds = xr.merge(ds)

    # Derive EF
    da_ef = (ds['surface_latent_heat_flux'] /
             (ds['surface_net_solar_radiation'] - ds['surface_net_thermal_radiation']))

    # Mask large values (occur when SW and LW are closely balanced)
    # Mask negative values (assume no net condensational heating + positive surface forcing)
    da_ef = da_ef.where((da_ef >= 0) & (da_ef < 2))

    ds['evaporative_fraction'] = da_ef

    # Calculate and save annual cycle

    if os.path.isfile(clim_savename):
        all_clim = xr.open_dataset(clim_savename)
    else:
        all_clim = []
        for v in list(ds.data_vars):
            print('Calculating climatology for %s' % v)
            clim = xr.apply_ufunc(get_smooth_clim,
                                  ds[v].groupby('time.dayofyear').mean(),
                                  input_core_dims=[["dayofyear"]],
                                  output_core_dims=[["dayofyear"]],
                                  vectorize=True)
            all_clim.append(clim)

        all_clim = xr.merge(all_clim)
        all_clim.to_netcdf(clim_savename)

    # Remove climatology
    ds_anom = ds.groupby('time.dayofyear') - all_clim

    # Subset to warm season
    ds_anom_warm_season = ds_anom.groupby('time.dayofyear').where(da_doy == 1).load()

    # Mask out start/end to match other data analyses
    ds_anom_warm_season = ds_anom_warm_season.map(lambda da: mask_start_end(da, start_year,
                                                                            end_year, end_SH_lat))

    # Save
    ds_anom_warm_season.to_netcdf(savename)

# Identify hot, cold, average days
dataname = 'ERA5_t2m_x'
idx_all = get_day_idx_temperature_percentiles_doy(ds_anom_warm_season['t2m_x'],
                                                  dataname, percentile_width, percentile_base,
                                                  start_year, end_year, procdir)


# # Make SEB prediction of temperature
# Forcing term uses LW after regressing out temperature
# e.g. da_F = LW' - beta*T'
# where primes indicate anomalies from the seasonal cycle
da_F = xr.apply_ufunc(helpful_stats.get_residual,
                      ds_anom_warm_season['t2m_x'],
                      ds_anom_warm_season['surface_net_thermal_radiation'],
                      input_core_dims=[["time"], ["time"]],
                      output_core_dims=[["time"]],
                      vectorize=True)
da_F = da_F.transpose('time', 'lat', 'lon')

# Load or save terms based on climatology
# Subset to warm season first
all_clim = xr.open_dataset(clim_savename)
all_clim_warm_season = all_clim.groupby('dayofyear').where(da_doy == 1)

EF_bar = all_clim_warm_season['evaporative_fraction'].mean('dayofyear')
Rn_bar = (all_clim_warm_season['surface_net_solar_radiation'] -
          all_clim_warm_season['surface_net_thermal_radiation']).mean('dayofyear')

# Convert to ranks and save
ranks_savename = '%s/T_SEB_ranks_%04i-%04i.nc' % (procdir, start_year, end_year)

if os.path.isfile(ranks_savename):
    ds_ranks_SEB = xr.open_dataset(ranks_savename)
else:
    # Calculate temperature anomalies based on surface energy budget (Eqn (1))
    T1 = (ds_anom_warm_season['surface_net_solar_radiation'] - da_F)*(1 - EF_bar)
    T2 = ds_anom_warm_season['evaporative_fraction']*Rn_bar

    T_SEB = T1 - T2
    T_SEB = T_SEB.transpose('time', 'lat', 'lon')

    # move SH forward to match analysis of temperatures
    T_SEB = shift_replace_SH(T_SEB, end_SH_lat)

    # cutoff first year, because we've shifted SH forward
    T_SEB = T_SEB.sel(time=slice('%i' % (start_year + 1), '%i' % end_year))

    # calculate anomaly from sample median
    anom_from_median = T_SEB.groupby('time.year') - T_SEB.groupby('time.year').median()

    # calculate heatwave metrics based on these anomalies
    ds_metrics_SEB = calc_heat_metrics(anom_from_median, extreme_names)

    ds_ranks_SEB = rank_and_sort_heat_metrics(ds_metrics_SEB)

    del ds_metrics_SEB

    ds_ranks_SEB.to_netcdf(ranks_savename)


# # For Figure 2, calculate trends in each term in Eqn. (1) on hot, average, and cold days
# Plot trends in the two terms on hot vs avg and cold vs avg days
savename = '%s/SEB_relative_trends_p%02i_%04i-%04i.nc' % (procdir, percentile_width,
                                                          start_year, end_year)

if os.path.isfile(savename):
    ds_relative_trends = xr.open_dataset(savename)
else:
    T1a = (ds_anom_warm_season['surface_net_solar_radiation'])*(1 - EF_bar)
    T1b = -da_F*(1 - EF_bar)
    T2 = -ds_anom_warm_season['evaporative_fraction']*Rn_bar
    T1 = T1a + T1b
    total = T1 + T2

    terms = 'T1a', 'T1b', 'T1', 'T2', 'total'
    ds_relative_trends = xr.Dataset()
    for term in terms:
        for p in percentile_base:
            print(term, p)
            this_term = eval(term).copy()
            this_idx = idx_all['base_p_%02i' % p]

            this_term = this_term.where(this_idx & is_land)
            beta = this_term.polyfit(dim='time', deg=1)
            beta = slope_normalizer*(beta['polyfit_coefficients'].sel(degree=1))
            ds_relative_trends['%s_p%02i' % (term, p)] = beta
            del this_term, this_idx, beta

    ds_relative_trends.to_netcdf(savename)
