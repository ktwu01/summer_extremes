"""
Calculate the relative trends in precipitation on different types of (temperature) days.
Used for observational datasets and CMIP6
"""
from summer_extremes.utils import get_mask_land_Greenland, get_cmip_tasmax_for_idx, get_cmip_mask
from summer_extremes.utils import calc_rel_precip_stats
from helpful_utilities.ncutils import lon_to_360
from helpful_utilities.data_proc import get_day_idx_temperature_percentiles_doy
import numpy as np
import xarray as xr
from glob import glob
import xesmf as xe


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
cmip6_dir = '/home/data/CMIP6'

percentile_width = 5  # half-width of percentile window
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
sec_per_day = 60*60*24

# Observational datasets
obs_pr_data = ['ERA5', 'MSWEP', 'PERSIANN', 'GPCC', 'CHIRPS', 'CPC']

# CMIP6 datasets: same models as we have used for temperature analyses
hist_global_fname = 'ranks_global_t2m_x_years-1958-2023_warm-season_2x45-days_lat-60-80.nc'
hist_ranks_Global = xr.open_dataset('%s/%s' % (procdir, hist_global_fname))
model_names = list(hist_ranks_Global.datasource.data)

# remove ERA5 which was saved in same file for plotting
model_names.remove('ERA5')

# split name and variant
models = ['-'.join(m.split('-')[:-1]) for m in model_names]
variants = [m.split('-')[-1] for m in model_names]

all_names = obs_pr_data + model_names

# Look at precipitation trends on hot/cold vs average days
for precip_name in all_names:

    # other products do not have long-term data
    if (start_year < 1970) & ((precip_name != 'ERA5') & ('-r' not in precip_name)):
        continue

    print(precip_name)

    # In most cases, use ERA5 to define hot/cold/average days
    # Exception is the climate models, where this will be overwritten
    temperature_source = 'ERA5_t2m_x'

    # Load data (all are already linearly interpolated to ERA5 grid)
    # All are processed to be in mm/day
    if precip_name == 'CPC':
        precip_dir = '/home/data/CPC'
        files = sorted(glob('%s/precip.????.nc' % (precip_dir)))
        da_precip = xr.open_mfdataset(files).convert_calendar('365_day')['precip']
        # Requires re-gridding onto ERA5 grid
        da_precip = da_precip.sortby('lat')
        regridder = xe.Regridder({'lat': da_precip.lat, 'lon': da_precip.lon},
                                 {'lat': is_land.lat.data, 'lon': is_land.lon.data},
                                 'bilinear',
                                 periodic=True, reuse_weights=False)

        da_precip = regridder(da_precip)
        da_precip = da_precip.rename('precip')
    elif precip_name == 'ERA5':
        # in my pre-processing script, I've averaged across hourly data to get daily,
        # so need to multiply by 24 hours
        # ERA5 in m, convert to mm
        mult = 24*1e3
        precip_dir = '/home/data/ERA5/day/total_precipitation/1x1'
        files = sorted(glob('%s/total_precipitation_????_1x1.nc' % (precip_dir)))
        da_precip = mult*(xr.open_mfdataset(files).convert_calendar('365_day')['total_precipitation'])
    elif precip_name == 'CHIRPS':  # only 1981-
        precip_dir = '/home/data/CHIRPS/1x1'
        files = sorted(glob('%s/chirps-v2.0.????.days_p25_1x1.nc' % (precip_dir)))
        da_precip = xr.open_mfdataset(files).convert_calendar('365_day')['pr']
    elif precip_name == 'GPCC':  # only 1982-2020
        precip_dir = '/home/data/GPCC'
        files = sorted(glob('%s/full_data_daily_v2022_10_????.nc' % (precip_dir)))
        da_precip = xr.open_mfdataset(files).convert_calendar('365_day')['precip']
        # switch lon
        da_precip = lon_to_360(da_precip)
    elif precip_name == 'PERSIANN':
        precip_dir = '/home/data/PERSIANN/1x1'
        files = sorted(glob('%s/PERSIANN-CDR.????.1x1.nc' % precip_dir))
        da_precip = xr.open_mfdataset(files).convert_calendar('365_day')['precipitation']
    elif precip_name == 'MSWEP':
        precip_dir = '/home/data/MSWEP_V280/Combined/1x1'
        files = sorted(glob('%s/MSWEP_V280_????.nc' % precip_dir))
        da_precip = xr.open_mfdataset(files).convert_calendar('365_day')['precipitation']
    elif '-r' in precip_name:  # climate models.
        this_model = '-'.join(precip_name.split('-')[:-1])
        this_variant = precip_name.split('-')[-1]
        hist_files = sorted(glob('%s/historical/day/pr/%s/%s/g*/*.nc' % (cmip6_dir, this_model,
                                                                         this_variant)))
        ssp_files = sorted(glob('%s/ssp370/day/pr/%s/%s/g*/*.nc' % (cmip6_dir, this_model,
                                                                    this_variant)))
        if len(hist_files + ssp_files) == 0:
            print('Missing precip files: %s' % precip_name)
            continue
        da_precip = xr.open_mfdataset((hist_files + ssp_files)).convert_calendar('365_day')['pr']
        da_precip = da_precip.sel(time=slice('%04i' % (start_year), '%04i' % (end_year)))
        # Units are kg/m2/s, so need to multiply by seconds per day to compare to other datasets
        da_precip *= sec_per_day
        # Round latitudes/longitudes for matching
        da_precip = da_precip.assign_coords(lat=np.round(da_precip.lat, 3),
                                            lon=np.round(da_precip.lon, 3))
        temperature_source = '%s_tasmax' % precip_name

    else:
        raise Exception('help! I do not recognize this datasource')

    # Get indices to mask for type of day
    if temperature_source == 'ERA5_t2m_x':
        # This has been already calculated in the SEB code
        idx_savename = '%s/%s_idx_w%02i_%04i-%04i.nc' % (procdir, temperature_source, percentile_width,
                                                         start_year, end_year)
        idx_all = xr.open_dataset(idx_savename)
        this_land = is_land.copy()
    else:  # CMIP models
        # Get temperature
        da_t = get_cmip_tasmax_for_idx(this_model, this_variant, start_year, end_year,
                                       lower_lat, upper_lat, land_cutoff, procdir, cmip6_dir)
        # Identify hot, cold, average days
        idx_all = get_day_idx_temperature_percentiles_doy(da_t, temperature_source, percentile_width,
                                                          percentile_base, start_year, end_year, procdir)
        # Get land mask
        this_land = get_cmip_mask(this_model, land_cutoff, cmip6_dir)

    # Mask to land
    da_precip = da_precip.where(this_land)

    # Calculate stats and trends in each type of day
    calc_rel_precip_stats(da_precip, idx_all, this_land, precip_name, percentile_width, percentile_base,
                          slope_normalizer, start_year, end_year, procdir)

    del da_precip, idx_all, this_land
