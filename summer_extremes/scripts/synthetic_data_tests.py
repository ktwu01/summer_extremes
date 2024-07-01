"""
Test rank averaging method using synthetic data
"""

import numpy as np
import xarray as xr

procdir = '/home/data/projects/summer_extremes/proc'
halfwidth = 45
alpha_fdr = 0.05

ndof = 50
ndays_per_summer = halfwidth*2 + 1
nyrs = 65
N = 100

variance_changes = np.arange(0.05, 0.51, 0.05)  # fractional
mean_changes = np.arange(0.25, 2.1, 0.25)

# get null hypothesis
null_indv = []
null_avg = []
for kk in range(N):
    this_data = np.random.randn(nyrs, ndays_per_summer, ndof)
    this_sample_median = np.median(this_data, axis=1)
    this_anom = this_data - this_sample_median[:, np.newaxis, :]

    # calculate max for each year
    seasonal_max = np.max(this_anom, axis=1)

    da_seasonal_max = xr.DataArray(seasonal_max,
                                   dims=('year', 'location'))

    da_ranks = (-da_seasonal_max).rank(dim='year')
    avg_rank = da_ranks.mean('location')
    indiv_rank_beta = da_ranks.polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)
    avg_rank_beta = avg_rank.polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)

    null_indv.append(indiv_rank_beta)
    null_avg.append(avg_rank_beta)

null_indv = xr.concat(null_indv, dim='sample')

null_avg = xr.concat(null_avg, dim='sample')

null95_indv = null_indv.quantile(q=(2.5/100, 97.5/100), dim=['sample', 'location'])

null95_avg = null_avg.quantile(q=(2.5/100, 97.5/100), dim=['sample'])

null_indv_vec = null_indv.data.flatten()
null_avg = null_avg.data.flatten()

ncases = 3
sig_indiv = np.empty((N, ncases, len(mean_changes), len(variance_changes)))
sig_avg = np.empty((N, ncases, len(mean_changes), len(variance_changes)))

for ct_m, mean_change in enumerate(mean_changes):
    for ct_v, var_change in enumerate(variance_changes):
        print(ct_m, ct_v)
        for kk in range(N):

            # case 1: no change in mean or variance
            data_case1 = np.random.randn(nyrs, ndays_per_summer, ndof)

            # case 2: change in mean, no change in variance
            delta_ts = np.linspace(0, mean_change, nyrs)
            delta_ts -= np.mean(delta_ts)
            data_case2 = delta_ts[:, np.newaxis, np.newaxis] + data_case1.copy()

            # case 3: no change in mean, only change in variance

            delta_std_ts = np.sqrt(np.linspace(1, 1 + var_change, nyrs))
            data_case3 = delta_std_ts[:, np.newaxis, np.newaxis]*data_case1.copy()

            # case 4: changes in both
            data_case4 = (data_case3.copy() + delta_ts[:, np.newaxis, np.newaxis])

            # Perform suggested method
            avg_ranks = []
            for ct in range(ncases):
                if ct == 0:
                    this_data = data_case2.copy()
                elif ct == 1:
                    this_data = data_case3.copy()
                else:
                    this_data = data_case4.copy()

                # remove median for each year
                this_sample_median = np.median(this_data, axis=1)
                this_anom = this_data - this_sample_median[:, np.newaxis, :]

                # calculate max for each year
                seasonal_max = np.max(this_anom, axis=1)

                da_seasonal_max = xr.DataArray(seasonal_max,
                                               dims=('year', 'location'))

                da_ranks = (-da_seasonal_max).rank(dim='year')
                avg_rank = da_ranks.mean('location')

                indiv_rank_beta = da_ranks.polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)
                avg_rank_beta = avg_rank.polyfit(dim='year', deg=1)['polyfit_coefficients'].sel(degree=1)

                # calculate p-values
                pvals = []
                for location in indiv_rank_beta.location:
                    this_trend = indiv_rank_beta.sel(location=location)
                    slope_vec = np.hstack((this_trend, null_indv_vec))
                    obs_idx = np.where(np.argsort(slope_vec) == 0)[0][0]
                    pval = obs_idx/len(null_indv_vec)
                    if pval > 0.5:  # other tail
                        pval = 1 - pval
                    # two tailed
                    pval *= 2
                    pvals.append(pval)

                # do FDR control
                pval_vec = np.array(pvals)
                a = np.arange(len(pval_vec)) + 1
                try:
                    cutoff_idx = np.where(np.sort(pval_vec) <= alpha_fdr*a/len(a))[0][-1]
                    cutoff_pval = np.sort(pval_vec)[cutoff_idx]
                except IndexError:
                    cutoff_pval = 0  # nothing sig
                is_sig = pval_vec <= cutoff_pval
                frac_sig_indiv = np.sum(is_sig)/len(is_sig)
                is_sig_avg = np.abs(avg_rank_beta) > np.abs(null95_avg).mean()

                sig_indiv[kk, ct, ct_m, ct_v] = frac_sig_indiv
                sig_avg[kk, ct, ct_m, ct_v] = is_sig_avg

da_sig_indiv = xr.DataArray(sig_indiv, dims=('sample', 'case', 'meanchange', 'varchange'),
                            coords={'sample': np.arange(N),
                                    'case': list(('mean_only', 'var_only', 'mean_var')),
                                    'meanchange': mean_changes,
                                    'varchange': variance_changes})

da_sig_avg = xr.DataArray(sig_avg, dims=('sample', 'case', 'meanchange', 'varchange'),
                          coords={'sample': np.arange(N),
                                  'case': list(('mean_only', 'var_only', 'mean_var')),
                                  'meanchange': mean_changes,
                                  'varchange': variance_changes})

da_sig_indiv.to_netcdf('%s/synthetic_indiv.nc' % procdir)
da_sig_avg.to_netcdf('%s/synthetic_avg.nc' % procdir)
