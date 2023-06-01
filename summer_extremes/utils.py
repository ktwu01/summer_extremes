import numpy as np
import pandas as pd
import xarray as xr
import os
from subprocess import check_call
from record_breaking_heat import utils as heat_utils
from statsmodels.regression.quantile_regression import QuantReg


def hello_world(x1, x2):
    print(x1)
    print(x2)


def process_ghcnd(yr_start, yr_end, ghcnd_dir='/home/data/GHCND', var_names=['TMIN', 'TMAX'], country_list=None):
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


def fit_seasonal_cycle(da_fit, varname, nseasonal, return_beta=False):
    """
    Parameters
    ----------
    da_fit : xr.DataArray
        Data to fit
    varname : str
        Name of variable being fit
    nseasonal : int
        Number of seasonal harmonics to use
    return_beta : bool
        Whether to return the associated regression coefficients.
    Returns
    -------
    ds_fitted : xr.Dataset
        Dataset containing original data, fitted data, and residual
    """
    # number of predictors
    npred = 1 + 2*nseasonal  # seasonal harmonics + intercept
    nt = len(da_fit.time)

    # create design matrix
    # seasonal harmonics
    doy = da_fit['time.dayofyear']
    omega = 1/365.25

    X = np.empty((npred, nt))
    X[0, :] = np.ones((nt, ))
    for i in range(nseasonal):
        s = np.exp(2*(i + 1)*np.pi*1j*omega*doy)
        X[(1 + 2*i):(1 + 2*(i+1)), :] = np.vstack((np.real(s), np.imag(s)))

    X_mat = np.matrix(X).T

    if 'station' in da_fit.coords:  # station data, will have missing values, so need to loop through
        ds_fitted = []
        ds_residual = []

        for this_station in da_fit.station:
            this_X = X_mat.copy()
            this_y = da_fit.sel({'station': this_station}).values.copy()
            has_data = ~np.isnan(this_y)

            if np.isnan(this_y).all():
                continue

            this_y = this_y[has_data]
            this_X = this_X[has_data, :]
            this_y = np.matrix(this_y).T

            # fit
            beta = np.linalg.multi_dot(((np.dot(this_X.T, this_X)).I, this_X.T, this_y))

            # predict
            yhat = np.dot(X_mat, beta)
            yhat = np.array(yhat).flatten()
            residual = da_fit.sel({'station': this_station}).copy() - yhat

            ds_fitted.append(da_fit.sel({'station': this_station}).copy(data=yhat))
            ds_residual.append(residual)

        ds_fitted = xr.concat(ds_fitted, dim='station')
        ds_fitted = ds_fitted.to_dataset(name='%s_fit' % varname)
        ds_fitted['%s_residual' % varname] = xr.concat(ds_residual, dim='station')

    else:  # reanalysis
        nt_fit, nlat, nlon = da_fit.shape
        vals = da_fit.values.reshape((nt_fit, nlat*nlon))
        has_data = ~np.isnan(vals[0, :])  # in case of masking
        y_mat = np.matrix(vals[:, has_data])
        del vals

        # fit
        beta = np.linalg.multi_dot(((np.dot(X_mat.T, X_mat)).I, X_mat.T, y_mat))
        del y_mat

        # predict
        yhat_data = np.dot(X_mat, beta)
        yhat = np.nan*np.ones((nt_fit, nlat*nlon))
        yhat[:, has_data] = yhat_data
        del yhat_data
        ds_fitted = da_fit.copy(data=np.array(yhat).reshape((nt, nlat, nlon))).to_dataset(name='%s_fit' % varname)
        del yhat
        residual = da_fit - ds_fitted['%s_fit' % varname]
        ds_fitted['%s_residual' % varname] = residual

    if return_beta:
        return ds_fitted, beta
    else:
        return ds_fitted


def fit_qr_residual_boot(ds, months, variables, qs_to_fit, nboot, max_iter=10000, lastyear=2021,
                         gmt_fname='/home/data/BEST/Land_and_Ocean_complete.txt', lowpass_freq=1/10, butter_order=3,
                         savedir=None):
    """Fit a quantile regression model with GMT as covariate. Use the residual bootstrap.

    Parameters
    ----------
    ds : xr.Dataset
        Contains data for a given station
    months : np.array
        Contains set of months to fit QR for
    variables : list
        Contains names of variables in dataset to fit QR on
    qs_to_fit : np.array
        Array of quantiles to fit (independently - noncrossing is not enforced)
    nboot : int
        Number of times to bootstrap data (block size of one year) and refit QR model
    max_iter : int
        Maximum number of iterations for QuantReg model
    lastyear : int
        Last year to calculate trend with
    gmt_fname : str
        Local location of GMT time series from BEST
    lowpass_freq : float
        Desired cutoff frequency for Butterworth filter (in 1/yr)
    butter_order : int
        Desired order for Butterworth filter
    savedir : None or str
        If None, return output. If string, save in that directory

    Returns
    -------
    ds_QR : xr.Dataset
        Contains all quantile regression trends and pvals, as well as bootstrapped trends

    """
    all_QR = []
    # Loop through variables and months, then merge and save
    for this_month in months:
        # initialize dataset for each month
        ds_QR = xr.Dataset(coords={'qs': qs_to_fit, 'sample': np.arange(nboot), 'order': np.arange(2)})

        # Get days of year for correct month or month range
        if this_month > 12:
            start_month = int(str(this_month).split('0')[0])
            end_month = int(str(this_month).split('0')[-1])
        else:
            start_month = this_month
            end_month = this_month

        for var_ct, this_var in enumerate(variables):
            # initialize arrays
            beta_qr = np.nan*np.ones((len(qs_to_fit), 2))
            pval_qr = np.nan*np.ones((len(qs_to_fit)))
            beta_qr_boot = np.nan*np.ones((len(qs_to_fit), nboot))

            # Keep the bootstrap seed the same across months, variables, and stations
            np.random.seed(123)

            # fit on this data only
            time_idx = (ds.time.dt.month >= start_month) & (ds.time.dt.month <= end_month)
            this_da = ds[this_var].sel(time=time_idx).sel(time=slice('%04i' % lastyear))

            # global mean temperature time series as a stand-in for climate change in the regression model
            da_gmt = heat_utils.get_GMT(lowpass_freq=lowpass_freq, gmt_fname=gmt_fname, butter_order=butter_order)
            # resample GMT to daily, and match data time stamps
            da_gmt = da_gmt.resample(time='1D').interpolate('linear')
            cc = da_gmt.sel(time=this_da['time'])
            cc -= np.mean(cc)

            this_x = cc
            this_y = this_da.copy()
            pl = ~np.isnan(this_y)
            if np.sum(pl) == 0:  # case of no data
                ds_QR['beta_QR_%s' % this_var] = (('qs', 'order'), beta_qr)
                ds_QR['pval_QR_%s' % this_var] = (('qs'), pval_qr)
                ds_QR['beta_QR_boot_%s' % this_var] = (('qs', 'sample'), beta_qr_boot)
                continue

            this_x_vec = this_x[pl].values
            this_y_vec = this_y[pl].values

            # Add jitter since data is rounded to 0.1
            half_width = 0.05
            jitter = 2*half_width*np.random.rand(len(this_y_vec)) - half_width
            this_y_vec += jitter

            this_x_vec = np.vstack((np.ones(len(this_x_vec)), this_x_vec)).T

            model = QuantReg(this_y_vec, this_x_vec)

            for ct_q, q in enumerate(qs_to_fit):
                mfit = model.fit(q=q, max_iter=max_iter)
                if mfit.iterations < max_iter:  # only save fit if it has converged
                    beta_qr[ct_q, :] = mfit.params
                    pval_qr[ct_q] = mfit.pvalues[-1]

            # Bootstrap with block size of one year to assess significance of differences
            yrs = np.unique(this_y['time.year'])

            for kk in range(nboot):
                # use the same years for each percentile in each bootstrap sample
                new_yrs = np.random.choice(yrs, size=len(yrs))
                # and therefore the same x values
                x_boot = []
                for yy in new_yrs:
                    x_boot.append(this_x.sel(time=slice('%04i' % yy, '%04i' % yy)))

                x_boot = xr.concat(x_boot, dim='time')

                for ct_q, q in enumerate(qs_to_fit):
                    # trend estimated using original dataset
                    signal = beta_qr[ct_q, 0] + beta_qr[ct_q, 1]*this_x
                    # residual from that trend
                    residual = this_y - signal

                    # signal given the bootstrapped sample of x's
                    boot_signal = beta_qr[ct_q, 0] + beta_qr[ct_q, 1]*x_boot

                    # resample the residuals, then add back to boot_signal
                    residual_boot = []
                    for yy in new_yrs:
                        residual_boot.append(residual.sel(time=slice('%04i' % yy, '%04i' % yy)))
                    residual_boot = xr.concat(residual_boot, dim='time')
                    y_boot = boot_signal + residual_boot

                    pl = ~np.isnan(y_boot)
                    if np.sum(pl) == 0:  # case of no data
                        continue
                    this_x_vec = x_boot[pl].values
                    this_y_vec = y_boot[pl].values

                    # Add jitter since data is rounded to 0.1
                    jitter = 2*half_width*np.random.rand(len(this_y_vec)) - half_width
                    this_y_vec += jitter

                    this_x_mat = np.vstack((np.ones(len(this_x_vec)), this_x_vec)).T
                    model = QuantReg(this_y_vec, this_x_mat)

                    mfit = model.fit(q=q, max_iter=max_iter)
                    if mfit.iterations < max_iter:  # only save fit if it has converged
                        beta_qr_boot[ct_q, kk] = mfit.params[-1]
            # within variable and month loop
            ds_QR['beta_QR_%s' % this_var] = (('qs', 'order'), beta_qr)
            ds_QR['pval_QR_%s' % this_var] = (('qs'), pval_qr)
            ds_QR['beta_QR_boot_%s' % this_var] = (('qs', 'sample'), beta_qr_boot)

        all_QR.append(ds_QR)

    all_QR = xr.concat(all_QR, dim='month')
    all_QR['month'] = months
    if savedir is None:
        return all_QR
    else:
        all_QR.to_netcdf('%s/%s_qr.nc' % (savedir, ds.station.values))
