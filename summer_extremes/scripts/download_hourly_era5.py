import cdsapi
import click
from datetime import datetime, timedelta


@click.command()
@click.option('--year', help='Year of data to process', type=click.INT)
def download_hourly_era5_temperature(year, era5_daily_dir='/home/data/ERA5/day'):
    # Get date to avoid error of trying to download near-realtime
    today = datetime.now()
    maxdate = today - timedelta(days=6)

    # check if we are downloading present year, in which case we need to stop before present
    if maxdate.year == year:
        maxmonth = maxdate.month
        maxday = maxdate.day
    else:  # otherwise get full year
        maxmonth = 12
        maxday = 31

    months = ['%02i' % i for i in range(1, maxmonth + 1)]
    days = ['%02i' % i for i in range(1, 32)]
    days_short = ['%02i' % i for i in range(1, maxday + 1)]

    c = cdsapi.Client()
    if maxdate.year == year:
        # get most recent month separately
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': '2m_temperature',
                'year': '%04i' % year,
                'month': months[:-1],
                'day': days,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
            },
            '%s/t2m_hourly_%04i.nc' % (era5_daily_dir, year))

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': '2m_temperature',
                'year': '%04i' % year,
                'month': months[-1],
                'day': days_short,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
            },
            '%s/t2m_hourly_%04i_lastmonth.nc' % (era5_daily_dir, year))
    else:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': '2m_temperature',
                'year': '%04i' % year,
                'month': months,
                'day': days,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
            },
            '%s/t2m_hourly_%04i.nc' % (era5_daily_dir, year))


if __name__ == '__main__':
    download_hourly_era5_temperature()
