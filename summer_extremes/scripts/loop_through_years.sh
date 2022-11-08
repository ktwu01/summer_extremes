for yy in {1959..2022}
do
    echo $yy
    python download_hourly_era5.py --year $yy
    python process_hourly_era5.py --year $yy
done
