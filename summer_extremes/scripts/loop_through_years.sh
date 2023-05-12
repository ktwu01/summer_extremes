for yy in {2022..2023}
do
    echo $yy
    # python download_hourly_era5.py --year $yy
    python3.9 process_hourly_era5.py --year $yy
done
