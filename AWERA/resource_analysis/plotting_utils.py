import xarray as xr
import sys
import getopt

from config import output_file_name, output_file_name_subset, start_year, final_year


def read_dataset_user_input():
    """"Interpret user input to open the dataset(s)

    Returns:
        nc (Dataset): Plotting input data (from processing)
    """

    help = """
        python plot_maps.py -c           : plot from files from combined output file
        python plot_maps.py -m max_id    : plot from files with maximal subset id of max_id
        python plot_maps.py -h           : display this help
        """

    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hm:c", ["help", "maxid=", "combined"])
        except getopt.GetoptError:  # User input not given correctly, display help and end
            print(help)
            sys.exit()
        for opt, arg in opts:
            if opt in ("-h", "--help"):  # Help argument called, display help and end
                print(help)
                sys.exit()
            elif opt in ("-m", "--maxid"):  # User Input maximal subset id given
                max_subset_id = int(arg)
                # find all subset files matching the settings in config.py - including all until max_subset_id
                all_year_subset_files = [output_file_name_subset.format(**{'start_year': start_year,
                                                                           'final_year': final_year,
                                                                           'lat_subset_id': subset_id,
                                                                           'max_lat_subset_id': max_subset_id})
                                         for subset_id in range(max_subset_id+1)]
                print('All data for {} to {} is read from subset_files from 0 to {}'.format(start_year, final_year,
                                                                                            max_subset_id))
                nc = xr.open_mfdataset(all_year_subset_files, concat_dim='latitude')
                # nc.to_netcdf('/cephfs/user/s6lathim/ERA5Data/results/processed_data_europe_2010_2020.nc')
            elif opt in ("-c", "--combined"):  # User Input to use combined file
                file_name = output_file_name.format(**{'start_year': start_year, 'final_year': final_year})
                nc = xr.open_dataset(file_name)
    else:
        file_name = output_file_name.format(**{'start_year': start_year, 'final_year': final_year})
        nc = xr.open_dataset(file_name)
    import numpy as np
    # TODO change this --> config
    # Ireland
    # lons = list(np.arange(-12, -5.0, .25))  # config.Data.all_lons
    # lats = list(np.arange(51, 56.25, .25))
    # UK
    # lons = list(np.arange(-7, 2.5, .25))  # config.Data.all_lons
    # lats = list(np.arange(49.5, 61.5, .25))
    # GER
    # lons = list(np.arange(5.5, 16.0, .25))  # config.Data.all_lons
    # lats = list(np.arange(47.0, 55.5, .25))
    # ES
    # lons = list(np.arange(-11.25, 6.5, .25))  # config.Data.all_lons
    # lats = list(np.arange(35.5, 47.0, .25))
    # FR
    # lons = list(np.arange(-5.75, 10.5, .25))  # config.Data.all_lons
    # lats = list(np.arange(42, 51.75, .25))
    # NO
    # lons = list(np.arange(4, 20.25, .25))  # config.Data.all_lons
    # lats = list(np.arange(57.5, 65.25, .25))
    # IT
    # lons = list(np.arange(6.5, 19.5, .25))  # config.Data.all_lons
    # lats = list(np.arange(35.5, 47.75, .25))
    # nc = nc.sel(latitude=lats)
    # nc = nc.sel(longitude=lons)


    return nc
