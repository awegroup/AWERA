# -*- coding: utf-8 -*-
"""Configuration file for wind resource analysis.

Attributes:
    start_year (4-digit int): Process the wind data starting from this year
    final_year (4-digit int): Process the wind data up to this year
    era5_data_dir (str): Directory path for reading era5 data files.
    model_level_file_name_format (str): Target name of the wind data files.
        Python's format() is used to fill in year and month at placeholders.
    surface_file_name_format (str): Target name of geopotential and surface
        pressure data files. Python's format() is
        used to fill in year and month at placeholders.

.FILL
.
.


.. _L137 model level definitions:
    https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels

"""
# TODO update docstring

# --------------------------- GENERAL
# Don't save plots directly as pdf to result_dir
plots_interactive = False

n_clusters = 8  # default: 8
n_pcs = 5  # default: 5

location_type_training, start_year_training, final_year_training = \
    '5000_locs', 2010, 2020
# ----------------------------------------------------------------
# -------------------------------- DATA - config input/output
# ----------------------------------------------------------------

# --------------------------- TIME --------------------------------------
start_year = 2010
final_year = 2020

# Only read data from January to year_final_month
# useful for testing on small amounts of data
year_final_month = 12  # default: 12

# --------------------------- LOCATION ----------------------------------
n_locs = -1  # -1: all, -2: manual
if n_locs == -1:
    locations = []
    location_type = 'all_locs'
elif n_locs == 5000:
    from location_selection import locations_europe_5000 as locations
    location_type = '5000_locs'
elif n_locs == 1000:
    from location_selection import locations_europe_1000 as locations
    location_type = '1000_locs'
elif n_locs == 100:
    from location_selection import locations_europe_5000 as locations
    locations = locations[4900:]
    location_type = '100_locs_predict'
elif n_locs == 22:
    # TODO include import of single locations here?
    locations = [(55.5, 3.25), (50, 13.5)]
    location_type = 'test_2_locations_22'
elif n_locs == 12:
    # TODO include import of single locations here?
    locations = [(55.5, 3.25)]
    location_type = 'test_2_locations_12'
elif n_locs == 21:
    # TODO include import of single locations here?
    locations = [(50, 13.5)]
    location_type = 'test_2_locations_21'
    # plus alpes -> south of bodensee
    # cabauw netherlands and north denmark
# !!! old, remove?
# Location processing (latitude, longitude)
# locations = [(58.25,-12.5), (63.0,-12.5), (61.25, -14.5)] #(40,1.25),
# location_type = 'test_mult_locations'

# from location_selection import training_locations10 as locations
# location_type = '10_random_locs'
# location_type = 'training_grid_small_test'

# from location_selection import training_locations100 as locations
# location_type = '100_random_locs'
# location_type = 'training_grid_large_test'

# from location_selection import training_locations1000 as locations
# location_type = '1000_random_locs'
# location_type = 'training_grid_very_large_test'

# -------- single
# locations = [(48.75,-12.25)]
# location_type = 'test_single'
# import numpy as np
# locations = np.round([(52.85,3.43)])
# location_type = 'mmc_1x1'

# single loc power test
# locations = [(48.75,-12.25)]
# location_type = 'training_grid_very_large_test_1x1'

# locations = [(52.85,3.43)]
# location_type = 'mmc'

# -------- 1x1
# from location_selection import training_locations10_1x1 as locations
# location_type = 'training_grid_small_test_1x1'

# from location_selection import training_locations100_1x1 as locations
# location_type = 'training_grid_large_test_1x1'

# from location_selection import training_locations1000_1x1 as locations
# location_type = 'training_grid_very_large_test_1x1'

if year_final_month < 12:
    location_type += 'up_to_month_{}'.format(year_final_month)

# --------------------------- DATASET INPUT ------------------------------
# Choose dataset
use_data_opts = ['ERA5', 'ERA5_1x1', 'DOWA', 'LIDAR']
use_data = use_data_opts[0]

# --------------------------- DOWA
# DOWA data contains years 2008 to 2017
DOWA_data_dir = "/cephfs/user/s6lathim/DOWA/"
# "/home/mark/WindData/DOWA/"  # '/media/mark/LaCie/DOWA/'

# TODO test multiple locations input DOWA

# --------------------------- ERA5
# General settings
if use_data == 'ERA5_1x1':
    era5_data_dir = '/cephfs/user/s6lathim/ERA5Data-redownload/'
    import numpy as np
    # TODO automatize from region
    all_lats = list(np.arange(65, 29, -1))  # 65 to 30
    all_lons = list(np.arange(-20, 21, 1))  # -20 to 20
    if len(locations) == 0:
        # Use all locations
        locations = [(lat, lon) for lat in all_lats for lon in all_lons]
    i_locations = [(all_lats.index(lat), all_lons.index(lon))
                   for lat, lon in locations]
else:
    era5_data_dir = '/cephfs/user/s6lathim/ERA5Data/'
    import numpy as np
    # TODO automatize from region
    all_lats = list(np.arange(65, 29.75, -.25))  # 65 to 30
    all_lons = list(np.arange(-20, 20.25, .25))  # -20 to 20
    if len(locations) == 0:
        # Use all locations
        locations = [(lat, lon) for lat in all_lats for lon in all_lons]
    i_locations = [(all_lats.index(lat), all_lons.index(lon))
                   for lat, lon in locations]


# ----------- ERA5 input format
era5_data_input_formats = ['monthly', 'loc_box', 'single_loc']
era5_data_input_format = era5_data_input_formats[2]

# ---- monthly and loc_box use:
surface_file_name_format = "{:d}_europe_{:d}_152.nc"
# Alternatives: 'sfc_{:d}_{:02d}.netcdf'
model_level_file_name_format = "{:d}_europe_{:d}_130_131_132_133_135.nc"
# Alternatives: 'ml_{:d}_{:02d}.netcdf'

# ---- single location files
# latitude_ds_file_name = era5_data_dir + 'loc_files/europe_{}_{}'.format(
#       start_year, final_year) + '_i_lat_{i_lat}_i_lon_{i_lon}.nc'
latitude_ds_file_name_idx = (era5_data_dir + 'cdo_single_loc/time_combined/' +
   'europe_2010_2020_130_131_132_133_135_{i_lat:05d}.nc')  # {i_lon:05d}.nc'
latitude_ds_file_name_idx_monthly = (era5_data_dir +
   'cdo_single_loc/{year}_europe_{month}_130_131_132_133_135.nc{i_lat:05d}.nc')

latitude_ds_file_name = (era5_data_dir + 'cdo_single_loc/time_combined/'
                         + 'loc-wise/europe_2010_2020_130_131_132_133_135'
                         + '_lat_{lat:.2f}_lon_{lon:.2f}.nc')

# Processing settings
read_model_level_up_to = 112


# --------------------------- OUTPUT -------------------------------------
# --------------------------- RESULT DIR
result_dir = "/cephfs/user/s6lathim/clustering_results/" + use_data + "/"

# --------------------------- FILE SUFFIX
if len(locations) == 1:
    lat, lon = locations[0]
    data_info = 'lat_{:2.2f}_lon_{:2.2f}_{}_{}_{}'.format(lat, lon,
                                                           use_data,
                                                           start_year,
                                                           final_year)
else:
    data_info = '_{}_{}_{}'.format(use_data, start_year, final_year)

# --------------------------- CLUSTERING OUTPUT
config_setting = '{}_clusters_{}'.format(n_clusters, location_type)
data_info = config_setting + data_info

file_name_cluster_profiles = (
    result_dir + 'cluster_wind_profile_shapes_{}.csv'.format(data_info))
file_name_freq_distr = (
    result_dir + 'freq_distribution_{}.pickle'.format(data_info))
file_name_cluster_labels = (
    result_dir + 'cluster_labels_{}.pickle'.format(data_info))
file_name_cluster_pipeline = (
    result_dir + 'cluster_pipeline_{}.pickle'.format(data_info))
cut_wind_speeds_file = (
    result_dir + 'cut_in_out_{}_estimate.csv'.format(data_info))

data_info_training = '{}_clusters_{}_{}_{}_{}'.format(n_clusters,
                                              location_type_training,
                                              use_data,
                                              start_year_training,
                                              final_year_training)
training_cluster_profiles = (
    result_dir + 'cluster_wind_profile_shapes_{}.csv'.format(
        data_info_training))
training_cluster_pipeline = (
    result_dir + 'cluster_pipeline_{}.pickle'.format(data_info_training))
training_cluster_labels = (
    result_dir + 'cluster_labels_{}.pickle'.format(data_info_training))
training_cut_wind_speeds_file = (
    result_dir + 'cut_in_out_{}_estimate.csv'.format(data_info_training))
training_refined_cut_wind_speeds_file = training_cut_wind_speeds_file.replace(
    'estimate','refined')
# Alternatives:
#  cut_wind_speeds_file = (
#  '/home/mark/Projects/quasi-steady-model-sandbox/wind_resource/'
#  'cut_in_out_8mmc.pickle')

# ----------------------------------------------------------------
# -------------------------------- VALIDATION - sample config
# ----------------------------------------------------------------
# PCA/ Clustering sample settings
validation_type_opts = ['full_training_full_test',
                        'cut_training_full_test',
                        'cut_training_cut_test']
validation_type = validation_type_opts[1]  # default: 1

# Height range settings
# DOWA height range
height_range = [10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160.,
                180., 200., 220., 250., 300., 500., 600.]
# Test linearized height range (ERA5 only)
# height_range = [70.,  100., 140., 170., 200., 240., 270., 300., 340., 370.,
#                400., 440., 470., 500., 540., 570., 600.]

height_range_name_opts = ['DOWA_height_range']  # , 'lin_height_range']
height_range_name = height_range_name_opts[0]  # default: 0

# Normalization settings preprocessing
do_normalize_data = True  # default: True

# Validation output directories
if do_normalize_data:
    result_dir_validation = (
        result_dir + validation_type + '/' + height_range_name + '/')
else:
    result_dir_validation = (
        result_dir + 'no_norm/' + validation_type + '/'
        + height_range_name + '/')

make_result_subdirs = True
