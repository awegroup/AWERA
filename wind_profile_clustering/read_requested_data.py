#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import sys
from os.path import join as path_join

try:
    from .era5_ml_height_calc import compute_level_heights
except ImportError:
    from era5_ml_height_calc import compute_level_heights

from config_clustering import use_data, start_year, final_year, year_final_month,\
                   DOWA_data_dir, locations, \
                   era5_data_dir, model_level_file_name_format, latitude_ds_file_name, era5_data_input_format,\
                   surface_file_name_format, read_model_level_up_to, height_range

from config_clustering import all_lats, all_lons # i_locations
from config_clustering import latitude_ds_file_name_idx, latitude_ds_file_name_idx_monthly
# FIXME what of this is still necessary?

import dask
# only as many threads as requested CPUs | only one to be requested, more threads don't seem to be used
dask.config.set(scheduler='synchronous')


def read_raw_data(start_year, final_year, year_final_month=12,
                  sel_sample_ids=[], lat0=50, lon0=0):
    """"Read ERA5 wind data for adjacent years.

    Args:
        start_year (int): Read data starting from this year.
        final_year (int): Read data up to this year.

    Returns:
        tuple of Dataset, ndarray, ndarray, ndarray, and ndarray: Tuple containing reading object of multiple wind
        data (netCDF) files, longitudes of grid, latitudes of grid, model level numbers, and timestamps in hours since
        1900-01-01 00:00:0.0.

    """
    if era5_data_input_format == 'loc_box':
        # match locations to loc-boxes? faster? TODO
        ds = read_ds_loc_boxes(start_year, final_year, year_final_month=12, n_boxes=21)
    elif era5_data_input_format == 'single_loc':
        ds = read_ds_single_loc_files(lat0,lon0)
    elif era5_data_input_format == 'monthly':
        # Construct the list of input NetCDF files
        ml_files = []
        sfc_files = []
        for y in range(start_year, final_year+1):
            for m in range(1, year_final_month+1):
                ml_files.append(path_join(era5_data_dir, model_level_file_name_format.format(y, m)))
                sfc_files.append(path_join(era5_data_dir, surface_file_name_format.format(y, m)))
        # Load the data from the NetCDF files.
        ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=True)
    else:
        print('Wrong input data format: {}'.format(era5_data_input_format))
    if len(sel_sample_ids) > 0:
        ds = ds.isel(time=sel_sample_ids)

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    levels = ds['level'].values.astype(int)  # Model level numbers.
    hours = ds['time'].values

    dlevels = np.diff(levels)
    if not (np.all(dlevels == 1) and levels[-1] == 137):
        i_highest_level = len(levels) - np.argmax(dlevels[::-1] > 1) - 1
        print("Not all the downloaded model levels are consecutive. Only model levels up to {} are evaluated."
              .format(levels[i_highest_level]))
        levels = levels[i_highest_level:]
    else:
        i_highest_level = 0

    return ds, lons, lats, levels, hours, i_highest_level


def read_ds_loc_boxes(start_year, final_year, year_final_month=12, n_boxes=21, sel_sample_ids=[]):
    """"Read ERA5 wind data for adjacent years.

    Args:
        start_year (int): Read data starting from this year.
        final_year (int): Read data up to this year.

    Returns:
        Dataset: Reading object of multiple wind data (netCDF) files

    """
    # Construct the list of input NetCDF files
    ml_files = []
    sfc_files = []
    ml_loc_box_file_name = 'loc-box/' + model_level_file_name_format + '000{:02d}.nc'
    for y in range(start_year, final_year+1):
        for m in range(1, year_final_month+1):
            for i_box in range(n_boxes):
                ml_files.append(path_join(era5_data_dir, ml_loc_box_file_name.format(y, m, i_box)))
            sfc_files.append(path_join(era5_data_dir, surface_file_name_format.format(y, m)))
    # Load the data from the NetCDF files.
    ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=True)
    if len(sel_sample_ids) > 0:
        ds = ds.isel(time=sel_sample_ids)
    return ds


def read_ds_single_loc_files(lat, lon, by_index=False, time_combined=True, sel_sample_ids=[]):
    """"Read ERA5 wind data from location wise files.

    Returns:
        Dataset: Reading object of multiple wind data (netCDF) files

    """
    i_lat, i_lon = [all_lats.index(lat), all_lons.index(lon)]
    #Read only first location to the firsts ds
    if by_index:
        print('i_loc: ', i_lat, i_lon)
        # Load the data from the NetCDF files.

        #ds_ml = xr.open_dataset(latitude_ds_file_name_idx.format(i_lat=i_lat, i_lon=i_lon), decode_times=True)
        if time_combined:
            ds_ml = xr.open_dataset(latitude_ds_file_name_idx.format(i_lat=i_lat), decode_times=True)
            ds_ml = ds_ml.isel(longitude=[i_lon])
        else:
            ml_files = []
            for y in range(start_year, final_year+1):
                for m in range(1, year_final_month+1):
                    ml_files.append(path_join(era5_data_dir, latitude_ds_file_name_idx_monthly.format(i_lat=i_lat, year=y, month=m)))
            ds_ml = xr.open_mfdataset(ml_files, decode_times=True)
            ds_ml = ds_ml.isel(longitude=[i_lon])

    else:
        ds_ml = xr.open_dataset(latitude_ds_file_name.format(lat=lat, lon=lon), decode_times=True)

    ds_ml = ds_ml.sel(time=slice(str(start_year), str(final_year)))

    # read surface pressure files
    sfc_files = []
    for y in range(start_year, final_year+1):
        for m in range(1, year_final_month+1):
            sfc_files.append(path_join(era5_data_dir, surface_file_name_format.format(y, m)))
    # Load the data from the NetCDF files.
    ds_sfc = xr.open_mfdataset(sfc_files, decode_times=True)
    ds_sfc = ds_sfc.sel(latitude=[lat], longitude=[lon])

    # Test matching lat/lon representations
    if ds_sfc.coords['longitude'].values != ds_ml.coords['longitude'].values:
        ds_ml.coords['longitude'] = ds_ml.coords['longitude'] - 360
        if ds_sfc.coords['longitude'].values != ds_ml.coords['longitude'].values:
            raise ValueError('Mismatching longitudes')
    if ds_sfc.coords['latitude'].values != ds_ml.coords['latitude'].values:
        ds_ml.coords['latitude'] = ds_ml.coords['latitude'] - 360
        if ds_sfc.coords['longitude'].values != ds_ml.coords['longitude'].values:
            raise ValueError('Mismatching latitudes')

    #print(ds_sfc)
    #print(ds_ml)
    ds = xr.merge([ds_ml, ds_sfc])
    if len(sel_sample_ids) > 0:
        ds = ds.isel(time=sel_sample_ids)
    #print(ds)
    return ds


def eval_single_loc_era5_input(sel_sample_ids, i_highest_level, levels, n_per_loc, heights_of_interest, loc_i_loc):
    # TODO improve arguments!
    lat, lon,  i_lat, i_lon = loc_i_loc
    if era5_data_input_format == 'single_loc':
        ds = read_ds_single_loc_files(lat, lon, sel_sample_ids=sel_sample_ids)

    # Extract wind data for single location
    v_levels_east = ds['u'][:, i_highest_level:, i_lat, i_lon].values
    v_levels_north = ds['v'][:, i_highest_level:, i_lat, i_lon].values

    t_levels = ds['t'][:, i_highest_level:, i_lat, i_lon].values  # TODO test -- better to call values later? or all together at beginning?
    q_levels = ds['q'][:, i_highest_level:, i_lat, i_lon].values

    try:
        surface_pressure = ds.variables['sp'][:, i_lat, i_lon].values
    except KeyError:
        surface_pressure = np.exp(ds.variables['lnsp'][:, i_lat, i_lon].values)


    # Calculate model level height
    level_heights, density_levels = compute_level_heights(levels,
                                                          surface_pressure,
                                                          t_levels,
                                                          q_levels)
    # Determine wind at altitudes of interest by
    # means of interpolating the raw wind data.

    # Interpolation results array.
    v_req_alt_east_loc = np.zeros((n_per_loc, len(heights_of_interest)))
    v_req_alt_north_loc = np.zeros((n_per_loc, len(heights_of_interest)))

    for i_hr in range(n_per_loc):
        if not np.all(level_heights[i_hr, 0] > heights_of_interest):
            raise ValueError("Requested height ({:.2f} m) is higher than \
                             height of highest model level."
                             .format(level_heights[i_hr, 0]))
        v_req_alt_east_loc[i_hr, :] = np.interp(heights_of_interest,
                                                level_heights[i_hr, ::-1],
                                                v_levels_east[i_hr, ::-1])
        v_req_alt_north_loc[i_hr, :] = np.interp(heights_of_interest,
                                                 level_heights[i_hr, ::-1],
                                                 v_levels_north[i_hr, ::-1])
    #print('Location: ', lat, lon, ' read in.')
    return(v_req_alt_east_loc, v_req_alt_north_loc)


def get_wind_data_era5(heights_of_interest,
                       locations=[(40, 1)],
                       start_year=2010, final_year=2010, max_level=112,
                       era5_data_input_format='monthly', sel_sample_ids=[],
                       parallel=False):
    lat, lon = locations[0]
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(
        start_year, final_year, year_final_month=year_final_month,
        sel_sample_ids=sel_sample_ids, lat0=lat, lon0=lon)
    n_per_loc = len(hours)

    if era5_data_input_format != 'single_loc':
        # Convert lat/lon lists to indices
        lats, lons = (list(lats), list(lons))
        i_locs = [(lats.index(lat), lons.index(lon)) for lat, lon in locations]

    else:
        # print('Processing single location input')
        i_locs = [(0, 0) for lat, lon in locations]

    v_req_alt_east = np.zeros((n_per_loc*len(locations),
                               len(heights_of_interest)))
    v_req_alt_north = np.zeros((n_per_loc*len(locations),
                                len(heights_of_interest)))
    n_cores = 23  # TODO make n optional
    if parallel:
        from multiprocessing import Pool
        from tqdm import tqdm
        loc_i_loc_combinations = [(locations[i][0], locations[i][1],
                                   i_loc[0], i_loc[1])
                                  for i, i_loc in enumerate(i_locs)]
        import functools
        funct = functools.partial(eval_single_loc_era5_input, sel_sample_ids,
                                  i_highest_level, levels, n_per_loc,
                                  heights_of_interest)
        with Pool(n_cores) as p:
            results = list(tqdm(p.imap(funct, loc_i_loc_combinations),
                                total=len(loc_i_loc_combinations),
                                file=sys.stdout)) # TODO stdout optional
            # TODO is this more RAM intensive?
            for i, val in enumerate(results):
                v_req_alt_east_loc, v_req_alt_north_loc = val
                v_req_alt_east[n_per_loc*i:n_per_loc*(i+1), :] = \
                    v_req_alt_east_loc
                v_req_alt_north[n_per_loc*i:n_per_loc*(i+1), :] = \
                    v_req_alt_north_loc
    else:
        # Not parallelized version:
        for i, i_loc in enumerate(i_locs):
            i_lat, i_lon = i_loc
            lat, lon = locations[i]
            v_req_alt_east_loc, v_req_alt_north_loc = \
                eval_single_loc_era5_input(
                    sel_sample_ids, i_highest_level,
                    levels, n_per_loc, heights_of_interest,
                    (lat, lon, i_lat, i_lon))

            v_req_alt_east[n_per_loc*i:n_per_loc*(i+1), :] = v_req_alt_east_loc
            v_req_alt_north[n_per_loc*i:n_per_loc*(i+1), :] = \
                v_req_alt_north_loc

        # if era5_data_input_format == 'single_loc':
        # Close dataset - free resources? #TODO
        #    ds.close()
    # TODO This could get too large for a large number of locations
    # - better use an xarray structure here?
    wind_data = {
        'wind_speed_east': v_req_alt_east,
        'wind_speed_north': v_req_alt_north,
        'n_samples': n_per_loc*len(i_locs),
        'n_samples_per_loc': n_per_loc,
        'datetime': ds['time'].values,
        'altitude': heights_of_interest,
        'years': (start_year, final_year),
        'locations': locations,
        }
    ds.close()  # Close the input NetCDF file.
    return wind_data


def get_wind_data(sel_sample_ids=[], locs=[], parallel=False):
    # TODO add single sample selection for all data types
    if len(locs) == 0:
        # Use all configuration locations
        locs = locations

    if use_data == 'DOWA':
        import os
        #HDF5 library has been updated (1.10.1) (netcdf uses HDF5 under the hood)
        #file system does not support the file locking that the HDF5 library uses.
        #In order to read your hdf5 or netcdf files, you need set this environment variable :
        os.environ["HDF5_USE_FILE_LOCKING"]="FALSE" # check - is this needed? if yes - where set, needed for era5? FIX
        from read_data.dowa import read_data
        wind_data = read_data({'mult_coords':locs}, DOWA_data_dir)

        # Use start_year to final_year data only
        hours = wind_data['datetime']
        start_date = np.datetime64('{}-01-01T00:00:00.000000000'.format(start_year))
        end_date = np.datetime64('{}-01-01T00:00:00.000000000'.format(final_year+1))

        start_idx = list(hours).index(start_date)
        end_idx = list(hours).index(end_date)
        data_range = range(start_idx, end_idx + 1)

        for key in ['wind_speed_east', 'wind_speed_north', 'datetime']:
            wind_data[key] = wind_data[key][data_range]
        wind_data['n_samples'] = len(data_range)
        wind_data['years'] = (start_year, final_year)

        print(len(hours))
        print(len(wind_data['wind_speed_east']), wind_data['n_samples'])
    elif use_data == 'LIDAR':
        from read_data.fgw_lidar import read_data
        wind_data = read_data()

    elif use_data in ['ERA5', 'ERA5_1x1']:
        wind_data = get_wind_data_era5(
            height_range, locations=locs, start_year=start_year,
            final_year=final_year, max_level=read_model_level_up_to,
            era5_data_input_format=era5_data_input_format,
            sel_sample_ids=sel_sample_ids, parallel=parallel)
    else:
        raise ValueError("Wrong data type specified: {} - no option to read data is executed".format(use_data))

    return wind_data

