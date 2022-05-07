#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import sys
from os.path import join as path_join

from .era5_ml_height_calc import compute_level_heights


# FIXME what of this is still necessary?


import dask
# only as many threads as requested CPUs | only one to be requested,
# more threads don't seem to be used
# TODO better option than synchronous?
dask.config.set(scheduler='synchronous')


def read_raw_data(data_config, sel_sample_ids=[], lat0=50, lon0=0):
    """"Read ERA5 wind data for adjacent years.

    Args:
        start_year (int): Read data starting from this year.
        final_year (int): Read data up to this year.

    Returns:
        tuple of Dataset, ndarray, ndarray, ndarray, and ndarray:
            Tuple containing reading object of multiple wind
        data (netCDF) files, longitudes of grid, latitudes of grid,
        model level numbers, and timestamps in hours since
        1900-01-01 00:00:0.0.

    """
    if data_config.era5_data_input_format == 'single_loc':
        ds = read_ds_single_loc_files(data_config, lat0, lon0)
    elif data_config.era5_data_input_format == 'monthly':
        # Construct the list of input NetCDF files
        ml_files = []
        sfc_files = []
        for y in range(data_config.start_year, data_config.final_year+1):
            for m in range(1, data_config.year_final_month+1):
                ml_files.append(
                    data_config.model_level_file_name_format.format(y, m))
                sfc_files.append(
                    data_config.surface_file_name_format.format(y, m))
        # Load the data from the NetCDF files.
        ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=True)
    else:
        print('Wrong input data format: {}'.format(
            data_config.era5_data_input_format))
    if len(sel_sample_ids) > 0:
        ds = ds.isel(time=sel_sample_ids)

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    levels = ds['level'].values.astype(int)  # Model level numbers.
    hours = ds['time'].values

    dlevels = np.diff(levels)
    if not (np.all(dlevels == 1) and levels[-1] == 137):
        i_highest_level = len(levels) - np.argmax(dlevels[::-1] > 1) - 1
        print("Not all the downloaded model levels are consecutive."
              " Only model levels up to {} are evaluated."
              .format(levels[i_highest_level]))
        levels = levels[i_highest_level:]
    else:
        i_highest_level = 0

    return ds, lons, lats, levels, hours, i_highest_level


def read_ds_single_loc_files(data_config,
                             lat, lon,
                             by_index=False,
                             time_combined=True,
                             sel_sample_ids=[]):
    """"Read ERA5 wind data from location wise files.

    Returns:
        Dataset: Reading object of multiple wind data (netCDF) files

    """
    ds_ml = xr.open_dataset(data_config.latitude_ds_file_name.format(
        lat=lat, lon=lon), decode_times=True)

    ds_ml = ds_ml.sel(time=slice(str(data_config.start_year),
                                 str(data_config.final_year)))

    # Read surface pressure files
    sfc_files = []
    for y in range(data_config.start_year, data_config.final_year+1):
        for m in range(1, data_config.year_final_month+1):
            sfc_files.append(
                data_config.surface_file_name_format.format(y, m))
    # Load the data from the NetCDF files.
    ds_sfc = xr.open_mfdataset(sfc_files, decode_times=True)
    ds_sfc = ds_sfc.sel(latitude=[lat], longitude=[lon])

    # Test matching lat/lon representations
    if ds_sfc.coords['longitude'].values != ds_ml.coords['longitude'].values:
        ds_ml.coords['longitude'] = ds_ml.coords['longitude'] - 360
        if (ds_sfc.coords['longitude'].values !=
                ds_ml.coords['longitude'].values):
            raise ValueError('Mismatching longitudes')
    if ds_sfc.coords['latitude'].values != ds_ml.coords['latitude'].values:
        ds_ml.coords['latitude'] = ds_ml.coords['latitude'] - 360
        if (ds_sfc.coords['longitude'].values !=
                ds_ml.coords['longitude'].values):
            raise ValueError('Mismatching latitudes')

    ds = xr.merge([ds_ml, ds_sfc])
    if len(sel_sample_ids) > 0:
        ds = ds.isel(time=sel_sample_ids)
    return ds


def eval_single_loc_era5_input(data_config,
                               sel_sample_ids,
                               i_highest_level,
                               levels,
                               n_per_loc,
                               loc_i_loc,
                               ds=None,
                               use_memmap=False):
    # TODO improve arguments!
    lat, lon,  i_lat, i_lon, i = loc_i_loc

    if data_config.era5_data_input_format == 'single_loc' or ds is None:
        # For single location data files, always reaad next file
        ds = read_ds_single_loc_files(data_config,
                                      lat, lon,
                                      sel_sample_ids=sel_sample_ids)

    # Extract wind data for single location
    v_levels_east = ds['u'][:, i_highest_level:, i_lat, i_lon].values
    v_levels_north = ds['v'][:, i_highest_level:, i_lat, i_lon].values

    t_levels = ds['t'][:, i_highest_level:, i_lat, i_lon].values
    # TODO test -- better to call values later? or all together at beginning?
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
    v_req_alt_east_loc = np.zeros((n_per_loc, len(data_config.height_range)))
    v_req_alt_north_loc = np.zeros((n_per_loc, len(data_config.height_range)))

    # same = 0
    for i_hr in range(n_per_loc):
        if not np.all(level_heights[i_hr, 0] > data_config.height_range):
            raise ValueError("Requested height ({:.2f} m) is higher than \
                             height of highest model level."
                             .format(level_heights[i_hr, 0]))
        v_req_alt_east_loc[i_hr, :] = np.interp(data_config.height_range,
                                                level_heights[i_hr, ::-1],
                                                v_levels_east[i_hr, ::-1])
        v_req_alt_north_loc[i_hr, :] = np.interp(data_config.height_range,
                                                 level_heights[i_hr, ::-1],
                                                 v_levels_north[i_hr, ::-1])
        # Sanity check height range oversampling
        # same_hr = sum(np.diff(np.round(np.interp(data_config.height_range,
        #                       level_heights[i_hr, ::-1],
        #                       np.arange(level_heights.shape[1])))) == 0)
    #     same += same_hr
    # print('Height Level Ids: ',
    #       np.round(np.interp(data_config.height_range,
    #                          level_heights[i_hr, ::-1],
    #                          np.arange(level_heights.shape[1]))))
    # print('Same level data used {} times.'.format(same))
    if use_memmap:
        v_east = np.memmap('tmp/v_east.memmap', dtype='float64', mode='r+',
                           shape=(n_per_loc, len(data_config.height_range)),
                           offset=n_per_loc*i*len(data_config.height_range)
                           * int(64/8))
        v_east[:, :] = v_req_alt_east_loc
        del v_east
        v_north = np.memmap('tmp/v_north.memmap', dtype='float64', mode='r+',
                            shape=(n_per_loc, len(data_config.height_range)),
                            offset=n_per_loc*i*len(data_config.height_range)
                            * int(64/8))
        v_north[:, :] = v_req_alt_north_loc
        del v_north
        return 0
    else:
        return(v_req_alt_east_loc, v_req_alt_north_loc)


def get_wind_data_era5(config,
                       locations=[(40, 1)],
                       sel_sample_ids=[]):
    lat, lon = locations[0]
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(
        config.Data, sel_sample_ids=sel_sample_ids, lat0=lat, lon0=lon)
    n_per_loc = len(hours)

    if config.Data.era5_data_input_format != 'single_loc':
        # Convert lat/lon lists to indices
        lats, lons = (list(lats), list(lons))
        i_locs = [(lats.index(lat), lons.index(lon)) for lat, lon in locations]

    else:
        i_locs = [(0, 0) for lat, lon in locations]

    if config.General.use_memmap:
        v_req_alt_east = np.memmap('tmp/v_east.memmap', dtype='float64',
                                   mode='w+',
                                   shape=(n_per_loc*len(locations),
                                          len(config.Data.height_range)))
        v_req_alt_north = np.memmap('tmp/v_north.memmap', dtype='float64',
                                    mode='w+',
                                    shape=(n_per_loc*len(locations),
                                           len(config.Data.height_range)))
    else:
        v_req_alt_east = np.zeros((n_per_loc*len(locations),
                                   len(config.Data.height_range)))
        v_req_alt_north = np.zeros((n_per_loc*len(locations),
                                    len(config.Data.height_range)))

    if config.Processing.parallel:
        # TODO import not here
        from multiprocessing import Pool
        from tqdm import tqdm
        loc_i_loc_combinations = [(locations[i][0], locations[i][1],
                                   i_loc[0], i_loc[1], i)
                                  for i, i_loc in enumerate(i_locs)]
        import functools
        funct = functools.partial(eval_single_loc_era5_input,
                                  config.Data, sel_sample_ids,
                                  i_highest_level, levels, n_per_loc,
                                  ds=ds,
                                  use_memmap=config.General.use_memmap)
        with Pool(config.Processing.n_cores) as p:
            if config.Processing.progress_out == 'stdout':
                file = sys.stdout
            else:
                file = sys.stderr
            results = list(tqdm(p.imap(funct, loc_i_loc_combinations),
                                total=len(loc_i_loc_combinations),
                                file=file))
            if not config.General.use_memmap:
                for i, val in enumerate(results):
                    v_req_alt_east_loc, v_req_alt_north_loc = val
                    v_req_alt_east[n_per_loc*i:n_per_loc*(i+1), :] = \
                        v_req_alt_east_loc
                    v_req_alt_north[n_per_loc*i:n_per_loc*(i+1), :] = \
                        v_req_alt_north_loc
    else:
        # Not parallelized version:
        for i, i_loc in enumerate(i_locs):
            # TODO add progress bar
            i_lat, i_lon = i_loc
            lat, lon = locations[i]
            v_req_alt_east_loc, v_req_alt_north_loc = \
                eval_single_loc_era5_input(
                    config.Data,
                    sel_sample_ids, i_highest_level,
                    levels, n_per_loc,
                    (lat, lon, i_lat, i_lon, i),
                    ds=ds)
            # TODO is this even xarray anymore?
            # check efficiency numpy, pandas, xarray
            v_req_alt_east[n_per_loc*i:n_per_loc*(i+1), :] = v_req_alt_east_loc
            v_req_alt_north[n_per_loc*i:n_per_loc*(i+1), :] = \
                v_req_alt_north_loc

        # if era5_data_input_format == 'single_loc':
        # Close dataset - free resources? #TODO
        #    ds.close()
    # TODO This could get too large for a large number of locations
    # - better use an xarray/ more efficient data structure here?
    if config.General.use_memmap:
        del v_req_alt_east
        del v_req_alt_north
        v_req_alt_east = np.memmap('tmp/v_east.memmap', dtype='float64',
                                   mode='r',
                                   shape=(n_per_loc*len(locations),
                                          len(config.Data.height_range)))
        v_req_alt_north = np.memmap('tmp/v_north.memmap', dtype='float64',
                                    mode='r',
                                    shape=(n_per_loc*len(locations),
                                           len(config.Data.height_range)))


    wind_data = {
        'wind_speed_east': v_req_alt_east,
        'wind_speed_north': v_req_alt_north,
        'n_samples': n_per_loc*len(i_locs),
        'n_samples_per_loc': n_per_loc,
        'datetime': ds['time'].values,
        'altitude': config.Data.height_range,
        'years': (config.Data.start_year, config.Data.final_year),
        'locations': locations,
        }
    ds.close()  # Close the input NetCDF file.
    return wind_data


def get_wind_data(config, sel_sample_ids=[], locs=[]):
    # TODO add single sample selection for all data types
    if len(locs) == 0:
        # Use all configuration locations
        locs = config.Data.locations

    if config.Data.use_data == 'DOWA':
        import os
        # HDF5 library has been updated (1.10.1)
        # (netcdf uses HDF5 under the hood) file system does not support
        # the file locking that the HDF5 library uses.
        # In order to read your hdf5 or netcdf files,
        # you need set this environment variable :
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # TODO check - is this still needed?
        # if yes - where set, needed for era5? FIX
        from .read_data.dowa import read_data
        wind_data = read_data({'mult_coords': locs}, config.Data.DOWA_data_dir)

        # Use start_year to final_year data only
        hours = wind_data['datetime']
        start_date = np.datetime64(
            '{}-01-01T00:00:00.000000000'.format(config.Data.start_year))
        end_date = np.datetime64(
            '{}-01-01T00:00:00.000000000'.format(config.Data.final_year+1))

        start_idx = list(hours).index(start_date)
        end_idx = list(hours).index(end_date)
        data_range = range(start_idx, end_idx + 1)

        for key in ['wind_speed_east', 'wind_speed_north', 'datetime']:
            wind_data[key] = wind_data[key][data_range]
        wind_data['n_samples'] = len(data_range)
        wind_data['years'] = (config.Data.start_year, config.Data.final_year)
        wind_data['locations'] = locs
        wind_data['n_samples_per_loc'] = wind_data['n_samples']/len(locs)
        print(len(hours))
        print(len(wind_data['wind_speed_east']), wind_data['n_samples'])

    elif config.Data.use_data == 'LIDAR':
        from read_data.fgw_lidar import read_data
        # FIXME config is not included here?
        wind_data = read_data()

    elif config.Data.use_data in ['ERA5', 'ERA5_1x1']:
        wind_data = get_wind_data_era5(config,
                                       locations=locs,
                                       sel_sample_ids=sel_sample_ids)
    else:
        raise ValueError("Wrong data type specified: "
                         "{} - no option to read data is executed".format(
                             config.Data.use_data))

    return wind_data
