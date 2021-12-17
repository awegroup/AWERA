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

import yaml
import os
import numpy as np
from .location_selection import get_locations

class Config:
    # TODO include production config

    # TODO handle config imports - config['....']
    # TODO import in files? or give config to functions?
    # TODO import in some way that changes on runtime?

    # Handle config initialization from yaml file and runtime updating

    def __init__(self, init_dict=None, interpret=True):
        if init_dict is None:
            self.update_from_file()
        else:
            self.update(init_dict, interpret=interpret)

    def update(self, update_dict, interpret=True):
        for key, val in update_dict.items():
            if isinstance(val, dict):
                try:
                    is_Config = isinstance(getattr(self, key), Config)
                except AttributeError:
                    is_Config = False
                if is_Config:
                    getattr(self, key).update(val, interpret=False)
                else:
                    setattr(self, key, Config(init_dict=val, interpret=False))
            else:
                setattr(self, key, val)

        if interpret:
            self.interpret()

    def update_from_file(self, path_config_yaml=None):
        # Read configuration from config.yaml file
        # TODO try to read from local file first
        if path_config_yaml is None:
            # Read default configuration from package
            path_program_directory = os.path.dirname(__file__)
            path_config_yaml = os.path.join(
                path_program_directory, '', 'config.yaml')
            # TODO add confg directory -> add in ''
        with open(path_config_yaml, 'r') as f:
            initial_config = yaml.safe_load(f)
        self.update(initial_config)

    # Handle file naming, location selection, .. depending on config

    def interpret(self):
        # All locations and seletect number of locations
        # Range of all locations
        for key in ['lat', 'lon']:
            if (getattr(self.Data, key + '_range')[0] >
                    getattr(self.Data, key + '_range')[1]):
                setattr(self.Data, 'all_{}s'.format(key), list(np.arange(
                    getattr(self.Data, key + '_range')[0],
                    getattr(self.Data, key + '_range')[1]-self.Data.grid_size,
                    -self.Data.grid_size)))
            else:
                setattr(self.Data, 'all_{}s'.format(key), list(np.arange(
                    getattr(self.Data, key + '_range')[0],
                    getattr(self.Data, key + '_range')[1]+self.Data.grid_size,
                    self.Data.grid_size)))

        # --------------------------- SELECT LOCATIONS
        # Set locations file dir
        setattr(self.IO, 'locations',
                self.IO.result_dir + getattr(self.IO.format, 'locations'))

        # Uniformly draw locations
        setattr(self.Data, 'locations', get_locations(
            self.IO.locations,
            self.Data.location_type,
            self.Data.n_locs,
            self.Data.lat_range,
            self.Data.lon_range,
            self.Data.grid_size
            ))
        # Get loction indices w.r.t. to full dataset
        setattr(self.Data, 'i_locations', [(self.Data.all_lats.index(lat),
                                            self.Data.all_lons.index(lon))
                                           for lat, lon in self.Data.locations]
                )
        # Check for individual training data selection, otherwise set to Data
        for key in self.Clustering.training.__dict__:
            if getattr(self.Clustering.training, key) is None:
                # Fill None Values in training with Data values
                setattr(self.Clustering.training, key, getattr(self.Data, key))
        # Evaluate which locations to use for the clustering training
        if self.Clustering.training.n_locs == self.Data.n_locs \
                and self.Clustering.training.location_type == \
                self.Data.location_type:
            # Same locations as Data
            setattr(self.Clustering.training, 'locations',
                    self.Data.locations)
        else:
            if self.Clustering.training.n_locs != self.Data.n_locs \
                    and self.Clustering.training.location_type == \
                    self.Data.location_type:
                # TODO only log? predefined, no problem?
                print('WARNING: Same location type but different n given for '
                      'training - training locations are also uniformly '
                      'selected, overlapping possible')
            setattr(self.Clustering.training, 'locations',
                    get_locations(self.IO.locations,
                                  self.Clustering.training.location_type,
                                  self.Clustering.training.n_locs,
                                  self.Data.lat_range,
                                  self.Data.lon_range,
                                  self.Data.grid_size
                                  ))
        # Set correct n_locs
        setattr(self.Data, 'n_locs', len(self.Data.locations))
        setattr(self.Clustering.training,
                'n_locs',
                len(self.Clustering.training.locations))

        # --------------------------- FILE SUFFIX
        if self.Data.year_final_month < 12:
            month_tag = '_up_to_month_{}'.format(
                self.Data.year_final_month)
        else:
            month_tag = ''
        n_max_locs = len(self.Data.all_lats)*len(self.Data.all_lons)
        # TODO better then this?
        if self.Data.n_locs in [-1, n_max_locs]:
            n_locs_tag = 'all'
        else:
            n_locs_tag = self.Data.n_locs
        data_info = '{}_clusters_{}_n_locs_{}{}_{}_{}_{}'.format(
            self.Clustering.n_clusters,
            self.Data.location_type,
            n_locs_tag,
            month_tag, self.Data.use_data,
            self.Data.start_year,
            self.Data.final_year
            )
        setattr(self.Data, 'data_info', data_info)
        # TODO remove!
        n_max_locs = 525
        if self.Clustering.training.n_locs in [-1, n_max_locs]:
            n_locs_tag = 'all'
        else:
            n_locs_tag = self.Clustering.training.n_locs
        data_info_training = \
            '{}_clusters_{}_n_locs_{}{}_{}_{}_{}'.format(
                self.Clustering.n_clusters,
                self.Clustering.training.location_type,
                n_locs_tag, month_tag,
                self.Data.use_data,
                self.Clustering.training.start_year,
                self.Clustering.training.final_year
                )
        setattr(self.Clustering.training, 'data_info', data_info)
        # --------------------------- DIR + FILE and SUFFIX FORMATTING
        for key in self.IO.format.__dict__:
            if key not in ['locations']:
                setattr(self.IO, key,
                        self.IO.result_dir + getattr(self.IO.format, key))
            if key in ['freq_distr', 'cluster_labels']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info=data_info,
                            data_info_training=data_info_training))
                if key == 'cluster_labels':
                    setattr(self.IO, 'training_' + key,
                            self.IO.result_dir +
                            getattr(self.IO.format, key).format(
                                data_info=data_info_training,
                                data_info_training=data_info_training))
                    # TODO do we want the trainng labels file?
            elif key in ['plot_output']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info=data_info))
                setattr(self.IO, 'training_' + key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info=data_info_training))
                # TODO is this always data or also training?
            elif key not in ['locations']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info_training=data_info_training))
                # TODO cluster profiles, pipeline always training

        for key in ['latitude_ds_file_name',
                    'model_level_file_name_format',
                    'surface_file_name_format']:
            setattr(self.Data, key,
                    self.Data.era5_data_dir + getattr(self.Data.format, key))

    # Output Configuration

    def dictify(self):
        # TODO print locations only if specifically asked
        # TODO why is this copy necessary..?
        out = self.__dict__.copy()
        for key, val in out.items():
            if isinstance(val, Config):
                out[key] = val.dictify()
        return out

    def pop_locations(self, d=None):
        if d is None:
            d = self.dictify()
        res = d.copy()
        for key, val in d.items():
            if isinstance(val, dict):
                res[key] = self.pop_locations(d=val)
            else:
                # Drop long location lists and all_lats, all_lons
                if key in ['i_locations', 'locations', 'all_lats', 'all_lons']:
                    res.pop(key, None)
        return res

    def print_full(self):
        print(self.dictify())

    def __str__(self):
        return str(self.pop_locations())


#config = Config()

# TODO fix validation - fix config validation
#                     - own config class? own yaml file
# ----------------------------------------------------------------
# -------------------------------- VALIDATION - sample config
# ----------------------------------------------------------------
# PCA/ Clustering sample settings
#validation_type_opts = ['full_training_full_test',
#                        #'cut_training_full_test',
#                        #'cut_training_cut_test']
#validation_type = validation_type_opts[1]  # default: 1

# Height range settings
# DOWA height range
#height_range = [10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160.,
                #180., 200., 220., 250., 300., 500., 600.]
# Test linearized height range (ERA5 only)
# height_range = [70.,  100., 140., 170., 200., 240., 270., 300., 340., 370.,
#                400., 440., 470., 500., 540., 570., 600.]

#height_range_name_opts = ['DOWA_height_range']  # , 'lin_height_range']
#height_range_name = height_range_name_opts[0]  # default: 0

# Normalization settings preprocessing
#do_normalize_data = True  # default: True

# Validation output directories
#if do_normalize_data:
#    result_dir_validation = (
#        config.IO.result_dir + validation_type + '/' + height_range_name + '/')
#else:
#    result_dir_validation = (
#        config.IO.result_dir + 'no_norm/' + validation_type + '/'
#        + height_range_name + '/')

#make_result_subdirs = True
