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
    # TODO include production config (?)

    def __init__(self, init_dict=None, interpret=True):
        # Handle config initialization from yaml file and runtime updating
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

    def update_from_file(self, path_config_yaml=None,
                         config_file='config.yaml'):
        # Read configuration from config.yaml file
        if path_config_yaml is None:
            # Read default configuration from package
            path_program_directory = os.path.dirname(__file__)
            path_config_yaml = os.path.join(
                path_program_directory, '', config_file)
        with open(path_config_yaml, 'r') as f:
            initial_config = yaml.safe_load(f)
        self.update(initial_config)

    def interpret(self):
        # Handle file naming, location selection, .. depending on config
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
                      'training - if not predefined, training locations are '
                      'also uniformly selected, overlapping possible.')
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

        if self.Data.n_locs in [-1, n_max_locs]:
            n_locs_tag = 'all'
        else:
            n_locs_tag = self.Data.n_locs
        data_info = '{}_n_locs_{}{}_{}_{}_{}'.format(
            self.Data.location_type,
            n_locs_tag,
            month_tag, self.Data.use_data,
            self.Data.start_year,
            self.Data.final_year
            )
        setattr(self.Data, 'data_info', data_info)

        if self.Clustering.training.n_locs in [-1, n_max_locs]:
            n_locs_tag = 'all'
        else:
            n_locs_tag = self.Clustering.training.n_locs
        data_info_training = \
            '{}_n_locs_{}{}_{}_{}_{}'.format(
                self.Clustering.training.location_type,
                n_locs_tag, month_tag,
                self.Data.use_data,
                self.Clustering.training.start_year,
                self.Clustering.training.final_year
                )
        setattr(self.Clustering.training, 'data_info', data_info_training)

        settings_info = '{}_clusters_{}_pcs{{}}'.format(
            self.Clustering.n_clusters,
            self.Clustering.n_pcs,
            )
        setattr(self.Clustering.training, 'settings_info', settings_info)

        try:
            norm = self.config.Clustering.do_normalize_data
            if norm:
                norm_tag = '_do_norm{}'
            else:
                norm_tag = '_no_norm{}'
            settings_info = settings_info.format(norm_tag)
        except AttributeError:
            pass
        try:
            train_cut = self.Clustering.Validation_type.training
            test_cut = self.Clustering.Validation_type.testing
            if train_cut == 'cut' and test_cut == 'full':
                # Default settings, no extra tagging
                pass
            else:
                cut_tag = '_{}_train_{}_data{{}}'.format(train_cut, test_cut)
                settings_info = settings_info.format(cut_tag)
        except AttributeError:
            pass
        settings_info = settings_info.format('')
        setattr(self.Clustering.training, 'settings_info', settings_info)

        # --------------------------- DIR + FILE and SUFFIX FORMATTING
        # TODO optimize this....
        for key in self.IO.format.__dict__:
            if key not in ['locations']:
                setattr(self.IO, key,
                        self.IO.result_dir + getattr(self.IO.format, key))

            if key in ['freq_distr', 'labels']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info=data_info,
                            data_info_training=data_info_training,
                            settings_info=settings_info))
                if key == 'labels':
                    setattr(self.IO, 'training_' + key,
                            self.IO.result_dir +
                            getattr(self.IO.format, key).format(
                                data_info=data_info_training,
                                data_info_training=data_info_training,
                                settings_info=settings_info))
            elif key in ['plot_output']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info=data_info,
                            data_info_training=data_info_training,
                            settings_info=settings_info))
            elif key in ['training_plot_output']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info_training=data_info_training,
                            settings_info=settings_info))
            elif key in ['cluster_validation_plotting',
                         'cluster_validation_plotting_pdfs',
                         'cluster_validation_processing']:
                setattr(self.IO, key,
                        self.IO.result_dir + self.IO.result_dir_validation +
                        getattr(self.IO.format, key).format(
                            data_info=data_info,
                            data_info_training=data_info_training,
                            settings_info=settings_info))
            elif key in ['pca_validation_plotting',
                         'pca_validation_plotting_pdfs',
                         'pca_validation_processing',
                         'pca_pipeline']:
                settings_info_pca = '_'.join(settings_info.split('_')[2:])
                if key == 'pca_pipeline':
                    setattr(self.IO, key,
                            self.IO.result_dir +
                            getattr(self.IO.format, key).format(
                                data_info_training=data_info_training,
                                # Remove cluster info from pca output
                                settings_info=settings_info_pca))
                else:
                    setattr(self.IO, key,
                            self.IO.result_dir +
                            self.IO.result_dir_validation +
                            getattr(self.IO.format, key).format(
                                data_info=data_info,
                                data_info_training=data_info_training,
                                # Remove cluster info from pca output
                                settings_info=settings_info_pca))
            elif key in ['sample_power',
                         'sample_vs_cluster_power']:
                if 'cluster' in key:
                    setattr(self.IO, key,
                            self.IO.result_dir
                            + getattr(self.IO.format, key).format(
                                sample_type=
                                self.Validation_Data.sample_type,
                                data_info=data_info,
                                data_info_training=data_info_training,
                                settings_info=settings_info))
                else:
                    setattr(self.IO, key,
                            self.IO.result_dir
                            + getattr(self.IO.format, key).format(
                                sample_type=
                                self.Validation_Data.sample_type,
                                data_info=data_info))
            elif key in ['plot_output_data']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info=data_info,
                            settings_info=settings_info))
            elif key not in ['locations']:
                setattr(self.IO, key,
                        self.IO.result_dir +
                        getattr(self.IO.format, key).format(
                            data_info_training=data_info_training,
                            settings_info=settings_info))

        for key in ['latitude_ds_file_name',
                    'model_level_file_name_format',
                    'surface_file_name_format']:
            setattr(self.Data, key,
                    self.Data.era5_data_dir + getattr(self.Data.format, key))

    # Output Configuration

    def dictify(self):
        # ??? why is this copy necessary..
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
