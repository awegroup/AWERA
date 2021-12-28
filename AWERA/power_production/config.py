# -*- coding: utf-8 -*-
"""Configuration file for wind power production.
"""
# TODO update docstring

import yaml
import os
import numpy as np


class Config:
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
        # --------------------------- FILE SUFFIX
        data_info, data_info_training = self.Power.profile_type, \
            self.Power.profile_type
        # --------------------------- DIR + FILE and SUFFIX FORMATTING
        for key in self.IO.format.__dict__:
            if key in ['plot_output']:
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
































# TODO Docstring

# Running on a job:
#import os
#jobnum = int(os.environ['LOCNUM'])
#step = 1
#location_number = jobnum*step
# Select loctions up to location number, -1: all locs #up to +400

# single_fail_loc_ids = [469, 805, 863, 975]
# print('Set Jobnumber/Locnumber: ', jobnum, 'in steps of ', step)
# location_number = single_fail_loc_ids[jobnum]

# TODO remove this function - put in config.py what is necessary
#def get_loc_brute_force_name_and_locs(location_number, data_info,
#                                      mult_loc=False):
    # location_number=10
    # -1 #select loctions up to location number, -1: all locs in mult loc case#

#    if mult_loc:
#        if location_number != -1:
#            locs = locations[:location_number]
#            data_info += 'first_{}_locs'.format(location_number)
#        else:
#            locs = locations
#    else:
#        locs = [locations[location_number]]
#        data_info += 'loc_{}'.format(location_number)
#    return data_info, locs


#data_info, locs = get_loc_brute_force_name_and_locs(location_number, data_info)
# print('Locations: ', locs)#

#brute_force_files = 'brute_force_power_{}_samples_{}.{}'
#brute_force_testing_file_name = (
#    result_dir
#    + brute_force_files.format(sample_selection, data_info, 'pickle'))
