import os
from AWERA import config, ChainAWERA
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

training_settings = []
prediction_settings = []

# Test run on 5 special locations
for i in range(5):
    for j in range(5):
        settings = {
            'Processing': {'n_cores': 10},
            'Data': {
                'n_locs': 1,
                'location_type': 'europe_ref_{}'.format(i)},
            'Clustering': {
                'training': {
                    'n_locs': 1,
                    'location_type': 'europe_ref_{}'.format(j)
                    }
                },
            }
        if i == 0:
            training_settings.append(settings)
        prediction_settings.append(settings)

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'n_clusters': 8,
            'training': {
                'n_locs': 5,
                'location_type': 'europe_ref'
                }
            },
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

    # multiple locations
    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {
                'n_locs': 50,
                'location_type': 'europe'
                }
            },
        }

    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {
                'n_locs': 200,
                'location_type': 'europe'
                }
            },
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

    # Multiple locations plus 5 special locs
    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {
                'n_locs': 55,
                'location_type': 'europe_incl_ref'
                }
            },
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {
                'n_locs': 205,
                'location_type': 'europe_incl_ref'
                }
            },
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

# very many locations
settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {
            'n_locs': 5000,
            'location_type': 'europe'
            }
        },
    }
training_settings.append(settings)
# TODO include prediction
settings = {
    'Processing': {'n_cores': 50},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {
            'n_locs': 1000,
            'location_type': 'europe'
            }
        },
    }
training_settings.append(settings)
settings = {
    'Processing': {'n_cores': 50},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {
            'n_locs': 5005,
            'location_type': 'europe_incl_ref'
            }
        },
    }
training_settings.append(settings)

settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {
            'n_locs': 1005,
            'location_type': 'europe_incl_ref'
            }
        },
    }
training_settings.append(settings)

# TODO make clear how each module runs with the inout of the previous
# power_model.load() .generate() how is the object made/passed


if __name__ == '__main__':
    import os
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # read config from jobnumber
    #settings_id = int(os.environ['SETTINGS_ID'])  # use 6, 7

    #settings = training_settings[settings_id]
    settings = {
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_0'},
        'Clustering': {
            'training': {
                'n_locs': 500,
                'location_type': 'europe'
                }
            },
        }

    settings['General'] = {'use_memmap': True}
    settings['Processing'] = {'n_cores': 15}

    # settings = {
    #     'Processing': {'n_cores': 100},
    #     'Data': {'n_locs': -1,
    #               'location_type': 'europe'},
    #     'Clustering': {
    #         'training': {
    #             'n_locs': 5000,
    #             'location_type': 'europe'
    #             }
    #         },
    # }

    print(settings)
    # Update settings to config
    config.update(settings)

    # Initialise AWERA chain with chosen config
    awera = ChainAWERA(config)

    # Code Profiling
    # TODO include in config -> optional
    # imports at top level / optional
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()

    # Run full clustering, production, aep estimation
    # depending on flags set in config
    # TODO include all important flags here to update conveniently

    # TODO check if clustering etc has to be done?

    working_title = 'run_clustering'  # 'run_production'
    #  'run_production' #'predict_labels' #  'file'
    prod_settings = {
        # 'Processing': {'n_cores': 57},
        'Clustering': {'n_clusters': 8}}
    awera.config.update(prod_settings)
    # awera.predict_labels()
    profiles, data = awera.train_profiles(return_data=True)
    print('8 clusters done.')
    print('------------------------------ Time:')
    write_timing_info('{} AWERA run finished.'.format(working_title),
                      time.time() - since)
    prod_settings = {
        # 'Processing': {'n_cores': 57},
        'Clustering': {'n_clusters': 16}}
    awera.config.update(prod_settings)
    # awera.predict_labels()
    profiles, data = awera.train_profiles(data=data)
    print('16 clusters done.')
    print('------------------------------ Time:')
    write_timing_info('{} AWERA run finished.'.format(working_title),
                      time.time() - since)
    prod_settings = {
        # 'Processing': {'n_cores': 57},
        'Clustering': {'n_clusters': 80}}
    awera.config.update(prod_settings)
    # awera.predict_labels()
    profiles, data = awera.train_profiles(data=data)

    print('Done.')
    print('------------------------------ Config:')
    print(awera.config)
    print('------------------------------ Time:')
    write_timing_info('{} AWERA run finished.'.format(working_title),
                      time.time() - since)
    profiler.disable()
    # # Write profiler output
    file_name = config.IO.plot_output.replace('.pdf', '.profile')

    with open(file_name.format(title='run_profile'), 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats('py:', .1)
    print('Profile output written to: ',
          file_name.format(title=working_title))

    import os
    import glob
    # Get a list of all the file paths that ends with .memmap from in specified directory
    fileList = glob.glob('tmp/*.memmap')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except OSError:
            print("Error while deleting file : ", filePath)
    folder_path = 'tmp'
    if os.path.exists(folder_path):

        # checking whether the folder is empty or not
        if len(os.listdir(folder_path)) == 0:
            # removing the file using the os.remove() method
            os.rmdir(folder_path)
            print('Tmp files cleaned up.')
        else:
            # messaging saying folder not empty
            print("Tmp folder is not empty")
    else:
        # file not found message
        print("Tmp Folder not found in the directory")
    #plt.show()



