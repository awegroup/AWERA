import os
from AWERA import config, ChainAWERA

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
    'Processing': {'n_cores': 23},
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
    'Processing': {'n_cores': 23},
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
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    loc_id = int(os.environ['LOC_ID'])

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': -1,
                 'location_type': 'europe'},
        'Clustering': {
            'training': {
                'n_locs': 5000,
                'location_type': 'europe'
                }
            },
    }
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

    #working_title = 'predict_labels_{}'.format(loc_id) # 'run_profile'
    #awera.predict_labels(locs_slice=(loc_id, 1000))
    #working_title = 'combine_labels'
    #awera.combine_labels()
    working_title = 'plotting'
    #awera.plot_cluster_shapes()
    #awera.get_frequency()
    from AWERA.eval.evaluation import evalAWERA
    e = evalAWERA(config)
    e.aep_map()
    #awera.run()
    e.power_freq()

    profiler.disable()
    # # Write profiler output
    file_name = (
        config.IO.result_dir
        + config.IO.format.plot_output.format(
                  data_info=config.Data.data_info,
                  data_info_training=config.Clustering.training.data_info,
                  settings_info=config.Clustering.training.settings_info)
        .replace('.pdf', '{}.profile'.format(loc_id))
        )
    with open(file_name.format(title='run_profile'), 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats('py:', .1)
    print('Profile output written to: ',
          file_name.format(title=working_title))


    #plt.show()
