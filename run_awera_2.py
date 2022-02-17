import os
from AWERA import config, ChainAWERA
import numpy as np
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
    #loc_id = int(os.environ['LOC_ID'])

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 5000,  # -1
                 'location_type': 'europe'},
        'Clustering': {
            'training': {
                'n_locs': 5000,
                'location_type': 'europe'
                }
            },
        'IO': {'result_dir':
               "/cephfs/user/s6lathim/AWERA_results/DOWA_height_range_results/"}
    }
    print(settings)
    # Update settings to config
    config.update(settings)

    # Initialise AWERA chain with chosen config
    awera = ChainAWERA(config)

    # Code Profiling
    # TODO include in config -> optional
    # imports at top level / optional
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # awera.get_frequency()
    v_test = []
    profiles = awera.read_profiles()
    freq, v_bin_limits = awera.read_frequency()
    for i_cluster in range(awera.config.Clustering.n_clusters):
        heights = profiles['height [m]']

        v_prof = list((profiles['u{} [-]'.format(i_cluster + 1)]**2
                       + profiles['v{} [-]'.format(i_cluster + 1)]**2)**.5)
        h_ref = 10  # m
        v_at_ref = 3  # m/s
        v_prof = awera.scale_profile(v_prof, v_at_ref, h_ref, heights=heights)
        h_test = 100  # m
        v_at_test = awera.get_wind_speed_at_height(v_prof, h_test, heights=heights)

        v_test.append(v_at_test)

        # Get Cluster frequency at v = v_at_test
        # TODO different if h_test is not standard h_ref!
        freq_c = freq[:, i_cluster, :]
        v_at_cluster_ref = v_at_test
        v_bins_c = v_bin_limits[i_cluster, :]
        v_bin_idx = int(np.round(np.interp(v_at_cluster_ref, v_bins_c,
                                           list(range(len(v_bins_c))))))
        freq_test = freq_c[:, v_bin_idx]
        # Plot freq map

        print('Cluster {} | V 100m {} | Mean freq {}'.format(
            i_cluster+1, v_at_test, np.mean(freq_test)))
    print(v_test)

    # Run full clustering, production, aep estimation
    # depending on flags set in config
    # TODO include all important flags here to update conveniently

    # TODO check if clustering etc has to be done?

    #working_title = 'predict_labels_{}'.format(loc_id) # 'run_profile'
    #awera.predict_labels() # locs_slice=(loc_id, 1000))
    #print('Labels predicted.')
    #working_title = 'combine_labels'
    #awera.combine_labels()
    #working_title = 'plotting'
    #awera.plot_cluster_shapes()
    # #awera.get_frequency()
    # from AWERA.eval.evaluation import evalAWERA
    # e = evalAWERA(config)
    # working_title = 'sliding_window_eval'
    # e.sliding_window_power()
    #e.aep_map()
    #awera.run()
    #e.power_freq()

    # profiler.disable()
    # # # Write profiler output
    # file_name = (
    #     config.IO.result_dir
    #     + config.IO.format.plot_output.format(
    #               data_info=config.Data.data_info,
    #               data_info_training=config.Clustering.training.data_info,
    #               settings_info=config.Clustering.training.settings_info)
    #     .replace('.pdf', '{}.profile'.format(loc_id))
    #     )
    # with open(file_name.format(title='run_profile'), 'w') as f:
    #     stats = pstats.Stats(profiler, stream=f)
    #     stats.strip_dirs()
    #     stats.sort_stats('cumtime')
    #     stats.print_stats('py:', .1)
    # print('Profile output written to: ',
    #       file_name.format(title=working_title))


    #plt.show()

