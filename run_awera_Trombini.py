import os
from AWERA import config, ChainAWERA
import numpy as np
# TODO make clear how each module runs with the inout of the previous
# power_model.load() .generate() how is the object made/passed


if __name__ == '__main__':
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    # loc_id = int(os.environ['LOC_ID'])

    # settings = {
    #     'Processing': {'n_cores': 2},  # 10
    #     'Data': {'n_locs': -1,
    #              'location_type': 'europe'},
    #     'Clustering': {
    #         'n_clusters': 8,  # TODO choose before predict labels...
    #         'training': {
    #             'n_locs': 5000,
    #             'location_type': 'europe'
    #             }
    #         },
    #     'IO': {'result_dir':
    #            "/cephfs/user/s6lathim/AWERA_results/"}
    # }
    n_clusters = 2

    # n_locs = 1 # [200, 500, 1000, 5000]
    n_l = 4  # n_locs[settings_id]
    scan_tag = '150m_'  # 'short_'  # 'full_freq'
    settings = {
        'Data': {'n_locs': 4,
                 'location_type': 'Trombini',
                 'init_locs': [[54.00, 0.00], [41.75, -9.25],
                               [37.75, 12.00], [41.00, 10.00]],
                 'use_data': 'ERA5',
                 'start_year': 2015,  # (4-digit int): Process the wind data starting from this year
                 'final_year': 2020,
                 'height_range':
                     [150.]},
        'Clustering': {
            'n_clusters': n_clusters,
            'training': {
                'n_locs': 1,
                'location_type': 'no_clustering'
                }
            },
        # !!! careful, this settings file does not (yet) change the rated power of the generator!!!
        'Power': {'kite_and_QSM_settings_file': 'kitepower_100kW'},
        'Processing': {'parallel': False,
                       'n_cores': n_clusters},
        'General': {'ref_height': 100},
        # 'Power':{ 'bounds': bounds},
        'IO': {
            'result_dir': "/cephfs/user/s6lathim/AWERA_results_Trombini/",
            'format': {
                'plot_output':
                    scan_tag + config.IO.format.plot_output,
                'power_curve':
                    scan_tag + config.IO.format.power_curve,
                'cut_wind_speeds':
                    scan_tag + config.IO.format.cut_wind_speeds,
                'refined_cut_wind_speeds':
                    scan_tag + config.IO.format.refined_cut_wind_speeds,
                # Only Power Production - no chain plot output for now
                'plot_output_data':
                    scan_tag + config.IO.format.plot_output_data,
                'training_plot_output':
                    scan_tag + config.IO.format.training_plot_output,
                'freq_distr':
                    scan_tag + config.IO.format.freq_distr,
                    }
                }
        }
    # settings['General'] = {'use_memmap': True}
    # settings[
    print('Settings:', settings)
    # Update settings to config
    config.update(settings)

    # working_title = 'eval_locs_{}'.format(loc_id)  # 'run_profile'

    from AWERA.eval.evaluation import evalAWERA
    # e = evalAWERA(config)
    # working_title = 'sliding_window_eval'
    # e.sliding_window_power(time_window=24,  # Hours for hourly data
    #                        at_night=False,  # True,
    #                        power_lower_bar=None,
    #                        power_lower_perc=15,
    #                        read_if_possible=True,
    #                        locs_slice=None,  # (loc_id, 1000)) # ,
    #                        read_from_slices=(23, 1000))  #
    # ----------------------------------------------------------
    # e.step_towing_AEP(power_consumption=20,
    #                   bridge_times=[1, 2, 3, 4, 5])
    # e.cut_in_out_distr()
    # ----------------------------------------------------------
    # e.aep_map()
    # print('Map plotted.')
    # e.power_freq()

    awera = ChainAWERA(config)
    print(awera.config)
    # Read and dump wind data
    data = awera.get_wind_data()
    import pickle
    with open('wind_data_Trombini_150m.pickle', 'wb') as f:
        pickle.dump(data, f)

    import sys
    sys.exit()
    awera.run_clustering()
    # Get 99.9% full frequency via:
    # Do this for 2016 to 2020
    awera.predict_labels()
    awera.get_frequency(bounds=[0, 35])
    awera.plot_cluster_frequency()

    awera.plot_cluster_shapes()

    print('Clustering Done.')


    # POWER
    limit_estimates = awera.estimate_wind_speed_operational_limits()
    pcs, limit_refined = awera.make_power_curves(limit_estimates=limit_estimates)
    # pcs = [awera.read_curve(i_profile=i+1, return_constructor=True)
    #         for i in range(n_clusters)]
    awera.compare_kpis(pcs, compare_profiles=list(range(1, 8+1)))

    print(awera.read_limits(refined=True))
    print(awera.read_profiles())
    awera.plot_power_curves(plot_full_electrical=True)
    awera.plot_power_curves(speed_at_op_height=True,
                            plot_full_electrical=True)
    # for i, pc in enumerate(pcs):
    #     pc.plot_output_file = config.IO.plot_output
    #     pc.plot_optimal_trajectories(plot_info='_profile_{}'.format(i+1))
    awera.get_frequency()

    awera.plot_cluster_frequency()
    awera.aep()

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

