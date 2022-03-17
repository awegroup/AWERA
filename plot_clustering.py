from AWERA import config, reference_locs, ChainAWERA
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

if __name__ == '__main__':
    add_plot_eval_dir = 'clustering/'
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    # settings_id = int(os.environ['SETTINGS_ID'])

    # test_final_setup_settings = training_settings[10:12]  # 5000, 1000
    # settings = training_settings[settings_id]
    settings = {
        'Data': {'n_locs': 5000,
                 'location_type': 'europe'},
        'Clustering': {
            'n_clusters': 8,
            'training': {
                'n_locs': 5000,
                'location_type': 'europe'
                }
            },
    }
    settings['General'] = {'use_memmap': True}
    settings['IO'] = {
        'result_dir': "/cephfs/user/s6lathim/AWERA_results/",
        'format': {
            'plot_output':
                add_plot_eval_dir + config.IO.format.plot_output,
            'plot_output_data':
                add_plot_eval_dir + config.IO.format.plot_output_data,
            'training_plot_output':
                add_plot_eval_dir + config.IO.format.training_plot_output,
            }
        }
    settings['Processing'] = {'n_cores': 15}
    print(settings)
    # Single locaion evaluation
    loc = reference_locs[0]
    sample_id = 27822
    # Update settings to config
    config.update(settings)

    # Initialise AWERA eval with chosen config
    awera = ChainAWERA(config)
    working_title = 'Plotting clustering'

    n_clusters_list = [80, 8, 16]
    for n_clusters in n_clusters_list:
        prod_settings = {
            'Clustering': {'n_clusters': n_clusters}}
        awera.config.update(prod_settings)
        # awera.cluster_frequency_maps(use_rel='cluster')
        # awera.cluster_frequency_maps(use_rel='loc')
        # # ----------------------------------------
        # awera.visualize_clustering_flow(loc=loc, sample_id=sample_id)

        # awera.original_vs_cluster_wind_profile_shapes(loc=loc,
        #                                               sample_id=sample_id,
        #                                               x_lim=(-12, 17),
        #                                               y_lim=(-12, 12))
        # awera.plot_cluster_shapes(scale_back_sf=False,
        #                           x_lim_profiles=[-2.2, 3.2],
        #                           y_lim_profiles=[-1.7, 1.7])

        data = awera.cluster_pc_projection(return_data=True)
        awera.analyse_pc(data=data)

        # ----------------------------------------
        print('{} clusters done.'.format(n_clusters))
        print('------------------------------ Time:')
        write_timing_info('{} AWERA run finished.'.format(working_title),
                          time.time() - since)
    # --------------------------------------

    print('Done.')
    print('------------------------------ Config:')
    print(awera.config)
    print('------------------------------ Time:')
    write_timing_info('{} AWERA run finished.'.format(working_title),
                      time.time() - since)

    # plt.show()
