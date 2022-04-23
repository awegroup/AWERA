from AWERA import config, reference_locs, ChainAWERA
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

if __name__ == '__main__':
    add_plot_eval_dir = 'production/'
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    # settings_id = int(os.environ['SETTINGS_ID'])

    # test_final_setup_settings = training_settings[10:12]  # 5000, 1000
    # settings = training_settings[settings_id]
    settings = {
        'Data': {'n_locs': -1,  # 5000,
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
            # 'power_curve':
            #     add_plot_eval_dir + config.IO.format.power_curve,
            # 'cut_wind_speeds':
            #     add_plot_eval_dir + config.IO.format.cut_wind_speeds,
            # 'refined_cut_wind_speeds':
            #     add_plot_eval_dir + config.IO.format.refined_cut_wind_speeds,
            # 'optimizer_history':
            #     add_plot_eval_dir + config.IO.format.optimizer_history,
            }
        }

    settings['Processing'] = {'n_cores': 1}
    print(settings)
    # Single locaion evaluation
    loc = reference_locs[0]
    sample_id = 27822
    # Update settings to config
    config.update(settings)

    # Initialise AWERA eval with chosen config
    awera = ChainAWERA(config)
    working_title = 'Plotting production'

    n_clusters_list = [80]  # 8, 80]
    for n_clusters in n_clusters_list:
        prod_settings = {
            'Clustering': {'n_clusters': n_clusters}}
        awera.config.update(prod_settings)
        # awera.plot_cut_wind_speeds()
        # awera.plot_scaled_cut_wind_speed_limits(limits_type='estimate')
        # awera.plot_scaled_cut_wind_speed_limits(limits_type='refined')
        # awera.plot_power_curves()
        # Cluster freq:
        # awera.plot_cluster_frequency()
        pcs = []
        for i in range(n_clusters):
            pcs.append(awera.read_curve(i_profile=i+1,
                                        return_constructor=True))
        compare_profiles = {80: [10, 15, 61, 80]}  # {8: [1, 4, 6], 80: [1, 63, 75]}
        awera.compare_kpis(pcs, compare_profiles=compare_profiles[n_clusters])

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
