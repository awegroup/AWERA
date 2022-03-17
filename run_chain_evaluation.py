from AWERA import config
from AWERA.eval.evaluation import evalAWERA
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

if __name__ == '__main__':
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    # settings_id = int(os.environ['SETTINGS_ID'])

    # test_final_setup_settings = training_settings[10:12]  # 5000, 1000
    # settings = training_settings[settings_id]
    settings = {
        'Data': {'n_locs': 5,  # 000,
                 'location_type': 'europe_ref'},
        'Clustering': {
            'training': {
                'n_locs': 5000,
                'location_type': 'europe'
                }
            },
    }
    settings['General'] = {'use_memmap': False}
    add_plot_eval_dir = 'eval/'
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
    settings['Processing'] = {'n_cores': 2}
    print(settings)
    # Update settings to config
    config.update(settings)

    # Initialise AWERA eval with chosen config
    eva = evalAWERA(config)

    # Code Profiling
    # TODO include in config -> optional
    # imports at top level / optional
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    # Run full clustering, production, aep estimation
    # depending on flags set in config
    # TODO include all important flags here to update conveniently

    # TODO check if clustering etc has to be done?

    working_title = 'run_evaluation'
    n_clusters_list = [8, 16] # , 80]
    for n_clusters in n_clusters_list:
        prod_settings = {
            'Clustering': {'n_clusters': n_clusters}}

        eva.config.update(prod_settings)
        # TODO the processing for this takes a lot of RAM still...
        eva.eval_wind_speed_at_harvesting_height()
        print('{} clusters done.'.format(n_clusters))
        print('------------------------------ Time:')
        write_timing_info('{} AWERA run finished.'.format(working_title),
                          time.time() - since)
        # --------------------------------------
    print('Done.')
    print('------------------------------ Config:')
    print(eva.config)
    print('------------------------------ Time:')
    write_timing_info('{} AWERA run finished.'.format(working_title),
                      time.time() - since)
    profiler.disable()
    # # Write profiler output
    file_name = eva.config.IO.plot_output.replace('.pdf', '.profile')

    with open(file_name.format(title='run_profile'), 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats('py:', .1)
    print('Profile output written to: ',
          file_name.format(title=working_title))


    # plt.show()



