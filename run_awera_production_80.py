import os
from AWERA import config, ChainAWERA
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

training_settings = []
prediction_settings = []



if __name__ == '__main__':
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    settings_id = int(os.environ['SETTINGS_ID'])

    # test_final_setup_settings = training_settings[10:12]  # 5000, 1000
    # settings = training_settings[settings_id]
    n_clusters_settings = [8, 16, 80]
    n_clusters = n_clusters_settings[2]

    n_locs = [200, 500, 1000, 5000]
    n_l = n_locs[settings_id]
    settings = {
        'Data': {'n_locs': 50,
                 'location_type': 'europe'},
        'Clustering': {
            'n_clusters': n_clusters,
            'training': {
                'n_locs': n_l,
                'location_type': 'europe'
                }
            },
        'Processing': {'n_cores': 50},
    }
    # settings['General'] = {'use_memmap': True}
    # settings[
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

    working_title = 'run_production'  #  'run_production' #'predict_labels' #  'file'
    # prod_settings = {
    #     #'Processing': {'n_cores': 57},
    #     'Clustering': {'n_clusters': 8}}
    # awera.config.update(prod_settings)
    # #awera.predict_labels()
    # profiles, data = awera.train_profiles(return_data=True)
    # # limit_estimates = awera.estimate_wind_speed_operational_limits()
    # # pcs, limit_refined = awera.make_power_curves(limit_estimates=limit_estimates)
    # # awera.compare_kpis(pcs)
    # print('8 clusters done.')
    # print('------------------------------ Time:')
    # write_timing_info('{} AWERA run finished.'.format(working_title),
    #                   time.time() - since)
    # prod_settings = {
    #     #'Processing': {'n_cores': 57},
    #     'Clustering': {'n_clusters': 16}}
    # awera.config.update(prod_settings)
    # awera.predict_labels()
    # profiles = awera.train_profiles(data=data)
    # limit_estimates = awera.estimate_wind_speed_operational_limits()
    # pcs, limit_refined = awera.make_power_curves(limit_estimates=limit_estimates)
    # awera.compare_kpis(pcs)
    # print('16 clusters done.')
    # print('------------------------------ Time:')
    # write_timing_info('{} AWERA run finished.'.format(working_title),
    #                   time.time() - since)
    # prod_settings = {
    #     # 'Processing': {'n_cores': 57},
    #     'Clustering': {'n_clusters': 80}}
    # awera.config.update(prod_settings)
    # # awera.predict_labels()
    # # profiles = awera.train_profiles(data=data)
    # limit_estimates = awera.estimate_wind_speed_operational_limits()
    # pcs, limit_refined = awera.make_power_curves(limit_estimates=limit_estimates)
    # awera.compare_kpis(pcs)

    # awera.predict_labels()
    # profiles = awera.train_profiles(data=data)
    limit_estimates = awera.estimate_wind_speed_operational_limits()
    pcs, limit_refined = awera.make_power_curves(limit_estimates=limit_estimates)
    awera.compare_kpis(pcs)

    print('Done.')
    print('------------------------------ Config:')
    print(awera.config)
    print('------------------------------ Time:')
    write_timing_info('{} AWERA run finished.'.format(working_title),
                      time.time() - since)
    profiler.disable()
    # # Write profiler output
    file_name = awera.config.IO.plot_output.replace('.pdf', '.profile')

    with open(file_name.format(title='run_profile'), 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats('py:', .1)
    print('Profile output written to: ',
          file_name.format(title=working_title))


    #plt.show()



