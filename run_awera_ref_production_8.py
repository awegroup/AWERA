import os
from AWERA import config, ChainAWERA
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()


if __name__ == '__main__':
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    settings_id = int(os.environ['SETTINGS_ID'])

    n_clusters_settings = [8, 16, 80]
    n_clusters = n_clusters_settings[0]

    n_locs = [1, 1, 1, 1, 1, 5]
    loc_types = ['europe_ref_0',
                 'europe_ref_1',
                 'europe_ref_2',
                 'europe_ref_3',
                 'europe_ref_4',
                 'europe_ref']
    n_l = n_locs[settings_id]
    loc_type = loc_types[settings_id]
    settings = {
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_0'},
        'Clustering': {
            'n_clusters': n_clusters,
            'training': {
                'n_locs': n_l,
                'location_type': loc_type
                }
            },
        'Processing': {'n_cores': n_clusters},
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

    working_title = 'run_awera'  #  'run_production' #'predict_labels' #  'file'
    awera.train_profiles()
    limit_estimates = awera.estimate_wind_speed_operational_limits()
    pcs, limit_refined = awera.make_power_curves(limit_estimates=limit_estimates)
    #awera.compare_kpis(pcs)

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



