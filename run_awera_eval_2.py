import os
from AWERA import config, ChainAWERA
import numpy as np
# TODO make clear how each module runs with the inout of the previous
# power_model.load() .generate() how is the object made/passed


if __name__ == '__main__':
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    loc_id = int(os.environ['LOC_ID'])

    settings = {
        'Processing': {'n_cores': 2},  # 10
        'Data': {'n_locs': -1,
                 'location_type': 'europe'},
        'Clustering': {
            'n_clusters': 8,  # TODO choose before predict labels...
            'training': {
                'n_locs': 5000,
                'location_type': 'europe'
                }
            },
        'IO': {'result_dir':
               "/cephfs/user/s6lathim/AWERA_results/"}
    }
    print(settings)
    # Update settings to config
    config.update(settings)

    working_title = 'eval_locs_{}'.format(loc_id)  # 'run_profile'

    from AWERA.eval.evaluation import evalAWERA
    e = evalAWERA(config)
    # working_title = 'sliding_window_eval'
    # e.sliding_window_power(time_window=24,  # Hours for hourly data
    #                        at_night=False,  # True,
    #                        power_lower_bar=None,
    #                        power_lower_perc=15,
    #                        read_if_possible=True,
    #                        locs_slice=None,  # (loc_id, 1000)) # ,
    #                        read_from_slices=(23, 1000))  #

    e.aep_map()
    # print('Map plotted.')
    # e.power_freq()

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

