from AWERA import config, reference_locs
from AWERA.validation.validation import ValidationPlottingClustering
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

if __name__ == '__main__':
    settings = {
        'Data': {'n_locs': 500,
                 'location_type': 'europe'},
        'Clustering': {
            'training': {
                'n_locs': 500,
                'location_type': 'europe'
                }
            },
    }
    settings['General'] = {'use_memmap': False}
    settings['Processing'] = {'n_cores': 1}
    print(settings)
    # Single locaion evaluation
    # loc = reference_locs[0]
    # sample_id = 27822
    # Update settings to config
    config.update(settings)

    # Initialise AWERA eval with chosen config
    val = ValidationPlottingClustering(config)
    working_title = 'Plotting clustering validation'

    # val.plot_all_single_loc(min_n_pcs=3,
    #                         plot_height_dependence=False,
    #                         plot_single_velocity_dependence=False,
    #                         plot_velocity_dependence=True,
    #                         plot_backscaling=True)

    settings = {
        'Data': {'n_locs': -1,
                  'location_type': 'europe'},
        'Clustering': {
            'n_clusters': 80,
            'eval_n_pc_up_to': 5,
            'training': {
                'n_locs': 1000,
                'location_type': 'europe'
                }
            },
    }
    val.config.update(settings)
    # val.plot_cluster_diff_maps(eval_type='cluster',
    #                             sel='v_greater_3')  # 'v_greater_3')  # 'full'
    val.plot_cluster_diff_maps(eval_type='cluster',
                                sel='v_greater_5',
                                locs_slice=(23, 1000))

    settings = {
        'Data': {'n_locs': 1000,
                  'location_type': 'europe_ref'},
        'Clustering': {
            'n_clusters': 80,
            'eval_n_pc_up_to': 5,
            'training': {
                'n_locs': 5000,
                'location_type': 'europe'
                }
            },
    }
    val.config.update(settings)

    # val.plot_all_single_loc(min_n_pcs=5,
    #                         plot_height_dependence=False,
    #                         plot_single_velocity_dependence=False,
    #                         plot_velocity_dependence=True,
    #                         plot_backscaling=True)

    # sel_list = ['full', 'v_greater_3', 'v_greater_1']
    # for sel in sel_list:
    #     # TODO abs = rel??
    #     val.plot_cluster_loc_diffs(training_locs=[# (5, 'europe_ref'),  # TODO this will be 200
    #                                               (500, 'europe'),
    #                                               (1000, 'europe'),
    #                                               (5000, 'europe')],
    #                                 # TODO data will be 1k ref and 1k
    #                                 data_locs=[(1000, 'europe_ref')]*3,  # [(5, 'europe_ref')]*4,
    #                                 sel=sel,
    #                                 n_pcs=5)
    # --------------------------------------

    print('Done.')
    print('------------------------------ Config:')
    print(val.config)
    print('------------------------------ Time:')
    write_timing_info('{} AWERA run finished.'.format(working_title),
                      time.time() - since)

    # plt.show()
