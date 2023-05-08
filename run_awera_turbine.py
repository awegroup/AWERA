"""
Use AWERA to evaluate a custom power curve of a wind turbine.
"""
import os
import numpy as np
from AWERA import config, ChainAWERA
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()


if __name__ == '__main__':
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    # settings_id = int(os.environ['SETTINGS_ID'])

    # n_clusters_settings = [80]
    # n_clusters = n_clusters_settings[settings_id]

    # n_locs = 1 # [200, 500, 1000, 5000]
    scan_tag = 'fullfreq_'  # full_ half_  35_vw_ more_, short full_powering_stages
    settings = {
        'Data': {'n_locs': 1,
                 'location_type': 'Marseille'},
        'Clustering': {
            'n_clusters': 8,
            'training': {
                'n_locs': 1,
                'location_type': 'Marseille'
                }
            },
        'Processing': {'n_cores': 8},
        'General': {'ref_height': 100},
        # 'Power':{ 'bounds': bounds},
        'IO': {
            'result_dir': "/cephfs/user/s6lathim/AWERA_results/",
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
    print(settings)
    # Update settings to config
    config.update(settings)

    print(config)

    # Initialise AWERA chain with chosen config
    awera = ChainAWERA(config)

    awera.get_frequency(bounds=[0, 35])
    awera.plot_cluster_frequency()
    # model = '100kW'
    model = '500kW'
    if model == '500kW':
        # Turbine power curve 500kW
        # https://en.wind-turbine-models.com/turbines/383-vestas-v39#powercurve
        p = np.array([0, 0, 0, 0,
             18, 39, 60, 81.52,
             105, 134, 163, 197.5,
             232, 270.09, 305, 340,
             375, 407.5, 440, 459.83,
             478, 485.5, 493, 495.87,
             498.5, 499.1, 499.7, 499.85,
             500, 500, 500, 500,
             500, 500, 500, 500,
             500, 500, 500, 500,
             500, 500, 500, 500, 500
             ])*1000  # W for 500kW system
        v_hub = [3.,  3.5,  4.,  4.5,  5.,  5.5,  6.,  6.5,  7.,  7.5,  8.,
                 8.5,  9.,  9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5,
                 14., 14.5, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.5, 19.,
                 19.5, 20., 20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5,
                 25.]
        h_hub = 53  # m
    elif model == '100kW':
        # https://www.wind-turbine-models.com/turbines/1682-hummer-h25.0-100kw
        p = np.array([0, 5, 10, 17, 25, 34, 50, 63,
                      81, 100, 100, 100, 100, 100,
                      100, 100, 100, 100, 100, 100])*1000  # W for 100kW system
        v_hub = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        h_hub = 37.5  # m  # Estimate, 3 * 12.5m
    h_ref = awera.config.General.ref_height
    # Get wind speed at reference height
    envs = awera.create_cluster_environments()
    power_curves = []
    for i in range(awera.config.Clustering.n_clusters):
        env = envs[i]
        env.set_reference_wind_speed(1)
        v_h_hub = env.calculate_wind(h_hub)
        v = [v_hub_i / v_h_hub for v_hub_i in v_hub]
        power_curve = [v, p]
        power_curves.append(power_curve)

    awera.evaluate_power_curve(power_curves=power_curves)

    print('Done.')
    print('------------------------------ Config:')
    print(awera.config)
    print('------------------------------ Time:')

    # profiler.disable()
    # # # Write profiler output
    # file_name = awera.config.IO.plot_output.replace('.pdf', '.profile')

    # with open(file_name.format(title='run_profile'), 'w') as f:
    #     stats = pstats.Stats(profiler, stream=f)
    #     stats.strip_dirs()
    #     stats.sort_stats('cumtime')
    #     stats.print_stats('py:', .1)
    # print('Profile output written to: ',
    #       file_name.format(title=working_title))
