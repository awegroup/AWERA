import matplotlib.pyplot as plt
import numpy as np
import pickle

from AWERA.power_production.config import Config
from AWERA.power_production.power_production import PowerProduction

from AWERA.power_production.qsm import LogProfile, TractionPhase

from upscaling_kite_specs_and_settings import kite_qsm_settings, kite_sys_props
# Get configuration from config.yaml file in power production
config = Config()
import os

sel_types = [
    # 10m
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    # 100m
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
    ]
do_settings_scan = True
read_curve = True
if do_settings_scan:
    # Scan kitepower 100kW system
    SEL_ID = 4  # 1 or 4 for 10m or 100m ref
    SCAN_SEL = int(os.environ['SEL_ID'])
else:
    SEL_ID = int(os.environ['SEL_ID'])
# 500kW only
# sel_types = [sel_types[2], sel_types[5]]

sel = sel_types[SEL_ID]
rated_power_sel = [100, 500]
rated_power = rated_power_sel[sel[0]]
kite_sel = ['', 'kitepower_']
kite_source = kite_sel[sel[1]]
add_plot_eval_dir = '{}kite_{}_kW/'.format(kite_source, rated_power)

kite = kite_source + str(rated_power)
print(kite)
sys_props = kite_sys_props[kite]
settings = kite_qsm_settings[kite]

# Update bounds
if kite == '100':
    # cycle start set to 150m at lowest, not optimised
    bounds = [None, None, None, None, [150, 250]]
    x0 = [8000., 1000., 0.5, 240., 150.0]
elif 'kitepower' in kite:
    # Tether between 200 and 500m
    bounds = [None, None, None, [150, 300], [200, 250]]
    x0 = [8000., 1000., 0.5, 300., 200.0]
    # x0 = [45000., 8000., 0.5, 300., 200.0]


if do_settings_scan:
    scan_opts = [
        [500, [2, 10], 110, 500, [25, 60]],  # Default
        [250, [2, 10], 110, 500, [25, 60]],  # low_min_ft
        [500, [0.5, 10], 110, 500, [25, 60]],  # low_min_vr
        [500, [2, 10], 90, 500, [25, 60]],  # course_ang_90
        [500, [2, 10], 110, 700, [25, 60]],  # high_max_dl
        [500, [2, 10], 110, 500, [20, 60]],  # low_min_elev
        [500, [2, 10], 110, 500, [25, 75]],  # high_max_elev
        ]
    scan_tags = [
        'default',
        'low_min_ft',
        'low_min_vr',
        'course_ang_90',
        'high_max_dl',
        'low_min_elev',
        'high_max_elev'
        ]
    scan_tag = scan_tags[SCAN_SEL] + '_'
    scan_opt = scan_opts[SCAN_SEL]
    print('scan opt:', scan_opt, scan_tag)
    # Settings scan
    # System:
    # min tether force, reeling speed limits, course angle
    sys_props['tether_force_min_limit'] = scan_opt[0]
    sys_props['reeling_speed_min_limit'] = scan_opt[1][0]
    sys_props['reeling_speed_max_limit'] = scan_opt[1][1]

    settings['traction']['course_angle'] = scan_opt[2] * np.pi / 180.

    # optimisation bounds:
    # max tether length, min/max elevation angle
    max_l = scan_opt[3]
    dl_max = max_l - x0[4]
    bounds[3] = [150, dl_max]
    angles = scan_opt[4]
    bounds[2] = [angles[0]*np.pi/180, angles[1]*np.pi/180.]
else:
    scan_tag = ''

ref_height_sel = [10, 100]
ref_height = ref_height_sel[sel[2]]  # m

wind_speeds_sel = {10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                        13, 14, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19,
                        19.5, 20, 21],
                   100:  # [11, 12, 13],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28]
                   }  # m/s
wind_speeds = wind_speeds_sel[ref_height]


config.update({'General': {'ref_height': ref_height},
               'Power': {'profile_type': '{}kW_log_profile'.format(
                   rated_power),
                         'bounds': bounds},
               'IO': {
                   'result_dir': "/cephfs/user/s6lathim/AWERA_results/",
                   'format': {
                       'plot_output':
                           add_plot_eval_dir
                           + scan_tag + '{}m_ref_'.format(ref_height)
                           + config.IO.format.plot_output,
                       'power_curve':
                           add_plot_eval_dir
                           + scan_tag + '{}m_ref_'.format(ref_height)
                           + config.IO.format.power_curve,
                       # Only Power Production - no chain plot output for now
                       # 'plot_output_data':
                       #     add_plot_eval_dir
                       #     + config.IO.format.plot_output_data,
                       # 'training_plot_output':
                       #     add_plot_eval_dir
                       #     + config.IO.format.training_plot_output  #
                           }
                       }
               })
setattr(config.IO, 'training_plot_output', config.IO.plot_output)
print(config)

# Initialise power production functions class
prod = PowerProduction(config)

# Create pumping cycle simulation object, run simulation, and plot results.
# cycle = Cycle(settings)
# cycle.follow_wind = True
# cycle.run_simulation(sys_props, env_state, print_summary=True)
# cycle.time_plot(('reeling_speed', 'power_ground', 'tether_force_ground'),
#                 ('Reeling\nspeed [m/s]', 'Power [W]', 'Tether\nforce [N]'))

# cycle.trajectory_plot()
# cycle.trajectory_plot3d()
# plt.show()

if read_curve:
    pc = prod.read_curve(i_profile='log', return_constructor=True)
    env_state = LogProfile()
    env_state.set_reference_height(ref_height)
    env_state.set_reference_wind_speed(wind_speeds[0])
    oc = prod.create_optimizer(env_state, wind_speeds[0],
                               sys_props=sys_props,
                               cycle_sim_settings=settings,
                               print_details=True)
else:
    pc, oc = prod.single_power_curve(
        wind_speeds,
        x0=x0,
        sys_props=sys_props,
        cycle_sim_settings=settings,
        plot_output_file=prod.config.IO.plot_output.format(
            title='{title}'+'_{}kw_log_profile_opt'.format(rated_power)),
        env_state=None,  # Defaults to Log Profile
        ref_height=ref_height,
        return_optimizer=True,
        print_details=True)
    pc.export_results(config.IO.power_curve.format(i_profile='log',
                                                   suffix='pickle'))

wind_speed, mean_cycle_power = pc.curve()
print(wind_speed, mean_cycle_power)


# mask failed simulation, export only good results
sel_succ = [kpis['sim_successful']
            for kpis in pc.performance_indicators]
print('# Failed runs: ', -sum(sel_succ)+len(sel_succ))
print('# Successful runs: ', sum(sel_succ))

# Plot power curve together with that of the other wind profile shapes.
p_cycle = np.array([kpis['average_power']['cycle']
                    for kpis in pc.performance_indicators])[sel_succ]
print('p_cycle: ', p_cycle)
wind = pc.wind_speeds

# pc.plot_output_file = prod.config.IO.plot_output
pc.plot_optimal_trajectories(plot_info='_profile_log',
                             circle_radius=
                             settings['cycle']['tether_length_end_retraction'])

pc.plot_optimization_results(oc.OPT_VARIABLE_LABELS,
                             oc.bounds_real_scale,
                             [sys_props['tether_force_min_limit'],
                              sys_props['tether_force_max_limit']],
                             [sys_props['reeling_speed_min_limit'],
                              sys_props['reeling_speed_max_limit']],
                             plot_info='_profile_log')

n_cwp = np.array([kpis['n_crosswind_patterns']
                  for kpis in pc.performance_indicators]
                 )[sel_succ]  # [sel_succ_power]
x_opts = np.array(pc.x_opts)[sel_succ]  # [sel_succ_power]
if config.General.write_output:
    from AWERA.power_production.power_curves import export_to_csv
    export_to_csv(config, wind, wind[-1], p_cycle,
                  x_opts, n_cwp, 'log')


prod.compare_kpis([pc])
prod.plot_trajectories(pc)
prod.plot_power_curves(pcs=pc, labels='{}kW log profile'.format(rated_power),
                       lim=[wind_speeds[0], wind_speeds[-1]],
                       speed_at_op_height=True)
with open(prod.config.IO.plot_output.format(
        title='power_curve_constructor_{}kw_log_profile_opt'.format(rated_power))
        .replace('.pdf', '.pickle'), 'wb')\
        as f:
    pickle.dump(pc, f)
print('Power Curve output written.')
wind_speed, mean_cycle_power = pc.curve()


# Plot power curve(s)
# prod.plot_power_curves(pc)
# plt.show()

# x0 and x_opt:
# OPT_VARIABLE_LABELS = [
#         "Reel-out\nforce [N]",
#         "Reel-in\nforce [N]",
#         "Elevation\nangle [rad]",
#         "Reel-in tether\nlength [m]",
#         "Minimum tether\nlength [m]"
#     ]

# op_res: optimizer output
# cons: optimizer constraints
# kpis: kite performance indicators
