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
    # # 10m
    # [0, 0, 0],
    # [0, 1, 0],
    # [1, 1, 0],
    # 100m
    # [0, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
    ]
do_settings_scan = False  # True
read_curve = False
use_phase_eff = False  # True
if do_settings_scan:
    # Scan kitepower 100kW system
    SEL_ID = 0  # 100kW or 500kW 10m or 100m ref
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
    bounds = [None, None, None, None, [150, 250], [0, 1]]
    x0 = [8000., 1000., 0.5, 240., 150.0, 1]
elif 'kitepower' in kite:
    # Tether between 200 and 500m
    bounds = [None, None, None, [150, 300], [200, 250], [0.5, 1]]  # , [150, 300] [150, 500]
    x0 = [25000., 1000., 0.5, 230., 200.0, 1]
    # 20m/s only:
    # x0 = [49000., 2000., 0.9, 300., 200.0, 1]
    # x0 = [5*49000., 2000., 0.9, 300., 200.0, 1]


if do_settings_scan:
    scan_opts = [
        [[500, None], [0.05, [10, 5]], 110, 500, [25, 60]],  # diff_max_vr  #  in out
        [[500, 78000], [0.05, [10, 5]], 110, 500, [25, 60]],  # diff_max_vr_kitepower  #  in out
        [[500, None], [2, 10], 110, 500, [25, 60]],  # Default
        [[500, None], [0.05, 5], 110, 500, [25, 60]],  # low_min_max_vr
        [[500, 78000], [0.05, 5], 110, 500, [25, 60]],  # kitepower
        [[500, None], [0.5, 10], 110, 500, [25, 60]],  # low_min_vr
        [[500, None], [0.000001, 10], 110, 500, [25, 60]],  # zero_min_vr
        [[500, None], [2, 20], 110, 500, [25, 60]],  # high_max_vr
        [[500, None], [2, 10], 90, 500, [25, 60]],  # course_ang_90
        [[500, None], [2, 10], 110, 700, [25, 60]],  # high_max_dl
        [[500, None], [2, 10], 110, 500, [20, 60]],  # low_min_elev
        [[500, None], [2, 10], 110, 500, [25, 75]],  # high_max_elev
        [[250, None], [2, 10], 110, 500, [25, 60]],  # low_min_ft
        ]
    scan_tags = [
        'diff_max_vr',
        'diff_max_vr_kitepower',
        'default',
        'low_min_max_vr',
        'kitepower',
        'low_min_vr',
        'zero_min_vr',
        'high_max_vr',
        'course_ang_90',
        'high_max_dl',
        'low_min_elev',
        'high_max_elev',
        'low_min_ft',
        ]
    scan_tag = scan_tags[SCAN_SEL] + '_'
    scan_opt = scan_opts[SCAN_SEL]
    print('scan opt:', scan_opt, scan_tag)
    # Settings scan
    # System:
    # min tether force, reeling speed limits, course angle
    sys_props['tether_force_min_limit'] = scan_opt[0][0]
    if scan_opt[0][1] is not None:
        sys_props['tether_force_max_limit'] = scan_opt[0][1]
    sys_props['reeling_speed_min_limit'] = scan_opt[1][0]
    if type(scan_opt[1][1]) == list:
        sys_props['reeling_speed_max_limit_retr'] = scan_opt[1][1][0]
        sys_props['reeling_speed_max_limit_trac'] = scan_opt[1][1][1]
    else:
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

if use_phase_eff:
    scan_tag += 'phase_eff_'
    # TODO define within settings style?
    # TODO depending on generator properties and reeling speed
    settings['phase_efficiencies'] = {'traction': 0.9,
                                      'retraction': 1/0.65}
ref_height_sel = [10, 100]
ref_height = ref_height_sel[sel[2]]  # m

wind_speeds_sel = {10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                        13, 14, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19,
                        19.5, 20, 21],
                   100:  [3.5, 5, 8, 9, 10, 11, 14, 17, 20, 23, 25, 28],  #
                   # [3.5, 5, 8, 9, 10, 10.5, 11, 11.5, 12, 13, 14, 15, 16, 17, 18.5, 20, 21.5, 23, 23.5, 25, 26.5, 28]
                         #  [4, 5, 6, 7, 8, 9, 10, 11, 12,
                         #  13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                         #  23, 24, 25, 26, 27, 28, 29]
                   }  # m/s
wind_speeds = wind_speeds_sel[ref_height]

scan_tag = 'final_U_90_' + scan_tag  # short, high, final_ final_half_powering_stages  #less_cons_even_short_vw_I
# scan_tag = 'final_half_powering' + scan_tag
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
print(config.IO.plot_output)
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
    env_state.set_reference_wind_speed(1)
    heights = [10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160.,
                        180., 200., 220., 250., 300., 500., 600.]
    wind_speeds_at_h = [env_state.calculate_wind(h) for h in heights]
    print(heights, wind_speeds_at_h)
    # env_state.plot_wind_profile(color='#2ca02c')

    cm = 1/2.54
    fig = plt.figure(figsize=(5.5*cm, 7.5*cm))
    ax = fig.add_subplot(111)

    ax.set_ylabel("Height [m]")
    ax.set_xlim([-2.2, 3.2])
    plt.ylim([0, 600])
    ax.set_title("Log Profile")
    ax.grid(True)
    ax.set_xlabel('v [-]')

    ax.plot(wind_speeds_at_h, heights, '--',
            label='Magnitude', ms=3, color='#2ca02c')
    plt.tight_layout()
    plt.savefig(prod.config.IO.plot_output.format(
        title='log_wind_profile_shape'))

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

# pc.plot_optimization_results(oc.OPT_VARIABLE_LABELS,
#                              oc.bounds_real_scale,
#                              [sys_props['tether_force_min_limit'],
#                               sys_props['tether_force_max_limit']],
#                              [sys_props['reeling_speed_min_limit'],
#                               sys_props['reeling_speed_max_limit']],
#                              plot_info='_profile_log')

n_cwp = np.array([kpis['n_crosswind_patterns']
                  for kpis in pc.performance_indicators]
                 )[sel_succ]  # [sel_succ_power]
eff = [kpis['generator']['eff']['cycle']
       for kpis in pc.performance_indicators]
print('eff:', eff)
for i, e in enumerate(eff):
    if e is None:
        eff[i] = 0
eff = np.array(eff)[sel_succ]
p_eff = eff * p_cycle
print(eff, p_cycle)
print(p_eff)
x_opts = np.array(pc.x_opts)[sel_succ]  # [sel_succ_power]
if config.General.write_output:
    from AWERA.power_production.power_curves import export_to_csv
    export_to_csv(config, wind, wind[-1], p_eff, p_cycle, eff,
                  x_opts, n_cwp, 'log')


prod.compare_kpis([pc])
prod.plot_trajectories(pc)
prod.plot_power_curves(pcs=pc, labels='{}kW log profile'.format(rated_power),
                       lim=[wind[0], wind[-1]],
                       speed_at_op_height=True, plot_full_electrical=True)  # False)
prod.plot_power_curves(pcs=pc, labels='{}kW log profile'.format(rated_power),
                       lim=[wind[0], wind[-1]],
                       speed_at_op_height=False, plot_full_electrical=True)
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
