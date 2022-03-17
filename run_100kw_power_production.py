import matplotlib.pyplot as plt
import numpy as np
import pickle

from AWERA.power_production.config import Config
from AWERA.power_production.power_production import PowerProduction

from AWERA.power_production.qsm import LogProfile, TractionPhase

# Get configuration from config.yaml file in power production
config = Config()
config.update({'Power':{'profile_type': '100kW_log_profile'}})
print(config)

# Initialise power production functions class
prod = PowerProduction(config)

sys_props = {
    'kite_projected_area': 44.15,  # [m^2] - 25 m^2 total flat area
    'kite_mass': 53,  # [kg] - 12 kg kite + 8 kg KCU
    'tether_density': 724.,  # [kg/m^3] - 0.85 GPa
    'tether_diameter': 0.014,  # [m]
    'tether_force_max_limit': 5000 * 9.81,  #5000/(.25*np.pi*0.004**2) * (.25*np.pi*0.014**2),  # ~61250 [N]
    'tether_force_min_limit': 500,  # [N]
    'kite_lift_coefficient_powered': 1.1,  # [-] - in the range of .9 - 1.0
    'kite_drag_coefficient_powered': .13,  # [-]
    'kite_lift_coefficient_depowered': .2,  # [-]
    'kite_drag_coefficient_depowered': .1,  # [-] - in the range of .1 - .2
    'reeling_speed_min_limit': 2,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
}

# Create pumping cycle simulation object, run simulation, and plot results.
settings = {
    'cycle': {
        'traction_phase': TractionPhase,
        'elevation_angle_traction': 35*np.pi/180.,
        'tether_length_start_retraction': 300.,
        'tether_length_end_retraction': 150.,
    },
    'retraction': {
        'control': ('tether_force_ground', 10000),
        'time_step': .25,  # .05,

    },
    'transition': {
        'control': ('reeling_speed', 0.),
        'time_step': .25,  # .05,
    },
    'traction': {
        'control': ('reeling_factor', .37),
        'time_step': .25,  # .05,
        'azimuth_angle': 15. * np.pi / 180.,
        'course_angle': 110. * np.pi / 180.,
    },
}

# cycle = Cycle(settings)
# cycle.follow_wind = True
# cycle.run_simulation(sys_props, env_state, print_summary=True)
# cycle.time_plot(('reeling_speed', 'power_ground', 'tether_force_ground'),
#                 ('Reeling\nspeed [m/s]', 'Power [W]', 'Tether\nforce [N]'))

# cycle.trajectory_plot()
# cycle.trajectory_plot3d()
# plt.show()
ref_height = 10  # m
wind_speeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21]  # m/s
ref_wind_speed = wind_speeds[0]

pc, oc = prod.single_power_curve(
    wind_speeds,
    x0=[8000., 1000., 0.5, 240., 150.0],
    sys_props=sys_props,
    cycle_sim_settings=settings,
    plot_output_file='{title}_100kw_log_profile_opt.pdf',
    env_state=None,  # Defaults to Log Profile
    ref_height=ref_height,
    ref_wind_speed=ref_wind_speed,
    return_optimizer=True)
wind_speed, mean_cycle_power = pc.curve()
print(wind_speed, mean_cycle_power)

pc.export_results(config.IO.power_curve.format(i_profile='10m_log', suffix='pickle'))

# mask failed simulation, export only good results
sel_succ = [kpis['sim_successful']
            for kpis in pc.performance_indicators]
print('# Failed runs: ', -sum(sel_succ)+len(sel_succ))
print('# Successful runs: ', sum(sel_succ))

# Plot power curve together with that of the other wind profile shapes.
p_cycle = np.array([kpis['average_power']['cycle']
                    for kpis in pc.performance_indicators])[sel_succ]
print('p_cycle: ', p_cycle)
# Log? print('p_cycle: ', p_cycle)
# Mask negative jumps in power
sel_succ_power = p_cycle > 0
# TODO  - check reason for negative power/ strong jumps
if sum(sel_succ_power) != len(sel_succ_power):
    print(len(sel_succ_power)-sum(sel_succ_power),
          'masked negative powers')
p_cycle_masked = p_cycle[sel_succ_power]
print('masked p_cycle: ', p_cycle_masked)
# TODO resolve source of problems - done right? leave in as check
while True:
    sel_succ_power_disc = [True] + list(np.diff(p_cycle_masked)
                                        > -1000)
    print('No disc: ', sel_succ_power_disc)
    sel_succ_power[sel_succ_power] = sel_succ_power_disc
    # p_cycle_masked = p_cycle_masked[sel_succ_power_disc]
    # if sum(sel_succ_power_disc) == len(sel_succ_power_disc):
    #    # No more discontinuities
    #       break
    # TODO this leads to errors sometimes:
    # IndexError: boolean index did not match indexed array along
    # dimension 0; dimension is 0
    # but corresponding boolean dimension is 1
    break
    print(len(sel_succ_power_disc)-sum(sel_succ_power_disc),
          'masked power discontinuities')

p_cycle = p_cycle[sel_succ_power]
wind = pc.wind_speeds[sel_succ_power]

pc.plot_optimal_trajectories(plot_info='_profile_{}'.format('10m_log'))

pc.plot_optimization_results(oc.OPT_VARIABLE_LABELS,
                             oc.bounds_real_scale,
                             [sys_props['tether_force_min_limit'],
                              sys_props['tether_force_max_limit']],
                             [sys_props['reeling_speed_min_limit'],
                              sys_props['reeling_speed_max_limit']],
                             plot_info='_profile_{}'.format('10m_log'))

n_cwp = np.array([kpis['n_crosswind_patterns']
                  for kpis in pc.performance_indicators]
                 )[sel_succ][sel_succ_power]
x_opts = np.array(pc.x_opts)[sel_succ][sel_succ_power]
if config.General.write_output:
    from AWERA.power_production.power_curves import export_to_csv
    export_to_csv(config, wind, wind[-1], p_cycle,
                  x_opts, n_cwp, '10m_log')


prod.compare_kpis([pc])
prod.plot_trajectories(pc)
prod.plot_power_curves(pcs=pc, labels='100kW log profile',
                       lim=[3, 21])
with open('power_curve_constructor_100kw_10m_log_profile_opt.pickle', 'wb')\
        as f:
    pickle.dump(pc, f)
print('Power Curve output written.')
wind_speed, mean_cycle_power = pc.curve()


# Plot power curve(s)
prod.plot_power_curves(pc)
plt.show()

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
