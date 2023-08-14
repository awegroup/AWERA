import matplotlib.pyplot as plt
import numpy as np
from qsm import LogProfile, SystemProperties, TractionPhase, Cycle

# Create environment object.
env_state = LogProfile()
env_state.set_reference_height(10)
env_state.set_reference_wind_speed(10)

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
sys_props = SystemProperties(sys_props)

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
        'time_step': .05,
        'azimuth_angle': 15. * np.pi / 180.,
        'course_angle': 110. * np.pi / 180.,
    },
    'transition': {
        'control': ('reeling_speed', 0.),
        'time_step': .05,
    },
    'traction': {
        'control': ('reeling_factor', .37),
        'time_step': .05,
    },
}
# pattern_settings = settings['traction']
# pattern_settings['tether_length'] = 100.
# pattern_settings['elevation_angle_ref'] = 25.*np.pi/180.
# cwp = EvaluatePattern(pattern_settings)
# cwp.calc_performance_along_pattern(sys_props, env_state, 100, print_details=True)
# # cwp.plot_traces((cwp.s, 'Normalised path distance [-]'), ('reeling_speed', 'power_ground', 'apparent_wind_speed', 'tether_force_ground', 'kite_tangential_speed'),
# #                 ('Reeling speed [m/s]', 'Power [W]', 'Apparent wind speed [m/s]', 'Tether force [N]', 'Tangential speed [m/s]'))
# cwp.plot_traces((cwp.s, 'Normalised path distance [-]'), ('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
#                 ('Radius [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
#                 (None, 180./np.pi, 180./np.pi, 180./np.pi))
# cwp.plot_traces((cwp.time, 'Time [s]'), ('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
#                 ('Radius [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
#                 (None, 180./np.pi, 180./np.pi, 180./np.pi))
# cwp.plot_pattern()

cycle = Cycle(settings)
cycle.follow_wind = True
cycle.run_simulation(sys_props, env_state, print_summary=True)
cycle.time_plot(('reeling_speed', 'power_ground', 'tether_force_ground'),
                ('Reeling\nspeed [m/s]', 'Power [W]', 'Tether\nforce [N]'))
# cycle.time_plot(('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
#                 ('Radius [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
#                 (None, 180./np.pi, 180./np.pi, 180./np.pi))
# cycle.time_plot(('straight_tether_length', 'reeling_speed', 'x', 'y', 'z'),
#                 ('r [m]', r'$\dot{\mathrm{r}}$ [m/s]', 'x [m]', 'y [m]', 'z [m]'))
cycle.trajectory_plot()
# cycle.trajectory_plot3d()
plt.show()