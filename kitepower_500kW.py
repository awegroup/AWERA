import numpy as np

# Kitepower 500kW
kite_name = 'kitepower_500kW'

sys_props = {
    # TODO surface area
    # TODO tether settings
    'kite_projected_area': 199,  # 250m^2 flat area, estimated by reduction of 1.28 (Kitepower Falcon: 60 -> 47) ,  # [m^2]
    'kite_mass': 300 + 70,  # [kg] - kite + KCU
    'tether_density': 724.,  # [kg/m^3] - 0.85 GPa
    'tether_diameter': np.sqrt(5) * 0.014,  # [m]
    'tether_force_max_limit': 5 * 5000 * 9.81,  #5000/(.25*np.pi*0.004**2) * (.25*np.pi*0.014**2),  # ~61250 [N]
    'tether_force_min_limit': 500,  # [N]
    'kite_lift_coefficient_powered': 1.1,  # [-]
    'kite_drag_coefficient_powered': .12,  # [-]
    'kite_lift_coefficient_depowered': .2,  # [-]
    'kite_drag_coefficient_depowered': .1,  # [-] - in the range of .1 - .2
    'reeling_speed_min_limit': 0.05,  # 2,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit_retr': 10,
    'reeling_speed_max_limit_trac': 5,
    # 'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
}

# Create pumping cycle simulation object, run simulation, and plot results.
settings = {
    'cycle': {
        # 'traction_phase': TractionPhase,
        'elevation_angle_traction': 35*np.pi/180.,
        # TODO improve this? Could go up to 500
        # TODO what does this setting do exacly?
        'tether_length_start_retraction': 500.,
        'tether_length_end_retraction': 200.,
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

x0 = [25000., 1000., 0.5, 230., 200.0]

