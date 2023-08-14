import numpy as np

# first 100kW implementation
sys_props_100 = {
    'kite_projected_area': 44.15,  # [m^2] - of ??? total flat area
    'kite_mass': 53,  # [kg] - kite + KCU
    'tether_density': 724.,  # [kg/m^3] - 0.85 GPa
    'tether_diameter': 0.014,  # [m]
    'tether_force_max_limit': 5000 * 9.81,  #5000/(.25*np.pi*0.004**2) * (.25*np.pi*0.014**2),  # ~61250 [N]
    'tether_force_min_limit': 500,  # [N]
    'kite_lift_coefficient_powered': 1.1,  # [-] - in the range of .9 - 1.0
    'kite_drag_coefficient_powered': .13,  # [-]
    'kite_lift_coefficient_depowered': .2,  # [-]
    'kite_drag_coefficient_depowered': .1,  # [-] - in the range of .1 - .2
    'reeling_speed_min_limit': 0.05,  # 2,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit_retr': 10,
    'reeling_speed_max_limit_trac': 5,
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
}

# Create pumping cycle simulation object, run simulation, and plot results.
settings_100 = {
    'cycle': {
        # THis settings does not do anything now - overwritten
        # 'traction_phase': TractionPhaseHybrid,
        'elevation_angle_traction': 35*np.pi/180.,
        # TODO improve this? Could go up to 500
        # TODO what does this setting do exacly?
        'tether_length_start_retraction': 300.,
        'tether_length_end_retraction': 150.,
    },
    'retraction': {
        # TODO retraction control -> x_opt parameter?
        'control': ('tether_force_ground', 10000),
        'time_step': .25,  # .05,

    },
    'transition': {
        'control': ('reeling_speed', 0.),
        'time_step': .25,  # .05,
    },
    'traction': {
        # TODO controlled reeling factor - does this have an effect?
        'control': ('reeling_factor', .37),
        'time_step': .25,  # .05,
        'azimuth_angle': 15. * np.pi / 180.,
        'course_angle': 110. * np.pi / 180.,
    },
}


# Kitepower 100kW
sys_props_kitepower_100 = {
    # TODO surface area
    'kite_projected_area': 78,  # 100m^2 flat area, estimated by reduction of 1.28,  # [m^2] - of ??? total flat area
    'kite_mass': 100 + 70,  # [kg] - kite + KCU
    'tether_density': 724.,  # [kg/m^3] - 0.85 GPa
    'tether_diameter': 0.014,  # [m]
    'tether_force_max_limit': 5000 * 9.81,  #5000/(.25*np.pi*0.004**2) * (.25*np.pi*0.014**2),  # ~61250 [N]
    'tether_force_min_limit': 500,  # [N]
    'kite_lift_coefficient_powered': 1.05,  # [-]
    'kite_drag_coefficient_powered': .13,  # [-]
    'kite_lift_coefficient_depowered': .2,  # [-]
    'kite_drag_coefficient_depowered': .1,  # [-] - in the range of .1 - .2
    'reeling_speed_min_limit': 0.05,  # 2,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit_retr': 9.3493,  # 10,
    'reeling_speed_max_limit_trac': 5,
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
}

# Create pumping cycle simulation object, run simulation, and plot results.
settings_kitepower_100 = {
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

# Kitepower 500kW
sys_props_kitepower_500 = {
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
settings_kitepower_500 = {
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


# all kites
kite_sys_props = {'100': sys_props_100,
                  'kitepower_100': sys_props_kitepower_100,
                  'kitepower_500': sys_props_kitepower_500}
kite_qsm_settings = {'100': settings_100,
                     'kitepower_100': settings_kitepower_100,
                     'kitepower_500': settings_kitepower_500}
