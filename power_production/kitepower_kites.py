from qsm import SystemProperties

# V2 - As presented in Van der Vlugt et al.
projected_area = 19.8  # [m^2]
max_wing_loading = 450  # [N/m^2]
min_wing_loading = 10  # [N/m^2] - min limit seems only active below cut-in
c_l_powered = .59
l_to_d_powered = 3.6  # [-]
c_l_depowered = .15
l_to_d_depowered = 3.5  # [-]
sys_props_v2 = {
    'kite_projected_area': projected_area,  # [m^2] - 25 m^2 total flat area
    'kite_mass': 19.6,  # [kg]
    'tether_density': 724.,  # [kg/m^3]
    'tether_diameter': 0.004,  # [m]
    'tether_force_max_limit': 10000,  # ~ max_wing_loading*projected_area [N]
    'tether_force_min_limit': 200,  # ~ min_wing_loading * projected_area [N]
    'kite_lift_coefficient_powered': c_l_powered,  # [-]
    'kite_drag_coefficient_powered': c_l_powered/l_to_d_powered,  # [-]
    'kite_lift_coefficient_depowered': c_l_depowered,  # [-]
    'kite_drag_coefficient_depowered': c_l_depowered/l_to_d_depowered,  # [-]
    'reeling_speed_min_limit': 0,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
}
sys_props_v2 = SystemProperties(sys_props_v2)

# Kitepower's V3.
projected_area = 19.75  # [m^2]
max_wing_loading = 500  # [N/m^2]
min_wing_loading = 10  # [N/m^2] - min limit seems only active below cut-in
c_l_powered = .9
l_to_d_powered = 4.2  # [-]
c_l_depowered = .2
l_to_d_depowered = 3.2  # [-]
sys_props_v3 = {
    'kite_projected_area': projected_area,  # [m^2] - 25 m^2 total flat area
    'kite_mass': 22.8,  # [kg] - 12 kg kite + 8 kg KCU
    'tether_density': 724.,  # [kg/m^3] - 0.85 GPa
    'tether_diameter': 0.004,  # [m]
    'tether_force_max_limit': 5000,  # ~ max_wing_loading*projected_area [N]
    'tether_force_min_limit': 300,  # ~ min_wing_loading * projected_area [N]
    'kite_lift_coefficient_powered': .9,  # [-] - in the range of .9 - 1.0
    'kite_drag_coefficient_powered': .2,  # [-]
    'kite_lift_coefficient_depowered': .2,  # [-]
    'kite_drag_coefficient_depowered': .1,  # [-] - in the range of .1 - .2
    'reeling_speed_min_limit': 2,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
}
sys_props_v3 = SystemProperties(sys_props_v3)
