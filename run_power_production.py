import matplotlib.pyplot as plt
import numpy as np

from AWERA.power_production.config import Config
from AWERA.power_production.power_production import PowerProduction

from AWERA.power_production.qsm import LogProfile

# Get configuration from config.yaml file in power production
config = Config()
print(config)
# Initialise power production functions class
prod = PowerProduction(config)

# Custom Profile with u, v, heights
# As an example use logarithmic profile as defined in qsm
heights = [10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160.,
           180., 200., 220., 250., 300., 500., 600.]
log_profile = LogProfile()
# Define absolute wind speed with logarithmic profile
u = np.array([log_profile.calculate_wind(height) for height in heights])
v = np.array([0]*len(heights))
# Convert to input profile format
profile = prod.as_input_profile(heights, u, v)
print(profile)

pc = prod.read_curve(i_profile=1, return_constructor=True)
# # Roughly estimate wind speed cut-in/out limits
# limit_estimates = prod.estimate_wind_speed_operational_limits(
#    input_profiles=profile)
# print(limit_estimates)
# # Run qsm simulation and optimisation for the full
# # range of wind speeds
# pcs, limits_refined = prod.make_power_curves(
#     input_profiles=profile,
#     wind_speed_limit_estimates=limit_estimates)
# # Only one profile:
# pc = pcs[0]
# Extract power curve from PowerCurveConstructor
wind_speed, mean_cycle_power = pc.curve()
print(wind_speed, mean_cycle_power)

# Plot power curve(s)
prod.plot_power_curves(pc)
plt.show()

# all in one, depending on config:
# pcs, limits_refined = run_curves(input_profiles=profiles)
# if config no production:
# limit_estimates = run_curves(input_profiles=profiles)


# Single profile optimization run
from AWERA.power_production.power_production import SingleProduction

prod_single = SingleProduction(ref_height=config.General.ref_height)

x0_opt, x_opt, op_res, cons, kpis = prod_single.single_profile_power(
    heights, u, v, x0=[4100., 850., 0.5, 240., 200.0], ref_wind_speed=1)

# x0 and x_opt:
# OPT_VARIABLE_LABELS = [
#        "Reel-out\nforce [N]",
#        "Reel-in\nforce [N]",
#        "Elevation\nangle [rad]",
#        "Reel-in tether\nlength [m]",
#        "Minimum tether\nlength [m]"
#    ]

# op_res: optimizer output
# cons: optimizer constraints
# kpis: kite performance indicators
avg_cycle_power = kpis['average_power']['cycle']
print(avg_cycle_power)
