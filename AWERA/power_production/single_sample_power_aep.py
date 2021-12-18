from pyoptsparse import Optimization, SLSQP, Gradient
from pyoptsparse import History
from scipy import optimize as op
import numpy as np
import matplotlib as mpl
#mpl.use('Pdf')
#TODO include config
import matplotlib.pyplot as plt

from .cycle_optimizer import OptimizerCycle

# TODO include? from .config import optimizer_history_file_name

# TODO update, use this y/n?

def test():
    from .qsm import LogProfile, TractionPhaseHybrid
    from .kitepower_kites import sys_props_v3

    import time
    since = time.time()
    env_state = LogProfile() #include env stat with read in normalised wind profile
    env_state.set_reference_wind_speed(12.) # & ref wind speed set to scaling

    cycle_sim_settings = { #read in from optimizer output?
        'cycle': {
            'traction_phase': TractionPhaseHybrid,
            'include_transition_energy': False,
        },
        'retraction': {},
        'transition': {
            'time_step': 0.25,
        },
        'traction': {
            'azimuth_angle': 13 * np.pi / 180.,
            'course_angle': 100 * np.pi / 180.,
        },
    }
    oc = OptimizerCycle(cycle_sim_settings, sys_props_v3, env_state, reduce_x=np.array([0, 1, 2, 3]))
    oc.x0_real_scale = np.array([4500, 1000, 30*np.pi/180., 150, 230]) #read in from optimizer output
    print('Optimization on:', oc.x0_real_scale)
    #print(oc.optimize()) no optimization for now
    oc.eval_point(plot_results=True)
    time_elapsed = time.time() - since
    print('Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #optHist = History(optimizer_history_file_name)
    #print(optHist.getValues(major=True, scale=False, stack=False, allowSens=True)['isMajor'])

    plt.show()
