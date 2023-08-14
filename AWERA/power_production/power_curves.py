import os
import numpy as np
import copy
import importlib

import pandas as pd
from copy import deepcopy

import sys
import getopt
from .qsm import LogProfile, NormalisedWindTable1D, KiteKinematics,\
    SteadyState, TractionPhaseHybrid, TractionConstantElevation, \
    SteadyStateError, OperationalLimitViolation, TractionPhase, \
    SystemProperties
from .kitepower_kites import sys_props_v3
from .cycle_optimizer import OptimizerCycle
from .power_curve_constructor import PowerCurveConstructor
from ..utils.convenience_utils import write_timing_info

import matplotlib.pyplot as plt

# Assumptions representative reel-out state at cut-in wind speed.
theta_ro_ci = 25 * np.pi / 180.
phi_ro = 13 * np.pi / 180.
chi_ro = 100 * np.pi / 180.

l0 = 200  # Tether length at start of reel-out.
l1_lb = 350  # Lower bound of tether length at end of reel-out.
l1_ub = 450  # Upper bound of tether length at end of reel-out.


def read_system_settings(config):
    cycle_sim_settings_pc_phase1, cycle_sim_settings_pc_phase2, \
        sys_props, x0 = None, None, None, None
    if config.Power.kite_and_QSM_settings_file is not None:  # add to config, abs or relative path?
        settings_mod = importlib.import_module(
            config.Power.kite_and_QSM_settings_file)
        try:
            cycle_sim_settings_pc_phase1 = settings_mod.settings
            phase_1_settings_imported = True
        except AttributeError:
            phase_1_settings_imported = False
        try:
            cycle_sim_settings_pc_phase2 = settings_mod.settings_phase_2
        except AttributeError:
            if phase_1_settings_imported:
                cycle_sim_settings_pc_phase2 = deepcopy(
                    cycle_sim_settings_pc_phase1)

        try:
            sys_props = settings_mod.sys_props
            if not isinstance(sys_props, SystemProperties):
                sys_props = SystemProperties(sys_props)
        except AttributeError:
            pass
        try:
            x0 = settings_mod.x0
        except AttributeError:
            pass
    return cycle_sim_settings_pc_phase1, cycle_sim_settings_pc_phase2, \
        sys_props, x0


def calc_tether_force_traction(env_state, straight_tether_length, sys_props):
    """Calculate tether force for the minimum allowable reel-out speed
    and given wind conditions and tether length."""
    kinematics = KiteKinematics(straight_tether_length,
                                phi_ro, theta_ro_ci, chi_ro)
    env_state.calculate(kinematics.z)
    sys_props.update(kinematics.straight_tether_length, True)
    ss = SteadyState({'enable_steady_state_errors': True})
    ss.control_settings = ('reeling_speed',
                           sys_props.reeling_speed_min_limit)
    ss.find_state(sys_props, env_state, kinematics)
    return ss.tether_force_ground


def get_cut_in_wind_speed(env, sys_props):
    """Iteratively determine lowest wind speed for which, along the entire
    reel-out path, feasible steady flight states
    with the minimum allowable reel-out speed are found."""
    dv = 1e-2  # Step size [m/s].
    v0 = 2  # Lowest wind speed [m/s] with which the iteration is started.

    v = v0
    rescan_finer_steps = False
    while True:
        env.set_reference_wind_speed(v)
        try:
            # Setting tether force as setpoint in qsm yields infeasible region
            tether_force_start = calc_tether_force_traction(env, l0, sys_props)
            tether_force_end = calc_tether_force_traction(env, l1_lb, sys_props)

            start_critical = tether_force_end > tether_force_start
            if start_critical:
                critical_force = tether_force_start
            else:
                critical_force = tether_force_end

            if tether_force_start > sys_props.tether_force_min_limit and \
                    tether_force_end > sys_props.tether_force_min_limit:
                if v == v0:
                    raise ValueError("Starting speed is too high.")
                if rescan_finer_steps:
                    return v, start_critical, critical_force
                v = v - dv  # Reset to last working wind speed
                dv = dv/10  # Scan with finer binning
                rescan_finer_steps = True
        except SteadyStateError:
            pass

        v += dv


def calc_n_cw_patterns(env, sys_props, theta=60. * np.pi / 180.):
    """Calculate the number of cross-wind manoeuvres flown."""
    trac = TractionPhaseHybrid({
            'control': ('tether_force_ground',
                        sys_props.tether_force_max_limit),
            'azimuth_angle': phi_ro,
            'course_angle': chi_ro,
        })
    trac.enable_limit_violation_error = True

    # Start and stop conditions of traction phase. Note that the traction
    # phase uses an azimuth angle in contrast to
    # the other phases, which results in jumps of the kite position.
    trac.tether_length_start = l0
    trac.tether_length_start_aim = l0
    trac.elevation_angle = TractionConstantElevation(theta)
    trac.tether_length_end = l1_ub
    trac.finalize_start_and_end_kite_obj()
    trac.run_simulation(sys_props, env,
                        {'enable_steady_state_errors': True})

    return trac.n_crosswind_patterns


def get_max_wind_speed_at_elevation(sys_props,
                                    env=LogProfile(),
                                    theta=60. * np.pi / 180.):
    """Iteratively determine maximum wind speed allowing at least one
    cross-wind manoeuvre during the reel-out phase for
    provided elevation angle."""
    dv = 0.2  # Step size [m/s].
    v0 = 22.  # Lowest wind speed [m/s] with which the iteration is started.
    # Check if the starting wind speed gives a feasible solution.
    env.set_reference_wind_speed(v0)
    try:
        n_cw_patterns = calc_n_cw_patterns(env, sys_props, theta)
    except (SteadyStateError, OperationalLimitViolation) as e:
        if e.code == 8:
            pass
        else:
            try:
                # Iteration fails for low wind speed
                # -> try very low and check if it works then
                v0 = v0 * 0.8
                print('second test: ', v0)
                # Check if the starting wind speed gives a feasible solution.
                env.set_reference_wind_speed(v0)
                n_cw_patterns = calc_n_cw_patterns(env, sys_props, theta)
            except (SteadyStateError, OperationalLimitViolation) as e:
                if e.code == 8:
                    pass
                else:
                    try:

                        # Iteration fails for low wind speed
                        # -> try very low and check if it works then
                        v0 = v0 * 0.8
                        print('third test: ', v0)
                        # Check if the starting wind speed gives
                        # a feasible solution.
                        env.set_reference_wind_speed(v0)
                        n_cw_patterns = calc_n_cw_patterns(env, sys_props, theta)
                    except (SteadyStateError,
                            OperationalLimitViolation) as e:
                        try:

                            # Iteration fails for low wind speed
                            # -> try very low and check if it works then
                            v0 = v0 * 0.8
                            print('fourth test: ', v0)
                            # Check if the starting wind speed gives
                            # a feasible solution.
                            env.set_reference_wind_speed(v0)
                            n_cw_patterns = calc_n_cw_patterns(env, sys_props, theta)
                        except (SteadyStateError,
                                OperationalLimitViolation) as e:
                            if e.code != 8:
                                print("No feasible solution found for"
                                      " first assessed cut out wind speed.")
                                raise e
    # Increase wind speed until number of cross-wind manoeuvres subceeds one.
    v = v0 + dv

    rescan_finer_steps = False
    fail_counter = 0
    while True:
        env.set_reference_wind_speed(v)
        # TODO output? print('velocity: ', v, 'rescan: ', rescan_finer_steps,
        #       'fail_couter: ', fail_counter)
        try:
            n_cw_patterns = calc_n_cw_patterns(env, sys_props, theta)
            fail_counter = 0  # Simulation worked -> reset fail counter
            if n_cw_patterns < 1.:  # No full crosswind pattern flown - cut out
                if rescan_finer_steps:
                    return v
                v = v - dv  # Reset to last working wind speed
                dv = dv/10  # Scan with finer binning
                rescan_finer_steps = True
        except OperationalLimitViolation:
            if rescan_finer_steps:
                return v
            v = v - dv  # Reset to last working wind speed
            dv = dv/10  # Scan with finer binning

            rescan_finer_steps = True

        except SteadyStateError as e:
            if e.code != 8:  # Speed is too low to yield a solution.
                if fail_counter < 9:
                    v = v - dv - dv/10  # Vary wind speed backwards slightly
                else:
                    # Up to previously working velocity no solution found
                    v = v - dv/10  # -> set last working wind speed as cut out
                    return v
                fail_counter += 1

        if v > 40.:
            raise ValueError("Iteration did not find feasible cut-out speed.")
        v += dv


def get_cut_out_wind_speed(sys_props, env=LogProfile()):
    """In general, the elevation angle is increased with wind speed as a last
    means of de-powering the kite. In that case, the wind speed at which the
    elevation angle reaches its upper limit is the cut-out wind speed. This
    procedure verifies if this is indeed the case. Iteratively the elevation
    angle is determined giving the highest wind speed allowing at least one
    cross-wind manoeuvre during the reel-out phase."""
    beta = 60*np.pi/180.
    dbeta = 1*np.pi/180.
    vw_last = 0.
    while True:
        vw = get_max_wind_speed_at_elevation(sys_props, env, beta)
        if vw is not None:
            if vw <= vw_last:
                return vw_last, beta+dbeta
            vw_last = vw
        beta -= dbeta


def export_to_csv(config, v, v_cut_out, p, p_mech, eff,
                  x_opts, n_cwp, i_profile):
    df = {
        'v_100m [m/s]': v,
        'v/v_cut-out [-]': np.array(v)/v_cut_out,
        'P [W]': p,
        'P mech [W]': p_mech,
        'eff [-]': eff,
        'F_out [N]': [x[0] for x in x_opts],
        'F_in [N]': [x[1] for x in x_opts],
        'theta_out [rad]': [x[2] for x in x_opts],
        'dl_tether [m]': [x[3] for x in x_opts],
        'l0_tether [m]': [x[4] for x in x_opts],
        'n_crosswind_patterns [-]': n_cwp,
    }
    df = pd.DataFrame(df)
    df.to_csv(config.IO.power_curve.format(i_profile=i_profile,
                                           suffix='csv'),
              index=False, sep=";")


def create_environment(df, i_profile):
    """Flatten wind profile shapes resulting from the clustering and use
        to create the environment object."""
    env = NormalisedWindTable1D()
    # velocities w.r.t. env.h_ref = 100.
    env.heights = list(df['height [m]'])
    env.normalised_wind_speeds = list((df['u{} [-]'.format(i_profile)]**2
                                       + df['v{} [-]'.format(i_profile)]**2
                                       )**.5)
    return env


def estimate_wind_speed_operational_limits(config,
                                           export_operational_limits=True,
                                           input_profiles=None):
    """Estimate the cut-in and cut-out wind speeds for each wind profile shape.

    These wind speeds are refined when determining the power curves.
    """
    # TODO include descrition of inpuf profiles
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 3), sharey=True)
    plt.subplots_adjust(top=0.92, bottom=0.164, left=0.11,
                        right=0.788, wspace=0.13)

    # TODO format with reference height everywhere
    res = {'vw_100m_cut_in': [],
           'vw_100m_cut_out': [],
           'tether_force_cut_in': [],
           }
    if input_profiles is None:
        input_profiles = pd.read_csv(config.IO.profiles, sep=";")
    # 1 height column, 3 columns each profile (u,v,scale factor)
    # TODO remove scale factor?
    n_profiles = int((input_profiles.shape[1]-1)/3)
    # TODO option to read arbitrary profile, n_prifles: len(profiles)
    for i_profile in range(1, n_profiles+1):
        print('Estimating wind speed for profile {}/{}'
              .format(i_profile, n_profiles))
        # TODO logging? / timing info print('Profile {}'.format(i_profile))
        env = create_environment(input_profiles, i_profile)
        print(env.heights, env.normalised_wind_speeds)
        # heights = [10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160.,
        #                     180., 200., 220., 250., 300., 500., 600.]
        # wind_speeds_at_h = [env.calculate_wind(h) for h in heights]
        # print(heights, wind_speeds_at_h)
        # continue
        # Get cut-in wind speed.
        sys_props = sys_props_v3
        _, _, read_sys_props, read_x0 = read_system_settings(config)
        if read_sys_props is not None:
            sys_props = read_sys_props
        # vw_cut_in, _, tether_force_cut_in = get_cut_in_wind_speed(env, sys_props)
        vw_cut_in, tether_force_cut_in = 3, 500
        res['vw_100m_cut_in'].append(vw_cut_in)
        res['tether_force_cut_in'].append(tether_force_cut_in)

        # Get cut-out wind speed
        # vw_cut_out, elev = get_cut_out_wind_speed(sys_props, env)
        vw_cut_out, elev = 30, 1.1
        # TODO remove
        # if vw_cut_out > 27:
        #     vw_cut_out = 27

        res['vw_100m_cut_out'].append(vw_cut_out)

        # Plot the wind profiles corresponding to the wind speed operational
        # limits and the profile shapes.
        env.set_reference_height(
            config.General.ref_height)
        env.set_reference_wind_speed(vw_cut_in)
        plt.sca(ax[0])
        env.plot_wind_profile()

        env.set_reference_wind_speed(vw_cut_out)
        plt.sca(ax[1])
        env.plot_wind_profile("{}".format(i_profile))
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.ylabel('')

    df = pd.DataFrame(res)
    # TODO log? print(df)

    if export_operational_limits:
        df.to_csv(config.IO.cut_wind_speeds)

    ax[0].set_title("Cut-in")
    ax[0].set_xlim([0, None])
    ax[0].set_ylim([0, 400])
    ax[1].set_title("Cut-out")
    ax[1].set_xlim([0, None])
    ax[1].set_ylim([0, 400])
    if not config.Plotting.plots_interactive:
        plt.savefig(config.IO.training_plot_output.format(
            title='estimate_wind_speed_operational_limits'))
    return df


def generate_power_curves(config,
                          run_profiles,
                          input_profiles=None,
                          limit_estimates=None):
    """Determine power curves - requires estimates of the cut-in
        and cut-out wind speed to be available."""
    if limit_estimates is None:
        limit_estimates = pd.read_csv(config.IO.cut_wind_speeds)

    # Cycle simulation settings for different phases of the power curves.
    cycle_sim_settings_pc_phase1 = {
        'cycle': {
            'traction_phase': TractionPhase,
            'include_transition_energy': False,
        },
        'retraction': {},
        'transition': {
            'time_step': 0.25,
        },
        'traction': {
            'azimuth_angle': phi_ro,
            'course_angle': chi_ro,
        },
    }

    cycle_sim_settings_pc_phase2 = deepcopy(cycle_sim_settings_pc_phase1)
    cycle_sim_settings_pc_phase2['cycle']['traction_phase'] = \
        TractionPhaseHybrid

    # TODO make kitepower_3 default settings file
    sys_props = sys_props_v3

    read_cycle_sim_settings_pc_phase1, read_cycle_sim_settings_pc_phase2, \
        read_sys_props, read_x0 = read_system_settings(config)
    if read_cycle_sim_settings_pc_phase1 is not None:
        cycle_sim_settings_pc_phase1 = read_cycle_sim_settings_pc_phase1
    if read_cycle_sim_settings_pc_phase2 is not None:
        cycle_sim_settings_pc_phase1 = read_cycle_sim_settings_pc_phase2
    if read_sys_props is not None:
        sys_props = read_sys_props
    if read_x0 is not None:
        x0 = read_x0

    fig, ax_pcs = plt.subplots(2, 1)
    for a in ax_pcs:
        a.grid()

    limits_refined = {'i_profile': [],
                      'vw_100m_cut_in': [],
                      'vw_100m_cut_out': []}
    res_pcs = []
    if input_profiles is None:
        input_profiles = pd.read_csv(config.IO.profiles, sep=";")
    for i_profile in run_profiles:
        # TODO log? print("Power curve generation for profile number {}"
        # .format(i_profile))
        # Pre-configure environment object for optimizations by setting
        # normalized wind profile.
        env = create_environment(input_profiles, i_profile)

        # Optimizations are performed sequentially with increased wind speed.
        # The solution of the previous optimization
        # is used to initialise the next. With trial and error the lower
        # configuration, a reasonably robust approach is
        # obtained. The power curves may however still exhibit discontinuities
        # and therefore need to be checked and
        # possibly post-processed.

        # The optimization incessantly fails for the estimated cut-out wind
        # speed. Therefore, the highest wind speed for
        # which the optimization is performed is somewhat lower than the
        # estimated cut-out wind speed.
        vw_cut_in = limit_estimates.iloc[i_profile-1]['vw_100m_cut_in']
        vw_cut_out = limit_estimates.iloc[i_profile-1]['vw_100m_cut_out']
        # wind_speeds = np.linspace(vw_cut_in, vw_cut_out, 20)
        # wind_speeds = np.linspace(3, 29, 70)
        wind_speeds = np.array([3.5, 5, 8, 9, 10, 11, 14, 17, 20, 23, 25, 28])  # final_U_
        # wind_speeds = np.array([3.5, 5, 8, 9, 10, 10.5, 11, 11.5, 12, 13, 14, 15, 15.5, 17, 18.5, 20, 21.5, 23, 23.5, 25, 26.5, 28])  # final_U_X1
        # wind_speeds = np.linspace(3, 29, 15)
        # wind_speeds = np.linspace(vw_cut_in, vw_cut_out-1, 50)
        # wind_speeds = np.concatenate((wind_speeds,
        #                               np.linspace(vw_cut_out-1,
        #                                           vw_cut_out-0.05, 15)))

        # For the first phase of the power curve, the constraint on the number
        # of cross-wind patterns flown is not
        # active. It is assumed that sufficient cross-wind patterns are flown
        # up to vw_100m = 7 m/s (verify this).
        # Therefore, the number of cross-wind patterns is not calculated for
        # this phase. Also the upper elevation bound
        # is set to 30 degrees.
        op_cycle_pc_phase1 = OptimizerCycle(
            deepcopy(cycle_sim_settings_pc_phase1),
            sys_props, env,
            reduce_x=np.array([0, 1, 2, 3]),
            bounds=copy.deepcopy(
                config.Power.bounds))
        op_cycle_pc_phase1.bounds_real_scale[2][1] = 30*np.pi/180.

        op_cycle_pc_phase2 = OptimizerCycle(
            deepcopy(cycle_sim_settings_pc_phase2),
            sys_props, env,
            reduce_x=np.array([0, 1, 2, 3]),
            bounds=copy.deepcopy(
                config.Power.bounds))

        # Configuration of the sequential optimizations for which is
        # differentiated between the wind speed ranges
        # bounded above by the wind speed of the dictionary key. If dx0
        # does not contain only zeros, the starting point
        # of the new optimization is not the solution of the
        # preceding optimization.
        op_seq = {
            7.: {'power_optimizer': op_cycle_pc_phase1,
                 'dx0': np.array([0., 0., 0., 0.1, 0., 0.])},
            17.: {'power_optimizer': op_cycle_pc_phase2,
                  'dx0': np.array([0., 0., 0., 0.2, 0., 0.])},
            np.inf: {'power_optimizer': op_cycle_pc_phase2,
                     'dx0': np.array([0., 0., 0.1, 0.2, 0., 0.])},
            # Convergence for
            # profiles 2 and 6 are sensitive to starting elevation.
            # The number of patterns constraint exhibits a
            # minimum along the feasible elevation angle range. When the
            # elevation angle of the starting point is lower
            # than that of the minimum, the optimizer is driven towards lower
            # elevation angles which do not yield a
            # feasible solution.
        }

        # Optmise depowering at end of traction:
        op_cycle_pc_phase1_powering = OptimizerCycle(
            cycle_sim_settings_pc_phase1,
            sys_props, env,
            reduce_x=np.array([0, 1, 3, 5]),
            bounds=copy.deepcopy(
                config.Power.bounds))
        op_cycle_pc_phase1_powering.bounds_real_scale[2][1] = 30*np.pi/180.

        op_cycle_pc_phase2_powering = OptimizerCycle(
            cycle_sim_settings_pc_phase2,
            sys_props, env,
            reduce_x=np.array([0, 1, 3, 5]),
            bounds=copy.deepcopy(
                config.Power.bounds))

        op_seq_powering = {
            7.: {'power_optimizer': op_cycle_pc_phase1_powering,
                 'dx0': np.array([0., 0., 0., 0.1, 0., 0.])},
            17.: {'power_optimizer': op_cycle_pc_phase2_powering,
                  'dx0': np.array([0., 0., 0., 0.2, 0., 0.])},
            np.inf: {'power_optimizer': op_cycle_pc_phase2_powering,
                     'dx0': np.array([0., 0., 0.1, 0.2, 0., 0.])},
            # Convergence for
            # profiles 2 and 6 are sensitive to starting elevation.
            # The number of patterns constraint exhibits a
            # minimum along the feasible elevation angle range. When the
            # elevation angle of the starting point is lower
            # than that of the minimum, the optimizer is driven towards lower
            # elevation angles which do not yield a
            # feasible solution.
        }

        # Define starting point for the very first optimization at
        # the cut-in wind speed.
        critical_force = limit_estimates.iloc[i_profile-1][
            'tether_force_cut_in']
        if read_x0 is None:
            x0 = np.array([critical_force, 300., theta_ro_ci, 150., 200.0, 1])
        else:
            x0 = read_x0
        # Start optimizations.
        pc = PowerCurveConstructor(wind_speeds)
        setattr(pc, 'plots_interactive', config.Plotting.plots_interactive)
        setattr(pc, 'plot_output_file', config.IO.training_plot_output)
        pc.run_predefined_sequence(op_seq, x0, depowering_seq=op_seq_powering)

        # export all results, including failed simulations, tagged in
        # kip and other performance flags
        pc.export_results(config.IO.power_curve.format(
            i_profile=i_profile, suffix='pickle'))
        res_pcs.append(pc)

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
        # sel_succ_power = p_cycle > 0
        # # TODO  - check reason for negative power/ strong jumps
        # if sum(sel_succ_power) != len(sel_succ_power):
        #     print(len(sel_succ_power)-sum(sel_succ_power),
        #           'masked negative powers')
        # p_cycle_masked = p_cycle[sel_succ_power]
        # print('masked p_cycle: ', p_cycle_masked)
        # # TODO resolve source of problems - done right? leave in as check
        # while True:
        #     sel_succ_power_disc = [True] + list(np.diff(p_cycle_masked)
        #                                         > -1000)
        #     print('No disc: ', sel_succ_power_disc)
        #     sel_succ_power[sel_succ_power] = sel_succ_power_disc
        #     # p_cycle_masked = p_cycle_masked[sel_succ_power_disc]
        #     # if sum(sel_succ_power_disc) == len(sel_succ_power_disc):
        #     #    # No more discontinuities
        #     #       break
        #     # TODO this leads to errors sometimes:
        #     # IndexError: boolean index did not match indexed array along
        #     # dimension 0; dimension is 0
        #     # but corresponding boolean dimension is 1
        #     break
        #     print(len(sel_succ_power_disc)-sum(sel_succ_power_disc),
        #           'masked power discontinuities')

        # p_cycle = p_cycle # [sel_succ_power]
        wind = pc.wind_speeds  # [sel_succ_power]

        ax_pcs[0].plot(wind, p_cycle/1000, label=i_profile)
        ax_pcs[1].plot(wind/vw_cut_out, p_cycle/1000, label=i_profile)

        pc.plot_optimal_trajectories(
            plot_info='_profile_{}'.format(i_profile),
            circle_radius=op_cycle_pc_phase2.bounds_real_scale[4][0])
        pc.plot_optimization_results(op_cycle_pc_phase2.OPT_VARIABLE_LABELS,
                                     op_cycle_pc_phase2.bounds_real_scale,
                                     [sys_props.tether_force_min_limit,
                                      sys_props.tether_force_max_limit],
                                     [sys_props.reeling_speed_min_limit,
                                      sys_props.reeling_speed_max_limit],
                                     plot_info='_profile_{}'.format(i_profile))

        n_cwp = np.array([kpis['n_crosswind_patterns']
                          for kpis in pc.performance_indicators]
                         )[sel_succ]  # [sel_succ_power]
        eff = [kpis['generator']['eff']['cycle']
               for kpis in pc.performance_indicators]
        for i, e in enumerate(eff):
            if e is None:
                eff[i] = 0
        eff = np.array(eff)[sel_succ]
        p_eff = eff * p_cycle
        x_opts = np.array(pc.x_opts)[sel_succ]  # [sel_succ_power]
        if config.General.write_output:
            export_to_csv(config, wind, vw_cut_out, p_eff, p_cycle, eff,
                          x_opts, n_cwp, i_profile)
        # Refine the wind speed operational limits to wind speeds for
        # which optimal solutions are found.
        limits_refined['i_profile'].append(i_profile)
        limits_refined['vw_100m_cut_in'].append(wind[0])
        limits_refined['vw_100m_cut_out'].append(wind[-1])

        # TODO in log print("Cut-in and -out speeds changed from [{:.3f},
        # {:.3f}] to "
        #      "[{:.3f}, {:.3f}].".format(vw_cut_in, vw_cut_out,
        #                                 wind[0],
        #                                 wind[-1]))
        if len(run_profiles) == 1 and config.General.write_output:
            df = pd.DataFrame(limits_refined)
            # TODO log? print(df)
            df.to_csv(config.IO.refined_cut_wind_speeds.replace(
                '.csv', '_profile_{}.csv'.format(i_profile)))
            # TODO include this change in config?
            # TODO log? print("Exporting single profile operational limits.")
    ax_pcs[1].legend()
    x_label = '$v_{w,' + str(config.General.ref_height) + 'm}$ [m/s]'
    ax_pcs[0].set_xlabel(x_label)
    ax_pcs[1].set_xlabel('$v_{w,' + str(config.General.ref_height)
                         + 'm}/v_{cut-out}$ [-]')
    ax_pcs[0].set_ylabel('Mean cycle Power P [kW]')
    ax_pcs[1].set_ylabel('Mean cycle Power P [kW]')

    if not config.Plotting.plots_interactive:
        fig.savefig(config.IO.training_plot_output.format(
            title='generated_power_vs_wind_speeds'))
    try:
        n_max_profiles = config.Clustering.n_clusters
    except AttributeError:
        n_max_profiles = 2
    if len(run_profiles) >= n_max_profiles \
            and config.General.write_output:
        df = pd.DataFrame(limits_refined)
        # TODO log? print(df)
        df.to_csv(config.IO.refined_cut_wind_speeds)
        # TODO log? print("Exporting operational limits.")

    return res_pcs, limits_refined


def load_power_curve_results_and_plot_trajectories(config, i_profile=1):
    """Plot trajectories from previously generated power curve."""
    pc = PowerCurveConstructor(None)
    setattr(pc, 'plots_interactive', config.Plotting.plots_interactive)
    setattr(pc, 'plot_output_file', config.IO.training_plot_output)
    pc.import_results(config.IO.power_curve.format(i_profile=i_profile,
                                                   suffix='pickle'))
    pc.plot_optimal_trajectories(wind_speed_ids=[0, 9, 18, 33, 48, 64],
                                 plot_info='_profile_{}'.format(i_profile))
    plt.gcf().set_size_inches(5.5, 3.5)
    plt.subplots_adjust(top=0.99, bottom=0.1, left=0.12, right=0.65)
    pc.plot_optimization_results(plot_info='_profile_{}'.format(i_profile))


def compare_kpis(config, power_curves, compare_profiles=None):
    """Plot how performance indicators change with wind speed for all
        generated power curves."""
    # TODO adapt label to config ref height
    x_label = '$v_{w,' + str(config.General.ref_height) + 'm}$ [m/s]'
    if compare_profiles is not None:
        fig_nums = [plt.figure().number for _ in range(18)]
        tag = '_compare'
    else:
        tag = ''
    for idx, pc in enumerate(power_curves):
        if compare_profiles is None:
            fig_nums = [plt.figure().number for _ in range(18)]
        if compare_profiles is not None:
            if idx+1 not in compare_profiles:
                continue
        sel_succ = [kpis['sim_successful']
                    for kpis in pc.performance_indicators]
        performance_indicators_success = [kpis for i, kpis in enumerate(
            pc.performance_indicators) if sel_succ[i]]
        x_opts_success = [x_opt for i, x_opt in enumerate(pc.x_opts)
                          if sel_succ[i]]

        plt.figure(fig_nums[0])
        f_out_min = [kpis['min_tether_force']['out']
                     for kpis in performance_indicators_success]
        f_out_max = [kpis['max_tether_force']['out']
                     for kpis in performance_indicators_success]
        f_out = [x[0] for x in x_opts_success]
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None, None]
        else:
            labels = ['control', 'min', 'max']
        p = plt.plot(pc.wind_speeds, f_out, label=labels[0])

        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None

        plt.plot(pc.wind_speeds, f_out_min, linestyle='None', marker=6,
                 color=clr, markersize=7, markerfacecolor="None",
                 label=labels[1])
        plt.plot(pc.wind_speeds, f_out_max, linestyle='None', marker=7,
                 color=clr, markerfacecolor="None", label=labels[2])
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Reel-out force [N]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_reel_out_force_vs_wind_'
                       'speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[1])
        f_in_min = [kpis['min_tether_force']['in']
                    for kpis in performance_indicators_success]
        f_in_max = [kpis['max_tether_force']['in']
                    for kpis in performance_indicators_success]
        f_in = [x[1] for x in x_opts_success]
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None, None]
        else:
            labels = ['control', 'min', 'max']
        p = plt.plot(pc.wind_speeds, f_in, label=labels[0])
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None

        plt.plot(pc.wind_speeds, f_in_min, linestyle='None', marker=6,
                 color=clr, markersize=7, markerfacecolor="None",
                 label=labels[1])
        plt.plot(pc.wind_speeds, f_in_max, linestyle='None', marker=7,
                 color=clr, markerfacecolor="None", label=labels[2])
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Reel-in force [N]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_reel_in_force_vs_wind_'
                       'speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[2])
        f_in_min = [kpis['min_reeling_speed']['out']
                    for kpis in performance_indicators_success]
        f_in_max = [kpis['max_reeling_speed']['out']
                    for kpis in performance_indicators_success]
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None]
        else:
            labels = ['min', 'max']
        p = plt.plot(pc.wind_speeds, f_in_min, label=labels[0], marker=6)
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None

        plt.plot(pc.wind_speeds, f_in_max, linestyle='None', marker=7,
                 color=clr, markerfacecolor="None", label=labels[1])
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Reel-out speed [m/s]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_reel_out_speed_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[3])
        v_in_min = [kpis['min_reeling_speed']['in']
                    for kpis in performance_indicators_success]
        v_in_max = [kpis['max_reeling_speed']['in']
                    for kpis in performance_indicators_success]
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None]
        else:
            labels = ['min', 'max']
        p = plt.plot(pc.wind_speeds, v_in_min, label=labels[0], marker=6)
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None

        plt.plot(pc.wind_speeds, v_in_max, linestyle='None', marker=7,
                 color=clr, markerfacecolor="None", label=labels[1])
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Reel-in speed [m/s]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_reel_in_speed_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[4])
        n_cwp = [kpis['n_crosswind_patterns']
                 for kpis in performance_indicators_success]
        plt.plot(pc.wind_speeds, n_cwp, label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Number of cross-wind patterns [-]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_cw_patterns_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[5])
        elev_angles = [x_opt[2]*180./np.pi for x_opt in x_opts_success]
        plt.plot(pc.wind_speeds, elev_angles, label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Reel-out elevation angle [deg]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_reel_out_elev_angle_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[6])
        duty_cycle = [kpis['duty_cycle']
                      for kpis in performance_indicators_success]
        plt.plot(pc.wind_speeds, np.array(duty_cycle)*100,
                 label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Duty cycle [%]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_duty_cycle_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[7])
        pump_eff = [kpis['pumping_efficiency']
                    for kpis in performance_indicators_success]
        plt.plot(pc.wind_speeds, np.array(pump_eff)*100,
                 label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Pumping efficiency [%]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_pumping_efficiency_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[8])
        trac_height = [kpis['average_traction_height']
                       for kpis in performance_indicators_success]
        plt.plot(pc.wind_speeds, np.array(trac_height),
                 label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Average traction height [m]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_average_traction_height_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[9])
        v_trac_height = [kpis['wind_speed_at_avg_traction_height']
                         for kpis in performance_indicators_success]
        plt.plot(pc.wind_speeds, pc.wind_speeds,
                 color='darkslategrey', linestyle='--', alpha=0.5)
        plt.plot(pc.wind_speeds, np.array(v_trac_height),
                 label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Wind speed @ avg. traction height [m/s]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_v_at_avg_traction_height_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[10])
        p_cycle = [kpis['average_power']['cycle']
                   for kpis in performance_indicators_success]
        p_in = [kpis['average_power']['in']
                for kpis in performance_indicators_success]
        p_out = [kpis['average_power']['out']
                 for kpis in performance_indicators_success]
        for i in range(len(p_out)):
            if p_out[i] is None:
                print('Warning: Results contain None values for traction power')
                p_out[i] = -1

        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None, None, None]
        else:
            labels = ['cycle', 'reel-in', 'reel-out', 'transition']

        p = plt.plot(pc.wind_speeds, np.array(p_cycle)/1000,
                     label=labels[0])
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None

        plt.plot(pc.wind_speeds, np.array(p_in)/1000,
                 label=labels[1], marker=6, color=clr)
        plt.plot(pc.wind_speeds, np.array(p_out)/1000,
                 label=labels[2], marker=7, color=clr)
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Power [kW]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_phase_power_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[11])
        p_trans = [kpis['average_power']['trans']
                   for kpis in performance_indicators_success]
        plt.plot(pc.wind_speeds, np.array(p_trans)/1000,
                 label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Power [kW]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_transisiton_phase_power_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[12])
        eff_cycle = [kpis['generator']['eff']['cycle']
                     for kpis in performance_indicators_success]
        eff_in = [kpis['generator']['eff']['in']
                  for kpis in performance_indicators_success]
        eff_out = [kpis['generator']['eff']['out']
                   for kpis in performance_indicators_success]
        for i in range(len(p_out)):
            if eff_cycle[i] is None:
                print('Warning: Results contain None values for cycle eff')
                eff_cycle[i] = -1
            if eff_out[i] is None:
                print('Warning: Results contain None values for traction eff')
                eff_out[i] = -1
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None, None]
        else:
            labels = ['cycle', 'reel-in', 'reel-out']
        p = plt.plot(pc.wind_speeds, np.array(eff_cycle)*100,
                     label=labels[0])
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None
        plt.plot(pc.wind_speeds, np.array(eff_in)*100, color=clr,
                 label=labels[1], marker=6)  # , linestyle='None')
        plt.plot(pc.wind_speeds, np.array(eff_out)*100, color=clr,
                 label=labels[2], marker=7)  # , linestyle='None')
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Generator Efficiency [%]')

        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_eff_gen_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[13])
        load_in = [kpis['generator']['load']['in']
                   for kpis in performance_indicators_success]
        load_out = [kpis['generator']['load']['out']
                    for kpis in performance_indicators_success]
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None]
        else:
            labels = ['reel-in', 'reel-out']
        p = plt.plot(pc.wind_speeds, np.array(load_in),
                     label=labels[0], marker=6)
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None

        plt.plot(pc.wind_speeds, np.array(load_out), color=clr,
                 label=labels[1], marker=7)
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Load [%]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_load_gen_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[14])
        freq_in = [kpis['generator']['freq']['in']
                   for kpis in performance_indicators_success]
        freq_out = [kpis['generator']['freq']['out']
                    for kpis in performance_indicators_success]
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None]
        else:
            labels = ['reel-in', 'reel-out']
        p = plt.plot(pc.wind_speeds, np.array(freq_in),
                     label=labels[0], marker=6)
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None
        plt.plot(pc.wind_speeds, np.array(freq_out), color=clr,
                 label=labels[1], marker=7)
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Electric Current Frequency [Hz]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_freq_gen_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[15])
        powering_out = [x[5] for x in x_opts_success]
        p = plt.plot(pc.wind_speeds, powering_out, label=str(int(idx + 1)))

        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Traction Powering Factor [-]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_powering_trac_vs_wind_'
                       'speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[16])
        t_cycle = [kpis['duration']['cycle']
                   for kpis in performance_indicators_success]
        t_in = [kpis['duration']['in']
                for kpis in performance_indicators_success]
        t_out = [kpis['duration']['out']
                 for kpis in performance_indicators_success]
        t_trans = [kpis['duration']['trans']
                   for kpis in performance_indicators_success]
        if compare_profiles is not None:
            labels = [str(int(idx + 1)), None, None, None]
        else:
            labels = ['cycle', 'reel-in', 'reel-out', 'transition']

        p = plt.plot(pc.wind_speeds, np.array(t_cycle),
                     label=labels[0])
        if compare_profiles is not None:
            clr = p[-1].get_color()
        else:
            clr = None
        plt.plot(pc.wind_speeds, np.array(t_in), color=clr,
                 label=labels[1], marker=6)
        plt.plot(pc.wind_speeds, np.array(t_out), color=clr,
                 label=labels[2], marker=7)
        plt.plot(pc.wind_speeds, np.array(t_trans), color=clr,
                 label=labels[3], linestyle='-.')
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('Time [s]')
        plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_phase_duration_vs_'
                       'wind_speeds_profile_{}{}'.format(idx+1, tag))))

        plt.figure(fig_nums[17])
        l_tether = [x[3]+x[4] for x in x_opts_success]
        p = plt.plot(pc.wind_speeds, l_tether, label=str(int(idx + 1)))
        plt.grid(True)
        plt.xlabel(x_label)
        plt.ylabel('max tether length [m]')
        if compare_profiles is not None:
            plt.legend()
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.training_plot_output.format(
                title=('performance_indicator_max_tether_length_vs_wind_'
                       'speeds_profile_{}{}'.format(idx+1, tag))))

def combine_separate_profile_files(config,
                                   io_file='refined_cut_wind_speeds',
                                   delete_component_files=True):
    # TODO include as chain functionality
    single_profile_file_name = getattr(config.IO, io_file).replace(
        '.csv', '_profile_{}.csv')
    for i_profile in range(1, config.Clustering.n_clusters+1):
        if i_profile == 1:
            df = pd.read_csv(single_profile_file_name.format(i_profile))
        else:
            df_n = pd.read_csv(single_profile_file_name.format(i_profile))
            df = df.append(df_n, ignore_index=True)
    if delete_component_files:
        for i_profile in range(1, config.Clustering.n_clusters+1):
            os.remove(single_profile_file_name.format(i_profile))
    df.to_csv(getattr(config.IO, io_file))


def get_power_curves(config):
    # TODO include multiprocessing option here
    import time
    since = time.time()
    if config.Power.estimate_cut_in_out:
        # TODO this is estimated every time for all profiles, but
        # also if only one profile is run at a time
        estimate_wind_speed_operational_limits(config)
        write_timing_info('Cut-in/out estimation finished.',
                          time.time() - since)

    if config.Power.make_power_curves:
        if config.Processing.parallel:
            # TODO import not here
            from multiprocessing import Pool
            from tqdm import tqdm
            # TODO tqdm is not useful here - all 8 profile run, no updates
            # until opt is finished
            i_profile = [[i+1] for i in range(config.Clustering.n_clusters)]
            import functools
            funct = functools.partial(generate_power_curves,
                                      config)
            with Pool(config.Processing.n_cores) as p:
                if config.Processing.progress_out == 'stdout':
                    file = sys.stdout
                else:
                    file = sys.stderr
                res = list(tqdm(p.imap(funct, i_profile),
                                total=len(i_profile),
                                file=file))
            # Interpret res: funct returns a list of the result for each
            # process
            pcs = []
            for i, res_n in enumerate(res):
                pcs.append(res_n[0][0])
                refined_limits_n = res_n[1]
                if i == 0:
                    refined_limits = copy.deepcopy(refined_limits_n)
                else:
                    for key, val in refined_limits_n.items():
                        refined_limits[key].append(val)
            combine_separate_profile_files(
                config,
                io_file='refined_cut_wind_speeds')
        else:
            run_profiles = list(range(config.Clustering.n_clusters))
            pcs, limits_refined = generate_power_curves(
                config,
                run_profiles)
        compare_kpis(config, pcs)
        write_timing_info('Power curves finished.',
                          time.time() - since)

def interpret_input_args():
    estimate_cut_in_out, make_power_curves = (False, False)
    if len(sys.argv) > 1:  # User input was given
        help = """
        python power_curves.py                  : run qsm to estimate the
                                                  cut-in and cut-out wind
                                                  speeds and power curves for
                                                  the resulting range of
                                                  absolute wind speeds
        python power_curves.py -p               : run qsm to estimate the
                                                  power curves
        python power_curves.py -c               : run qsm to estimate the
                                                  cut-in and cut-out wind
                                                  speeds
        python power_curves.py -h               : display this help
        """
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hpc",
                                       ["help", "power", "cut"])
        except getopt.GetoptError:
            # User input not given correctly, display help and end
            print(help)
            sys.exit()
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                # Help argument called, display help and end
                print(help)
                sys.exit()
            elif opt in ("-p", "--power"):
                make_power_curves = True
            elif opt in ("-c", "--cut"):
                estimate_cut_in_out = True
    else:
        estimate_cut_in_out, make_power_curves = (True, True)

    return estimate_cut_in_out, make_power_curves


if __name__ == '__main__':
    from ..config import Config
    config = Config()
    # TODO is this necessary here?
    if not config.Plotting.plots_interactive:
        import matplotlib as mpl
        mpl.use('Pdf')
    import matplotlib.pyplot as plt

    # Read program parameters
    estimate_cut_in_out, make_power_curves, run_single_profile = \
        interpret_input_args()
    setattr(config.Power, 'estimate_cut_in_out', estimate_cut_in_out)
    setattr(config.Power, 'make_power_curves', make_power_curves)
    get_power_curves(config)
    if config.Plotting.plots_interactive:
        plt.show()
