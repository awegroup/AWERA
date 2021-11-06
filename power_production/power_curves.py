import os
import numpy as np

import pandas as pd
from copy import deepcopy

import sys
import getopt
from qsm import LogProfile, NormalisedWindTable1D, KiteKinematics,\
    SteadyState, TractionPhaseHybrid, TractionConstantElevation, \
    SteadyStateError, OperationalLimitViolation, TractionPhase
from kitepower_kites import sys_props_v3
from cycle_optimizer import OptimizerCycle
from power_curve_constructor import PowerCurveConstructor
from utils import write_timing_info
from config_production import file_name_cluster_profiles, \
    cut_wind_speeds_file, \
    refined_cut_wind_speeds_file, power_curve_output_file_name, \
    plots_interactive, plot_output_file, n_clusters

if not plots_interactive:
    import matplotlib as mpl
    mpl.use('Pdf')
import matplotlib.pyplot as plt

# Assumptions representative reel-out state at cut-in wind speed.
theta_ro_ci = 25 * np.pi / 180.
phi_ro = 13 * np.pi / 180.
chi_ro = 100 * np.pi / 180.

l0 = 200  # Tether length at start of reel-out.
l1_lb = 350  # Lower bound of tether length at end of reel-out.
l1_ub = 450  # Upper bound of tether length at end of reel-out.


def calc_tether_force_traction(env_state, straight_tether_length):
    """Calculate tether force for the minimum allowable reel-out speed
    and given wind conditions and tether length."""
    kinematics = KiteKinematics(straight_tether_length,
                                phi_ro, theta_ro_ci, chi_ro)
    env_state.calculate(kinematics.z)
    sys_props_v3.update(kinematics.straight_tether_length, True)
    ss = SteadyState({'enable_steady_state_errors': True})
    ss.control_settings = ('reeling_speed',
                           sys_props_v3.reeling_speed_min_limit)
    ss.find_state(sys_props_v3, env_state, kinematics)
    return ss.tether_force_ground


def get_cut_in_wind_speed(env):
    """Iteratively determine lowest wind speed for which, along the entire
    reel-out path, feasible steady flight states
    with the minimum allowable reel-out speed are found."""
    dv = 1e-2  # Step size [m/s].
    v0 = 5.6  # Lowest wind speed [m/s] with which the iteration is started.

    v = v0
    rescan_finer_steps = False
    while True:
        env.set_reference_wind_speed(v)
        try:
            # Setting tether force as setpoint in qsm yields infeasible region
            tether_force_start = calc_tether_force_traction(env, l0)
            tether_force_end = calc_tether_force_traction(env, l1_lb)

            start_critical = tether_force_end > tether_force_start
            if start_critical:
                critical_force = tether_force_start
            else:
                critical_force = tether_force_end

            if tether_force_start > sys_props_v3.tether_force_min_limit and \
                    tether_force_end > sys_props_v3.tether_force_min_limit:
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


def calc_n_cw_patterns(env, theta=60. * np.pi / 180.):
    """Calculate the number of cross-wind manoeuvres flown."""
    trac = TractionPhaseHybrid({
            'control': ('tether_force_ground',
                        sys_props_v3.tether_force_max_limit),
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
    trac.run_simulation(sys_props_v3, env,
                        {'enable_steady_state_errors': True})

    return trac.n_crosswind_patterns


def get_max_wind_speed_at_elevation(env=LogProfile(),
                                    theta=60. * np.pi / 180.):
    """Iteratively determine maximum wind speed allowing at least one
    cross-wind manoeuvre during the reel-out phase for
    provided elevation angle."""
    dv = 0.2  # Step size [m/s].
    v0 = 15.  # Lowest wind speed [m/s] with which the iteration is started.
    # Check if the starting wind speed gives a feasible solution.
    env.set_reference_wind_speed(v0)
    try:
        n_cw_patterns = calc_n_cw_patterns(env, theta)
    except SteadyStateError as e:
        if e.code != 8:
            raise ValueError("No feasible solution found for first"
                             "assessed cut out wind speed.")

    # Increase wind speed until number of cross-wind manoeuvres subceeds one.
    v = v0 + dv

    rescan_finer_steps = False
    fail_counter = 0
    while True:
        env.set_reference_wind_speed(v)
        print('velocity: ', v, 'rescan: ', rescan_finer_steps,
              'fail_couter: ', fail_counter)
        try:
            n_cw_patterns = calc_n_cw_patterns(env, theta)
            fail_counter = 0  # Simulation worked -> reset fail counter
            print('n-cw-patterns', n_cw_patterns)
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


def get_cut_out_wind_speed(env=LogProfile()):
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
        vw = get_max_wind_speed_at_elevation(env, beta)
        if vw is not None:
            if vw <= vw_last:
                return vw_last, beta+dbeta
            vw_last = vw
        beta -= dbeta


def export_to_csv(v, v_cut_out, p, x_opts, n_cwp, i_profile):
    df = {
        'v_100m [m/s]': v,
        'v/v_cut-out [-]': v/v_cut_out,
        'P [W]': p,
        'F_out [N]': [x[0] for x in x_opts],
        'F_in [N]': [x[1] for x in x_opts],
        'theta_out [rad]': [x[2] for x in x_opts],
        'dl_tether [m]': [x[3] for x in x_opts],
        'l0_tether [m]': [x[4] for x in x_opts],
        'n_crosswind_patterns [-]': n_cwp,
    }
    df = pd.DataFrame(df)
    df.to_csv(power_curve_output_file_name.format(i_profile=i_profile,
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


def estimate_wind_speed_operational_limits(n_clusters=8,
                                           export_operational_limits=True):
    """Estimate the cut-in and cut-out wind speeds for each wind profile shape.
    These wind speeds are refined when determining the power curves."""

    fig, ax = plt.subplots(1, 2, figsize=(5.5, 3), sharey=True)
    plt.subplots_adjust(top=0.92, bottom=0.164, left=0.11,
                        right=0.788, wspace=0.13)

    res = {'vw_100m_cut_in': [],
           'vw_100m_cut_out': [],
           'tether_force_cut_in': [],
           }
    input_profiles = pd.read_csv(file_name_cluster_profiles, sep=";")
    for i_profile in range(1, n_clusters+1):
        print('Profile {}'.format(i_profile))
        env = create_environment(input_profiles, i_profile)

        # Get cut-in wind speed.
        vw_cut_in, _, tether_force_cut_in = get_cut_in_wind_speed(env)
        res['vw_100m_cut_in'].append(vw_cut_in)
        res['tether_force_cut_in'].append(tether_force_cut_in)

        # Get cut-out wind speed, which proved to work better when using
        # 250m reference height.
        env.set_reference_height(250)
        vw_cut_out250m, elev = get_cut_out_wind_speed(env)
        env.set_reference_wind_speed(vw_cut_out250m)
        vw_cut_out = env.calculate_wind(100.)
        res['vw_100m_cut_out'].append(vw_cut_out)

        # Plot the wind profiles corresponding to the wind speed operational
        # limits and the profile shapes.
        env.set_reference_height(100.)
        env.set_reference_wind_speed(vw_cut_in)
        plt.sca(ax[0])
        env.plot_wind_profile()

        env.set_reference_wind_speed(vw_cut_out)
        plt.sca(ax[1])
        env.plot_wind_profile("{}".format(i_profile))
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.ylabel('')

    df = pd.DataFrame(res)
    print(df)

    if export_operational_limits:
        df.to_csv(cut_wind_speeds_file)

    ax[0].set_title("Cut-in")
    ax[0].set_xlim([0, None])
    ax[0].set_ylim([0, 400])
    ax[1].set_title("Cut-out")
    ax[1].set_xlim([0, None])
    ax[1].set_ylim([0, 400])
    if not plots_interactive:
        plt.savefig(plot_output_file.format(
            title='estimate_wind_speed_operational_limits'))


def generate_power_curves(n_clusters=8, run_single_profile=-1):
    """Determine power curves - requires estimates of the cut-in
        and cut-out wind speed to be available."""
    limit_estimates = pd.read_csv(cut_wind_speeds_file)

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

    fig, ax_pcs = plt.subplots(2, 1)
    for a in ax_pcs:
        a.grid()

    limits_refined = {'i_profile': [],
                      'vw_100m_cut_in': [],
                      'vw_100m_cut_out': []}
    res_pcs = []
    input_profiles = pd.read_csv(file_name_cluster_profiles, sep=";")
    for i_profile in range(1, n_clusters+1):
        if run_single_profile != -1 and i_profile != run_single_profile:
            continue
        print("Power curve generation for profile number {}".format(i_profile))
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
        wind_speeds = np.linspace(vw_cut_in, vw_cut_out-1, 50)
        wind_speeds = np.concatenate((wind_speeds,
                                      np.linspace(vw_cut_out-1,
                                                  vw_cut_out-0.05, 15)))

        # For the first phase of the power curve, the constraint on the number
        # of cross-wind patterns flown is not
        # active. It is assumed that sufficient cross-wind patterns are flown
        # up to vw_100m = 7 m/s (verify this).
        # Therefore, the number of cross-wind patterns is not calculated for
        # this phase. Also the upper elevation bound
        # is set to 30 degrees.
        op_cycle_pc_phase1 = OptimizerCycle(cycle_sim_settings_pc_phase1,
                                            sys_props_v3, env,
                                            reduce_x=np.array([0, 1, 2, 3]))
        op_cycle_pc_phase1.bounds_real_scale[2][1] = 30*np.pi/180.

        op_cycle_pc_phase2 = OptimizerCycle(cycle_sim_settings_pc_phase2,
                                            sys_props_v3, env,
                                            reduce_x=np.array([0, 1, 2, 3]))

        # Configuration of the sequential optimizations for which is
        # differentiated between the wind speed ranges
        # bounded above by the wind speed of the dictionary key. If dx0
        # does not contain only zeros, the starting point
        # of the new optimization is not the solution of the
        # preceding optimization.
        op_seq = {
            7.: {'power_optimizer': op_cycle_pc_phase1,
                 'dx0': np.array([0., 0., 0., 0., 0.])},
            17.: {'power_optimizer': op_cycle_pc_phase2,
                  'dx0': np.array([0., 0., 0., 0., 0.])},
            np.inf: {'power_optimizer': op_cycle_pc_phase2,
                     'dx0': np.array([0., 0., 0.1, 0., 0.])},
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
        x0 = np.array([critical_force, 300., theta_ro_ci, 150., 200.0])

        # Start optimizations.
        pc = PowerCurveConstructor(wind_speeds)
        pc.run_predefined_sequence(op_seq, x0)
        # export all results, including failed simulations, tagged in
        # kip and other performance flags
        pc.export_results(power_curve_output_file_name.format(
            i_profile=i_profile, suffix='pickle'))
        res_pcs.append(pc)

        # mask failed simulation, export only good results
        sel_succ = [kpis['sim_successful']
                    for kpis in pc.performance_indicators]

        # Plot power curve together with that of the other wind profile shapes.
        p_cycle = np.array([kpis['average_power']['cycle']
                            for kpis in pc.performance_indicators])[sel_succ]
        print('p_cycle: ', p_cycle)
        # Mask negative jumps in power
        sel_succ_power = p_cycle > 0
        # TODO  - check reason for negative power/ strong jumps
        if sum(sel_succ_power) != len(sel_succ_power):
            print(len(sel_succ_power)-sum(sel_succ_power),
                  'masked negative powers')
        p_cycle_masked = p_cycle[sel_succ_power]

        # TODO resolve source of problems
        while True:
            sel_succ_power_disc = [True] + list(np.diff(p_cycle_masked)
                                                > -1000)
            sel_succ_power[sel_succ_power] = sel_succ_power_disc
            p_cycle_masked = p_cycle_masked[sel_succ_power_disc]
            if sum(sel_succ_power_disc) == len(sel_succ_power_disc):
                # No more discontinuities
                break
            print(len(sel_succ_power_disc)-sum(sel_succ_power_disc),
                  'masked power discontinuities')

        p_cycle = p_cycle[sel_succ_power]
        wind = pc.wind_speeds[sel_succ_power]

        ax_pcs[0].plot(wind, p_cycle_masked/1000, label=i_profile)
        ax_pcs[1].plot(wind/vw_cut_out, p_cycle_masked/1000, label=i_profile)

        pc.plot_optimal_trajectories(plot_info='_profile_{}'.format(i_profile))
        pc.plot_optimization_results(op_cycle_pc_phase2.OPT_VARIABLE_LABELS,
                                     op_cycle_pc_phase2.bounds_real_scale,
                                     [sys_props_v3.tether_force_min_limit,
                                      sys_props_v3.tether_force_max_limit],
                                     [sys_props_v3.reeling_speed_min_limit,
                                      sys_props_v3.reeling_speed_max_limit],
                                     plot_info='_profile_{}'.format(i_profile))

        n_cwp = np.array([kpis['n_crosswind_patterns']
                          for kpis in pc.performance_indicators]
                         )[sel_succ][sel_succ_power]
        x_opts = np.array(pc.x_opts)[sel_succ][sel_succ_power]

        export_to_csv(wind, vw_cut_out, p_cycle, x_opts, n_cwp, i_profile)
        # Refine the wind speed operational limits to wind speeds for
        # which optimal solutions are found.
        limits_refined['i_profile'].append(i_profile)
        limits_refined['vw_100m_cut_in'].append(wind[0])
        limits_refined['vw_100m_cut_out'].append(wind[-1])

        print("Cut-in and -out speeds changed from [{:.3f}, {:.3f}] to "
              "[{:.3f}, {:.3f}].".format(vw_cut_in, vw_cut_out,
                                         wind[0],
                                         wind[-1]))
    # into funtion: combine single power curve results starting here
    ax_pcs[1].legend()
    ax_pcs[0].set_xlabel('$v_{w,100m}$ [m/s]')
    ax_pcs[1].set_xlabel('$v_{w,100m}/v_{cut-out}$ [-]')
    ax_pcs[0].set_ylabel('Mean cycle Power P [kW]')
    ax_pcs[1].set_ylabel('Mean cycle Power P [kW]')

    if not plots_interactive:
        fig.savefig(plot_output_file.format(
            title='generated_power_vs_wind_speeds'))

    df = pd.DataFrame(limits_refined)
    print(df)
    if run_single_profile != -1:
        df.to_csv(refined_cut_wind_speeds_file.replace(
            '.csv', '_profile_{}.csv'.format(run_single_profile)))
        # TODO include this change in config?
        # TODO recombine file after all power curves are finished
        print("Exporting single location operational limits.")
    else:
        df.to_csv(refined_cut_wind_speeds_file)
        print("Exporting operational limits.")

    return res_pcs


def load_power_curve_results_and_plot_trajectories(n_clusters=8, i_profile=1):
    """Plot trajectories from previously generated power curve."""
    pc = PowerCurveConstructor(None)
    pc.import_results(power_curve_output_file_name.format(i_profile, 'pickle'))
    pc.plot_optimal_trajectories(wind_speed_ids=[0, 9, 18, 33, 48, 64],
                                 plot_info='_profile_{}'.format(i_profile))
    plt.gcf().set_size_inches(5.5, 3.5)
    plt.subplots_adjust(top=0.99, bottom=0.1, left=0.12, right=0.65)
    pc.plot_optimization_results(plot_info='_profile_{}'.format(i_profile))


def compare_kpis(power_curves):
    """Plot how performance indicators change with wind speed for all
        generated power curves."""
    fig_nums = [plt.figure().number for _ in range(5)]
    for idx, pc in enumerate(power_curves):
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
        p = plt.plot(pc.wind_speeds, f_out)
        clr = p[-1].get_color()
        plt.plot(pc.wind_speeds, f_out_min, linestyle='None', marker=6,
                 color=clr, markersize=7, markerfacecolor="None")
        plt.plot(pc.wind_speeds, f_out_max, linestyle='None', marker=7,
                 color=clr, markerfacecolor="None")
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-out force [N]')
        if not plots_interactive:
            plt.savefig(plot_output_file.format(
                title=('performance_indicator_reel_out_force_vs_wind_'
                       'speeds_profile_{}'.format(idx+1))))

        plt.figure(fig_nums[1])
        f_in_min = [kpis['min_tether_force']['in']
                    for kpis in performance_indicators_success]
        f_in_max = [kpis['max_tether_force']['in']
                    for kpis in performance_indicators_success]
        f_in = [x[1] for x in x_opts_success]
        p = plt.plot(pc.wind_speeds, f_in)
        clr = p[-1].get_color()
        plt.plot(pc.wind_speeds, f_in_min, linestyle='None', marker=6,
                 color=clr, markersize=7, markerfacecolor="None")
        plt.plot(pc.wind_speeds, f_in_max, linestyle='None', marker=7,
                 color=clr, markerfacecolor="None")
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-in force [N]')
        if not plots_interactive:
            plt.savefig(plot_output_file.format(
                title=('performance_indicator_reel_in_force_vs_wind_'
                       'speeds_profile_{}'.format(idx+1))))

        plt.figure(fig_nums[2])
        f_in_min = [kpis['min_reeling_speed']['out']
                    for kpis in performance_indicators_success]
        f_in_max = [kpis['max_reeling_speed']['out']
                    for kpis in performance_indicators_success]
        p = plt.plot(pc.wind_speeds, f_in_min)
        clr = p[-1].get_color()
        plt.plot(pc.wind_speeds, f_in_max, linestyle='None', marker=7,
                 color=clr, markerfacecolor="None")
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-out speed [m/s]')
        if not plots_interactive:
            plt.savefig(plot_output_file.format(
                title=('performance_indicator_reel_out_speed_vs_'
                       'wind_speeds_profile_{}'.format(idx+1))))

        plt.figure(fig_nums[3])
        n_cwp = [kpis['n_crosswind_patterns']
                 for kpis in performance_indicators_success]
        plt.plot(pc.wind_speeds, n_cwp)
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Number of cross-wind patterns [-]')
        if not plots_interactive:
            plt.savefig(plot_output_file.format(
                title=('performance_indicator_cw_patterns_vs_'
                       'wind_speeds_profile_{}'.format(idx+1))))

        plt.figure(fig_nums[4])
        elev_angles = [x_opt[2]*180./np.pi for x_opt in x_opts_success]
        plt.plot(pc.wind_speeds, elev_angles)
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-out elevation angle [deg]')
        if not plots_interactive:
            plt.savefig(plot_output_file.format(
                title=('performance_indicator_reel_out_elev_angle_vs_'
                       'wind_speeds_profile_{}'.format(idx+1))))


def combine_separate_profile_files(n_profiles=n_clusters,
                                   file_name=refined_cut_wind_speeds_file):
    single_profile_file_name = file_name.replace(
        '.csv', '_profile_{}.csv')
    for i_profile in range(n_profiles):
        if i_profile == 0:
            df = pd.read_csv(single_profile_file_name.format(i_profile+1))
        else:
            df_n = pd.read_csv(single_profile_file_name.format(i_profile+1))
            df = df.append(df_n, ignore_index=True)
    df.to_csv(refined_cut_wind_speeds_file)


def interpret_input_args():
    estimate_cut_in_out, make_power_curves = (False, False)
    run_single_profile = -1
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
            opts, args = getopt.getopt(sys.argv[1:], "hpcs:",
                                       ["help", "power", "cut", "single="])
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
            elif opt in ("-s", "--single"):
                run_single_profile = int(arg)
    else:
        estimate_cut_in_out, make_power_curves = (True, True)

    return estimate_cut_in_out, make_power_curves, run_single_profile


if __name__ == '__main__':
    # Read program parameters
    estimate_cut_in_out, make_power_curves, run_single_profile = \
        interpret_input_args()

    import time
    since = time.time()
    if estimate_cut_in_out:
        # TODO this is estimated every time for all profiles, but
        # also if only one profile is run at a time
        estimate_wind_speed_operational_limits(n_clusters=n_clusters)
        write_timing_info('Cut-in/out estimation finished.',
                          time.time() - since)
    if make_power_curves:
        pcs = generate_power_curves(n_clusters=n_clusters,
                                    run_single_profile=run_single_profile)
        compare_kpis(pcs)
        write_timing_info('Power curves finished.',
                          time.time() - since)
    if plots_interactive:
        plt.show()
