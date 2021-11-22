# from pyoptsparse import History
# !!! inlcude optional history file of optimizations?
import numpy as np
import pickle
import pandas as pd

import time
from .utils import write_timing_info

from .cycle_optimizer import OptimizerCycle

from ..wind_profile_clustering.read_requested_data import get_wind_data
from ..wind_profile_clustering.preprocess_data import \
    preprocess_data, reduce_wind_data

from .qsm import NormalisedWindTable1D

from ..wind_profile_clustering.config import locations, file_name_cluster_labels
# from config_production import optimizer_history_file_name

# Optimization trial runs
from scipy.stats import truncnorm
from .qsm import SteadyStateError, OperationalLimitViolation, PhaseError
from .cycle_optimizer import OptimizerError


# Assumptions representative reel-out state at cut-in wind speed.
# !!! import from somewhere
# theta_ro_ci = 25 * np.pi / 180.
phi_ro = 13 * np.pi / 180.
chi_ro = 100 * np.pi / 180.

since = time.time()

def read_matching_clustering_results(config, i_loc, single_sample_id, backscaling):
    with open(file_name_cluster_labels, 'rb') as f:
        labels_file = pickle.load(f)
    n_locs = len(locations)
    n_time_samples = int(labels_file['labels [-]'].shape[0]/n_locs)
    matching_profile = labels_file['labels [-]'][single_sample_id +
                                                 i_loc*n_time_samples]
    cluster_id = matching_profile+1
    clustering_df = pd.read_csv(config.IO.power_curve
                                .format(i_profile=cluster_id,
                                        suffix='csv'),
                                sep=";")
    clustering_v_bins = clustering_df['v_100m [m/s]']
    # Match sample wind speed to optimization output
    # Minimal distance 'fails' for backscaling out of cut-in/out range
    clustering_v_bin_idx = np.abs(clustering_v_bins - backscaling).argmin()
    x_opt_cluster = [clustering_df['F_out [N]'][clustering_v_bin_idx],
                     clustering_df['F_in [N]'][clustering_v_bin_idx],
                     clustering_df['theta_out [rad]'][clustering_v_bin_idx],
                     clustering_df['dl_tether [m]'][clustering_v_bin_idx],
                     clustering_df['l0_tether [m]'][clustering_v_bin_idx]]
    # Interpolate power via wind speeds:
    p_cluster = np.interp(backscaling,
                          clustering_v_bins,
                          clustering_df['P [W]'])
    return p_cluster, x_opt_cluster, cluster_id


def create_environment(wind_profile_u, wind_profile_v, heights):
    """Flatten wind profile shapes resulting from the clustering
    and use to create the environment object."""
    env = NormalisedWindTable1D()
    # velocities w.r.t. env.h_ref = 100.
    env.heights = list(heights)
    env.normalised_wind_speeds = list(
        (wind_profile_u**2 + wind_profile_v**2)**.5)
    return env


def mult_x_trial_run_optimization(x0,
                                  power_optimizer,
                                  n_x_test=3,
                                  stop_optimize_on_success=False,
                                  test_until_n_succ=3,
                                  x0_random=True):
    print("x0:", x0)
    # Optimize around x0
    # perturb x0:
    x0_range = [x0]
    # Optimization variables bounds defining the search space.
    bounds = power_optimizer.bounds_real_scale
    reduce_x = power_optimizer.reduce_x
    for n_test in range(n_x_test):
        # Gaussian random selection of x0 within bounds - truncated normal dist
        # !!! why not use uniform distribution here?
        if x0_random:
            x0_range.append(
                [truncnorm(a=bounds[i][0]/bounds[i][1], b=1,
                           scale=bounds[i][1]).rvs()
                 if i in reduce_x else x0[i] for i in range(len(x0))])

        # Gaussian random smearing of x0 within bounds
        smearing = 0.1  # 10% smearing of the respective values  # TODO option

        def get_smeared_x0():
            return [np.random.normal(x0[i], x0[i]*smearing)
                    if i in reduce_x else x0[i] for i in range(len(x0))]

        def test_smeared_x0(test_x0, precision=0):
            return np.all(
                [np.logical_and(test_x0[i] >= (bounds[i][0]-precision),
                                test_x0[i] <= (bounds[i][1]+precision))
                 for i in range(len(test_x0))])

        def smearing_x0():
            test_smearing = get_smeared_x0()
            bounds_adhered = test_smeared_x0(test_smearing)
            while not bounds_adhered:
                test_smearing = get_smeared_x0()
                bounds_adhered = test_smeared_x0(test_smearing)
            return test_smearing
        # Test on two smeared variations of x0
        x0_range.append(smearing_x0())
        x0_range.append(smearing_x0())

    x0_range = np.array(x0_range)
    print('Testing x0 range: ', x0_range)
    n_x0 = x0_range.shape[0]
    x_opts, op_ress, conss, kpiss, sim_successfuls, opt_successfuls, errs = \
        [], [], [], [], [], [], []
    for i in range(n_x0):
        x0_test = x0_range[i]
        power_optimizer.x0_real_scale = x0_test
        try:
            print("Testing the {}th starting values: {}".format(i, x0_test))
            x_opts.append(power_optimizer.optimize())
            # Safety check if variable bounds are adhered
            if test_smeared_x0(power_optimizer.x_opt_real_scale,
                               precision=power_optimizer.precision):
                op_ress.append(power_optimizer.op_res)
                opt_successfuls.append(True)
                try:
                    cons, kpis = power_optimizer.eval_point()
                    conss.append(cons)
                    kpiss.append(kpis)
                    sim_successfuls.append(True)
                    print('Simulation successful')
                    if stop_optimize_on_success or (sum(sim_successfuls)
                                                    == test_until_n_succ):
                        x0_range = x0_range[:i+1]
                        break
                except (SteadyStateError,
                        OperationalLimitViolation,
                        PhaseError) as e:
                    print("Error occurred while evaluating the resulting"
                          " optimal point: {}".format(e))
                    # Try with relaxed errors
                    # relaxed errors only relax OperationalLimitViolation
                    cons, kpis = power_optimizer.eval_point(relax_errors=True)
                    conss.append(cons)
                    kpiss.append(kpis)
                    sim_err = e
                    sim_successfuls.append(False)
                    print('Simulation failed -> errors relaxed')
            else:
                print("Optimization number {} finished with an error: {}"
                      .format(i, 'Optimization bounds violated'))
                opt_err = OptimizerError("Optimization bounds violated.")
                x_opts = x_opts[:-1]  # Drop last x_opt, bounds are not adhered
                opt_successfuls.append(False)

        except (OptimizerError) as e:
            print("Optimization number {} finished with an error: {}"
                  .format(i, e))
            opt_err = e
            opt_successfuls.append(False)
            continue
        except (SteadyStateError, PhaseError, OperationalLimitViolation) as e:
            print("Optimization number {} finished with a simulation error: {}"
                  .format(i, e))
            opt_err = e
            opt_successfuls.append(False)
            continue
        except (FloatingPointError) as e:
            print(("Optimization number {} finished due to a mathematical" +
                   " simulation error: {}").format(i, e))
            opt_err = e
            opt_successfuls.append(False)

    if sum(sim_successfuls) > 0:
        # Consistency check sim results - both optimization and simulation work
        x0_success = x0_range[opt_successfuls][sim_successfuls]
        # x0_failed = list(x0_range[np.logical_not(opt_successfuls)]) + \
        # list(x0_range[opt_successfuls][np.logical_not(sim_successfuls)])
        # print('Failed starting values: ', x0_failed)
        # print('Successful starting values: ', x0_success)

        # consistency check function values
        # corresponding eval function values from the optimizer
        # flag_unstable_opt_result = False

        # print('Optimizer x point results: ', x_opts)
        # print(' Leading to a successful simulation:', sim_successfuls)
        x_opts_succ = np.array(x_opts)[sim_successfuls]
        # (x_opt_mean, x_opt_std) = (np.mean(x_opts_succ, axis=0),
        #                           np.std(x_opts_succ, axis=0))
        # print('  The resulting mean {} with a standard deviation of {}'
        #    .format(x_opt_mean, x_opt_std))
        # if (x_opt_std > np.abs(0.1*x_opt_mean)).any():
        #     # TODO: lower/higher check? - make this as debug output?
        #     print('  More than 1% standard deviation - unstable result')
        #    flag_unstable_opt_result = True

        # Corresponding eval function values from the optimizer
        op_ress_succ = [op_ress[i] for i in range(len(op_ress))
                        if sim_successfuls[i]]
        f_opt = [op_res['fun'] for op_res in op_ress_succ]
        # print('Successful optimizer eval function results: ', f_opt)
        # (f_opt_mean, f_opt_std) = (np.mean(f_opt), np.std(f_opt))
        # print('  The resulting mean {} with a standard deviation of {}'
        #    .format(f_opt_mean, f_opt_std))

        # Chose best optimization result:
        minimal_f_opt = np.argmin(f_opt)  # Matching index in sim_successfuls

        x0 = x0_success[minimal_f_opt]
        x_opt = x_opts_succ[minimal_f_opt]
        op_res = op_ress_succ[minimal_f_opt]

        cons = [conss[i] for i in range(len(kpiss))
                if sim_successfuls[i]][minimal_f_opt]

        # Failed simulation results are later masked
        kpis = [kpiss[i] for i in range(len(kpiss))
                if sim_successfuls[i]][minimal_f_opt]
        kpis['sim_successful'] = True

        return x_opt, op_res, cons, kpis, sim_successfuls

    else:
        # optimizations/simulations all failed
        print('All optimizations/simulations failed:')
        return [-1]*len(x0), -1, -1, -1, sim_successfuls


def single_profile_power(config, processed_data, single_sample_id, i_loc):
    from qsm import TractionPhase, TractionPhaseHybrid
    from kitepower_kites import sys_props_v3

    # Normalise sample wind profile
    # Return profile and reference wind speed
    heights = processed_data['altitude']
    wind_profile_norm_v = processed_data['training_data'][:, len(heights):][0]
    wind_profile_norm_u = processed_data['training_data'][:, :len(heights)][0]
    w = (wind_profile_norm_u**2 + wind_profile_norm_v**2)**.5
    # Interpolate normalised wind speed at reference height 100m
    w_ref = np.interp(100, heights, w)
    # Normalise the absolute wind speed: at reference height to 1
    sf = 1/w_ref
    # wind_profile_norm_v = wind_profile_norm_v*sf
    # wind_profile_norm_u = wind_profile_norm_u*sf
    # Scale back to original profile
    backscaling = processed_data['normalisation_value'][0]/sf
    # Simulation/Optimization not using normalised wind profiles
    # backscaling used for cluster matching
    wind_u = processed_data['wind_speed_east'][0]
    wind_v = processed_data['wind_speed_north'][0]
    ref_wind_speed = 1  # No backscaling of non-normalised wind profiles
    # Create optimization wind environment
    env_state = create_environment(wind_u, wind_v, heights)
    env_state.set_reference_wind_speed(ref_wind_speed)
    # Cycle simulation settings for different phases of the power curves.
    cycle_sim_settings_pc = {
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
    # Modify settings in accordance with clustering settings
    if backscaling > 7:  # TODO test what happen if I always use Hybrid?
        cycle_sim_settings_pc['cycle']['traction_phase'] = TractionPhaseHybrid
    oc = OptimizerCycle(cycle_sim_settings_pc, sys_props_v3,
                        env_state, reduce_x=np.array([0, 1, 2, 3]))
    # !!! optional reduce_x, settingsm ref wind speed -> also in class?
    if backscaling <= 7:
        oc.bounds_real_scale[2][1] = 30*np.pi/180.

    # Read matching clustering results (cluster)
    p_cluster, x_opt_cluster, cluster_id = \
        read_matching_clustering_results(config,
                                         i_loc,
                                         single_sample_id,
                                         backscaling)

    # Test clustering output: evaluate point using cluster control (cc)
    oc.x0_real_scale = x_opt_cluster
    try:
        cons_cc, kpis_cc = oc.eval_point()
        p_cc = kpis_cc['average_power']['cycle']
    except (SteadyStateError, PhaseError, OperationalLimitViolation) as e:
        print("Clustering output point evaluation finished with a "
              "simulation error: {}".format(e))
        p_cc = -1
    # Optimise using the clustering control parameters (cc_opt)
    # [and gaussian smeared] as starting values
    x_opt_cc_opt, op_res, cons, kpis, \
        sim_successfuls = mult_x_trial_run_optimization(x_opt_cluster,
                                                        oc,
                                                        x0_random=False)
    if sum(sim_successfuls) == 0:
        p_cc_opt = -1
    else:
        p_cc_opt = kpis['average_power']['cycle']

    # Optimize individual sample (sample)
    # Starting control parameters from mean of ~180k optimizations
    # unbiased (starting controls independent of clustering)
    x0 = [4100., 850., 0.5, 240., 200.0]  # TODO make optional
    x_opt_sample, op_res, cons, kpis, \
        sim_successfuls = mult_x_trial_run_optimization(x0, oc)
    if sum(sim_successfuls) < 1:
        p_sample = -1

    else:
        p_sample = kpis['average_power']['cycle']

        print('opt output cluster: ', x_opt_cluster)
        print('End of Optimization: ', x_opt_sample)
        print(('Estimated power for sample {} {} - Cluster optimization:' +
               ' {}, Single Sample evaluation: {}, Single sample eval with' +
               ' cluster configuration: {}')
              .format(processed_data['datetime'], processed_data['locations'],
                      p_cluster, p_sample, p_cc))

        time_elapsed = time.time() - since
        print('Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                          time_elapsed % 60))

        # optHist = History(optimizer_history_file_name)
        # print(optHist.getValues(major=True, scale=False,
        #                         stack=False, allowSens=True)['isMajor'])
    power = [p_sample, p_cluster, p_cc, p_cc_opt]
    x_opt = [x_opt_sample, x_opt_cluster, x_opt_cc_opt]
    cluster_info = [cluster_id, backscaling]
    return power, x_opt, cluster_info
    # TODO optional what opts to run?


def run_location(config, loc, sel_sample_ids):
    power = np.zeros([4, len(sel_sample_ids)])
    # x_opt_sample, x_opt_cluster, x_opt_cc_opt
    x_opt = np.zeros([3, len(sel_sample_ids), 5])
    # cluster_id, backscaling
    cluster_info = np.zeros([2, len(sample_ids)])
    # Read all selected samples for location
    if len(sel_sample_ids) > 0:
        data = get_wind_data(locs=[loc], sel_sample_ids=sel_sample_ids)
    else:
        # No samples specified, run over all samples
        data = get_wind_data(locs=[loc])
        sel_sample_ids = list(range(data['n_samples']))
    # write_timing_info('Input read', time.time() - since)
    # Run preprocessing
    processed_data = preprocess_data(data, remove_low_wind_samples=False)
    # Identify i_loc with clustering i_location to read matching
    # clustering output power / control parameters
    i_location = locations.index(loc)
    # TODO timing
    # Iterate optimization for all samples individually
    for i_sample, sample_id in enumerate(sel_sample_ids):
        print('{}/{} Processing sample {}...'
              .format(i_sample+1, len(sel_sample_ids), sample_id))
        sample_time = time.time()
        mask_keep = processed_data['wind_speed'][:, 0] < -999
        mask_keep[i_sample] = True
        processed_data_sample = reduce_wind_data(processed_data,
                                                 mask_keep,
                                                 return_copy=True)
        power[:, i_sample], x_opt[:, i_sample, :], \
            cluster_info[:, i_sample] = \
            single_profile_power(config, processed_data_sample,
                                 sample_id,
                                 i_location)
        write_timing_info('Sample time', time.time() - sample_time)
    print('Processing location (lat {}, lon {}) done.'
          .format(loc[0], loc[1]))
    return power, x_opt, cluster_info


def run_single_location_sample(config, loc, sample_id):
    # Read selected sample for location
    start_time = time.time()
    data = get_wind_data(locs=[loc], sel_sample_ids=[sample_id])

    write_timing_info('Input read', time.time() - since)
    # Run preprocessing
    processed_data_sample = preprocess_data(data,
                                            remove_low_wind_samples=False)
    # Identify i_loc with clustering i_location to read matching
    # clustering output power / control parameters
    i_location = locations.index(loc)
    # TODO timing ( + compare to locaton funct?)
    power, x_opt, cluster_info = \
        single_profile_power(config,
                             processed_data_sample,
                             sample_id,
                             i_location)
    # print('Processing of location (lat {}, lon {}), sample {} done.'
    #      .format(loc[0], loc[1], sample_id))
    write_timing_info('Single sample, single location done after ',
                      time.time() - start_time)
    return power, x_opt, cluster_info


def multiple_locations(config, locs, sel_sample_ids, file_name):
    # TODO replace hard numbers n_x_opt, x_opt, ...
    # p_sample, p_cluster, p_cc, p_cc_opt
    power = np.zeros([4, len(locs), len(sel_sample_ids)])
    # x_opt_sample, x_opt_cluster, x_opt_cc_opt
    x_opt = np.zeros([3, len(locs), len(sel_sample_ids), 5])
    # cluster_id, backscaling
    cluster_info = np.zeros([2, len(locs), len(sel_sample_ids)])
    if not config.Processing.parallel:
        for i_loc, loc in enumerate(locs):
            # TODO include optional multiprocessing for samples?
            power[:, i_loc, :], x_opt[:, i_loc, :, :],\
                cluster_info[:, i_loc, :] = run_location(loc, sel_sample_ids)
    else:
        # Define mapping all locations and all samples, respectively
        # Same location input for all sample ids, one location after the other
        import functools
        funct = functools.partial(run_single_location_sample,
                                  config)
        mapping_iterables = [(loc, sample_id) for loc in locs
                             for sample_id in sel_sample_ids]
        # TODO tqdm include and same as other mutiprocessing
        # TODO include tqdm in production environment
        # TODO timing info: only for single sample production add up?
        # TODO run for the 5k locations? or 1k locations only?
        from multiprocessing import Pool
        with Pool(config.Processing.n_cores) as p:
            mapping_out = p.starmap(
                funct,
                mapping_iterables)
        # Interpret mapping output
        # ??? or write array during maping at different index?
        # -> parallel access
        n_samples = len(sel_sample_ids)
        for i, res_i in enumerate(mapping_out):
            i_loc = i//n_samples
            i_sample = i % n_samples
            power[:, i_loc, i_sample] = res_i[0]
            x_opt[:, i_loc, i_sample, :] = res_i[1]
            cluster_info[:, i_loc, i_sample] = res_i[2]

    # Define result dictionary
    res = {
        'p_sample': power[0, :, :],
        'p_cluster': power[1, :, :],
        'p_cc': power[2, :, :],
        'p_cc_opt': power[3, :, :],
        'x_opt_sample': x_opt[0, :, :, :],
        'x_opt_cluster': x_opt[1, :, :, :],
        'x_opt_cc_opt': x_opt[2, :, :, :],
        'cluster_id': cluster_info[0, :, :],
        'backscaling': cluster_info[1, :, :],
        'locs': locs,
        'sample_ids': sample_ids,
        }
    if np.any(power != -1):
        write_timing_info('All locations done.', time.time() - since)
    else:
        write_timing_info('All locations failed.', time.time() - since)
    # Pickle results
    with open(file_name, 'wb') as f:
        pickle.dump(res, f)
    return res


if __name__ == "__main__":
    from ..config import config
    from .config import sample_ids, \
        brute_force_testing_file_name, locs
    # TODO change to config.yaml

    # TODO: timeing
    # single sample test
    # p_sample, x_opt_sample, cluster_x_opt, \
    #    p_cluster, p_cc = single_profile_power(
    #        0, [(48.75,-12.25)])
    # Single location test
    # iterate_multiple_samples_single_loc(
    #    sample_ids = [0,1,2,3,4,5,6,7,8,9], loc = [(48.75,-12.25)])
    # Mult location, mult sample test
    los = locs[:2]
    res = multiple_locations(config,
                             locs,
                             sample_ids,
                             brute_force_testing_file_name)
