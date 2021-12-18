import pandas as pd
import pickle
import copy
import numpy as np

import sys
import getopt

from .read_requested_data import get_wind_data

from .preprocess_data import preprocess_data
from .wind_profile_clustering import \
    cluster_normalized_wind_profiles_pca, predict_cluster, \
    single_location_prediction
from ..power_production.utils import write_timing_info

import time
since = time.time()
# --------------------------- Cluster Profiles
# TODO cluster naming also with 1 to n_clusters not 0...


def export_wind_profile_shapes(heights, prl, prp,
                               output_file, ref_height=100.):
    assert output_file[-4:] == ".csv"
    df = pd.DataFrame({
        'height [m]': heights,
    })
    scale_factors = []
    for i, (u, v) in enumerate(zip(prl, prp)):
        w = (u**2 + v**2)**.5

        # Get normalised wind speed at reference height via linear
        # interpolation
        w_ref = np.interp(ref_height, heights, w)
        # Scaling factor such that the normalised absolute wind speed
        # at the reference height is 1
        sf = 1/w_ref
        dfi = pd.DataFrame({
            'u{} [-]'.format(i+1): u*sf,
            'v{} [-]'.format(i+1): v*sf,
            'scale factor{} [-]'.format(i+1): sf,
        })
        df = pd.concat((df, dfi), axis=1)

        scale_factors.append(sf)
    df.to_csv(output_file, index=False, sep=";")
    return df


def get_cluster_profiles(config):
    print('Perform full clustering algorithm')
    # Set Data to read to training data
    config_training_data = copy.deepcopy(config)
    config_training_data.update({'Data': config.Clustering.training.__dict__})
    data = get_wind_data(config_training_data)
    del config_training_data

    write_timing_info('Input read.', time.time() - since)

    processed_data = preprocess_data(config, data)
    write_timing_info('Training data preprocessed.', time.time() - since)

    res = cluster_normalized_wind_profiles_pca(
        processed_data['training_data'],
        config.Clustering.n_clusters,
        n_pcs=config.Clustering.n_pcs)
    prl, prp = res['clusters_feature']['parallel'], \
        res['clusters_feature']['perpendicular']
    write_timing_info('Clustering trained.', time.time() - since)
    # Free up some memory
    del processed_data

    processed_data_full = preprocess_data(config,
                                          data,
                                          remove_low_wind_samples=False)
    write_timing_info('Preprocessed full data.', time.time() - since)

    labels, frequency_clusters = predict_cluster(
        processed_data_full['training_data'], config.Clustering.n_clusters,
        res['data_processing_pipeline'].predict, res['cluster_mapping'])
    write_timing_info('Predicted full data', time.time() - since)

    backscaling = np.array([np.interp(config.Clustering.preprocessing.ref_vector_height,
                            processed_data_full['altitude'],
                            processed_data_full['wind_speed'][i_sample, :]) for i_sample in range(processed_data_full['wind_speed'].shape[0])])

    # Write cluster labels to file
    cluster_info_dict = {
        'n clusters': config.Clustering.n_clusters,
        'n samples': len(labels),
        'n pcs': config.Clustering.n_pcs,
        'labels [-]': labels,
        'cluster_mapping': res['cluster_mapping'],
        'backscaling [m/s]': backscaling,
        'training_data_info': config.Clustering.training.data_info,
        'locations': processed_data_full['locations'],
        'n_samples_per_loc': processed_data_full['n_samples_per_loc']
        }
    pickle.dump(cluster_info_dict, open(config.IO.training_cluster_labels, 'wb'))

    # TODO add option to predict already read in training data,
    # split prediction function to read input and run prediction
    # Write mapping pipeline to file
    pipeline = res['data_processing_pipeline']
    pickle.dump(pipeline, open(config.IO.cluster_pipeline, 'wb'))

    wind_profile_shapes = export_wind_profile_shapes(
            data['altitude'],
            prl, prp,
            config.IO.cluster_profiles,
            ref_height=config.Clustering.preprocessing.ref_vector_height)
    write_timing_info('Output written. Finished.', time.time() - since)
    return wind_profile_shapes, cluster_info_dict, pipeline
# --------------------------- Matching Cluster Prediction



def predict_cluster_labels(config):
    print('Predict cluster labels with existing clustering pipeline')
    # TODO this can also be done step by step for the data
    # or in parallel - fill labels incrementally
    with open(config.IO.cluster_pipeline, 'rb') as f:
        pipeline = pickle.load(f)
    # Sort cluster labels same as mapping from training
    # (largest to smallest cluster):
    with open(config.IO.training_cluster_labels, 'rb') as f:
        training_labels_file = pickle.load(f)
    cluster_mapping = training_labels_file['cluster_mapping']
    locations = config.Data.locations
    do_parallel = config.Processing.parallel
    # Unset parallel processing: reading input in single process
    # cannot start new processes for reading input in parallel
    setattr(config.Processing, 'parallel', False)
    n_samples_per_loc = get_wind_data(
        config,
        locs=[locations[0]])['n_samples_per_loc']
    res_labels = np.zeros(len(locations)*n_samples_per_loc)
    res_scale = np.zeros(len(locations)*n_samples_per_loc)
    write_timing_info('Setup completed. Start prediction...',
                      time.time() - since)
    if do_parallel:
        # TODO no import here
        # TODO check if parallel ->
        from multiprocessing import get_context
        from tqdm import tqdm
        import functools
        funct = functools.partial(single_location_prediction,
                                  config,
                                  pipeline,
                                  cluster_mapping)
        if config.Processing.progress_out == 'stdout':
            file = sys.stdout
        else:
            file = sys.stderr
        # Start multiprocessing Pool
        # use spawn instead of fork: pipeline can be used by child processes
        # otherwise same key/lock on pipeline object - leading to infinite loop
        with get_context("spawn").Pool(config.Processing.n_cores) as p:
            results = list(tqdm(p.imap(funct, locations),
                                total=len(locations), file=file))
            # TODO is this more RAM intensive?
            for i, val in enumerate(results):
                res_labels[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = \
                    val[0]
                res_scale[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = val[1]
        setattr(config.Processing, 'parallel', True)
    else:
        import functools
        funct = functools.partial(single_location_prediction,
                                  config,
                                  pipeline,
                                  cluster_mapping)
        # TODO add progress bar
        for i, loc in enumerate(locations):
            res_labels[i*n_samples_per_loc:(i+1)*n_samples_per_loc], \
                res_scale[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = \
                funct(loc)
    write_timing_info('Predicted full data.', time.time() - since)
    # Write cluster labels to file
    cluster_info_dict = {
        'n clusters': config.Clustering.n_clusters,
        'n samples': len(locations)*n_samples_per_loc,
        'n pcs': config.Clustering.n_pcs,
        'labels [-]': res_labels,
        'cluster_mapping': cluster_mapping,
        'backscaling [m/s]': res_scale,
        'training_data_info': training_labels_file['training_data_info'],
        'locations': locations,
        'n_samples_per_loc': n_samples_per_loc,
        }
    pickle.dump(cluster_info_dict, open(config.IO.cluster_labels, 'wb'))
    write_timing_info('Predicted labels written. Done.',
                      time.time() - since)

# --------------------------- Cluster Frequency


def export_single_loc_frequency_distribution(config,
                                             labels_full,
                                             backscaling,
                                             n_samples,
                                             write_output=True):
    cut_wind_speeds = pd.read_csv(
        config.IO.refined_cut_wind_speeds)
    freq_2d = np.zeros((config.Clustering.n_clusters,
                        config.Clustering.n_wind_speed_bins))
    v_bin_limits = np.zeros((config.Clustering.n_clusters,
                             config.Clustering.n_wind_speed_bins+1))
    for i_c in range(config.Clustering.n_clusters):
        v = np.linspace(cut_wind_speeds['vw_100m_cut_in'][i_c],
                        cut_wind_speeds['vw_100m_cut_out'][i_c],
                        config.Clustering.n_wind_speed_bins+1)
        v_bin_limits[i_c, :] = v

        # procedure consistent with the wind property used for characterizing
        # the cut-in and cut-out wind speeds, i.e. wind speed at 100m height.
        for j, (v0, v1) in enumerate(zip(v[:-1], v[1:])):
            # Denormalised assigned cluster wind speed at 100m per sample
            samples_in_bin = (labels_full == i_c) & \
                (backscaling >= v0) & \
                (backscaling < v1)
            freq_2d[i_c, j] = np.sum(samples_in_bin) / n_samples * 100.

    distribution_data = {'frequency': freq_2d,
                         'wind_speed_bin_limits': v_bin_limits,
                         }

    if write_output:
        with open(config.IO.freq_distr, 'wb') as f:
            pickle.dump(distribution_data, f, protocol=2)

    return freq_2d, v_bin_limits


def location_wise_frequency_distribution(config,
                                         locations,
                                         labels, n_samples, n_samples_per_loc,
                                         backscaling):
    #if len(locations) > 1:
    distribution_data = {
        'frequency': np.zeros((len(locations),
                               config.Clustering.n_clusters,
                               config.Clustering.n_wind_speed_bins)),
        'locations': locations
        }
    for i, loc in enumerate(locations):
        distribution_data['frequency'][i, :, :], \
            wind_speed_bin_limits = \
            export_single_loc_frequency_distribution(
                config,
                labels[n_samples_per_loc*i: n_samples_per_loc*(i+1)],
                backscaling[n_samples_per_loc*i:
                            n_samples_per_loc*(i+1)],
                n_samples_per_loc,
                write_output=False)
    distribution_data['wind_speed_bin_limits'] = wind_speed_bin_limits

    with open(config.IO.freq_distr, 'wb') as f:
        pickle.dump(distribution_data, f, protocol=2)
    # TODO ??? different np.zeros for single location? inconsistent
    # else:
    #     freq_2d, v_bin_limits = \
    #         export_single_loc_frequency_distribution(config,
    #                                                  labels,
    #                                                  backscaling,
    #                                                  n_samples,
    #                                                  backscaling)


def export_frequency_distr(config):
    # TODO make this also parallel/serial, not all input at same time
    print('Exporting frequency distribution only')
    profiles_file = pd.read_csv(
        config.IO.cluster_profiles, sep=";")
    scale_factors = []
    for i in range(config.Clustering.n_clusters):
        scale_factors.append(profiles_file['scale factor{} [-]'
                                           .format(i+1)][0])
    with open(config.IO.cluster_labels, 'rb') as f:
        labels_file = pickle.load(f)
    labels = labels_file['labels [-]']
    n_samples = len(labels)
    backscaling = labels_file['backscaling [m/s]']
    locations = labels_file['locations']
    n_samples_per_loc = labels_file['n_samples_per_loc']
    write_timing_info('Input read.', time.time() - since)

    location_wise_frequency_distribution(config,
                                         locations,
                                         labels,
                                         n_samples,
                                         n_samples_per_loc,
                                         backscaling)
    write_timing_info('Output written. Finished.', time.time() - since)


# --------------------------- Standalone Functionality
def export_profiles_and_probability(config):
    if config.Clustering.make_profiles:
        get_cluster_profiles(config)
        print('Make profiles done.')

    if config.Clustering.predict_labels:
        predict_cluster_labels(config)
        print('Predict labels done.')

    if config.Clustering.make_freq_distr:
        export_frequency_distr(config)
        print('Frequency distribution done.')


def interpret_input_args():
    make_profiles, make_freq_distr, predict_labels = (False, False, False)
    if len(sys.argv) > 1:  # User input was given
        a = "python export_profiles_and_probability.py"
        help = """
        {}    : run clustering, save both profiles and frequency distributions
        {} -p : run clustering, save new profiles
        {} -f : export frequency distributions
        {} -l : use existing pipeline to predict cluster labels for new data
        {} -h : display this help
        """.format(a, a, a, a, a)
        try:
            opts, args = getopt.getopt(sys.argv[1:],
                                       "hpfl",
                                       ["help", "profiles",
                                        "frequency", "labels"])
        except getopt.GetoptError:  # User input not given correctly
            print(help)
            sys.exit()
        for opt, arg in opts:
            if opt in ("-h", "--help"):  # Help argument called
                print(help)
                sys.exit()
            elif opt in ("-p", "--profiles"):
                make_profiles = True
            elif opt in ("-f", "--frequency"):
                make_freq_distr = True
            elif opt in ("-l", "--labels"):
                predict_labels = True
    else:
        make_profiles = True
        make_freq_distr = True

    return make_profiles, make_freq_distr, predict_labels


if __name__ == '__main__':
    from ..config import config
    # Read program parameters
    make_profiles, make_freq_distr, predict_labels = interpret_input_args()
    setattr(config.Clustering, 'make_profiles', make_profiles)
    setattr(config.Clustering, 'make_freq_distr', make_freq_distr)
    setattr(config.Clustering, 'predict_labels', predict_labels)
    export_profiles_and_probability(config)
