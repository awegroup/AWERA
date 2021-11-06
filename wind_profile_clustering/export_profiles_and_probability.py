import pandas as pd
import pickle
import numpy as np

import sys
import getopt

from config_clustering import n_clusters, n_pcs, file_name_cluster_profiles, \
    file_name_freq_distr, file_name_cluster_labels, \
    file_name_cluster_pipeline, data_info, training_cluster_labels, \
    training_cluster_pipeline, training_cluster_profiles,\
    training_refined_cut_wind_speeds_file
from config_production import \
    refined_cut_wind_speeds_file as cut_wind_speeds_file
from read_requested_data import get_wind_data

from preprocess_data import preprocess_data
from wind_profile_clustering import \
    cluster_normalized_wind_profiles_pca, predict_cluster

from utils import write_timing_info
n_wind_speed_bins = 100


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
    return scale_factors


def export_frequency_distribution(cut_wind_speeds_file, output_file,
                                  labels_full, normalisation_wind_speeds,
                                  n_samples, normalisation_wind_speed_scaling,
                                  n_wind_speed_bins=100, write_output=True):
    # TODO make optional training or full.
    cut_wind_speeds = pd.read_csv(training_refined_cut_wind_speeds_file)
    freq_2d = np.zeros((n_clusters, n_wind_speed_bins))
    v_bin_limits = np.zeros((n_clusters, n_wind_speed_bins+1))
    for i_c in range(n_clusters):
        v = np.linspace(cut_wind_speeds['vw_100m_cut_in'][i_c],
                        cut_wind_speeds['vw_100m_cut_out'][i_c],
                        n_wind_speed_bins+1)
        v_bin_limits[i_c, :] = v
        # Re-scaling to make the normalisation winds used in the clustering
        sf = normalisation_wind_speed_scaling[i_c]
        # procedure consistent with the wind property used for characterizing
        # the cut-in and cut-out wind speeds, i.e. wind speed at 100m height.
        for j, (v0, v1) in enumerate(zip(v[:-1], v[1:])):
            # Denormalised assigned cluster wind speed at 100m per sample
            samples_in_bin = (labels_full == i_c) & \
                (normalisation_wind_speeds/sf >= v0) & \
                (normalisation_wind_speeds/sf < v1)
            freq_2d[i_c, j] = np.sum(samples_in_bin) / n_samples * 100.

    distribution_data = {'frequency': freq_2d,
                         'wind_speed_bin_limits': v_bin_limits,
                         }

    if write_output:
        with open(output_file, 'wb') as f:
            pickle.dump(distribution_data, f, protocol=2)

    return freq_2d, v_bin_limits


def location_wise_frequency_distribution(locations, n_wind_speed_bins,
                                         labels, n_samples, n_samples_per_loc,
                                         scale_factors, normalisation_values):
    if len(locations) > 1:
        distribution_data = {'frequency': np.zeros((len(locations),
                                                    n_clusters,
                                                    n_wind_speed_bins)),
                             'wind_speed_bin_limits': np.zeros(
                                 (len(locations),
                                  n_clusters,
                                  n_wind_speed_bins+1)),
                             'locations': locations
                             }
        for i, loc in enumerate(locations):
            distribution_data['frequency'][i, :, :], \
                distribution_data['wind_speed_bin_limits'][i, :, :] = \
                export_frequency_distribution(
                    cut_wind_speeds_file,
                    file_name_freq_distr,
                    labels[n_samples_per_loc*i: n_samples_per_loc*(i+1)],
                    normalisation_values[n_samples_per_loc*i:
                                         n_samples_per_loc*(i+1)],
                    n_samples_per_loc,
                    scale_factors,
                    n_wind_speed_bins=n_wind_speed_bins,
                    write_output=False)

        with open(file_name_freq_distr, 'wb') as f:
            pickle.dump(distribution_data, f, protocol=2)
    else:
        freq_2d, v_bin_limits = export_frequency_distribution(
            cut_wind_speeds_file, file_name_freq_distr, labels,
            normalisation_values, n_samples, scale_factors)


def single_location_prediction(pipeline, cluster_mapping, loc):
    data = get_wind_data(locs=[loc], parallel=False)
    # write_timing_info('Input read.', time.time() - since)

    processed_data_full = preprocess_data(
        data, remove_low_wind_samples=False)
    # TODO no make copy here -> need less RAM
    # write_timing_info('Preprocessed full data.', time.time() - since)

    labels_unsorted = pipeline.predict(
        processed_data_full['training_data'])
    n_samples = len(labels_unsorted)
    labels = np.zeros(n_samples).astype(int)
    for i_new, i_old in enumerate(cluster_mapping):
        labels[labels_unsorted == i_old] = i_new
    # write_timing_info('Predicted full data', time.time() - since)
    return labels, processed_data_full['normalisation_value']


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
    # Read program parameters
    make_profiles, make_freq_distr, predict_labels = interpret_input_args()

    import time
    since = time.time()
    if predict_labels:
        print('Predict cluster labels with existing clustering pipeline')
        from config_clustering import locations
        # TODO this can also be done step by step for the data
        # or in parallel - fill labels incrementally
        with open(training_cluster_pipeline, 'rb') as f:
            pipeline = pickle.load(f)
        # Sort cluster labels same as mapping from training
        # (largest to smallest cluster):
        with open(training_cluster_labels, 'rb') as f:
            training_labels_file = pickle.load(f)
        cluster_mapping = training_labels_file['cluster_mapping']
        n_samples_per_loc = get_wind_data(locs=[locations[0]],
                                          parallel=False)['n_samples_per_loc']
        res_labels = np.zeros(len(locations)*n_samples_per_loc)
        res_norm = np.zeros(len(locations)*n_samples_per_loc)
        write_timing_info('Setup completed. Start prediction...',
                          time.time() - since)
        from multiprocessing import Pool
        from tqdm import tqdm
        n_cores = 40  # TODO in config
        import functools
        funct = functools.partial(single_location_prediction, pipeline,
                                  cluster_mapping)
        with Pool(n_cores) as p:
            results = list(tqdm(p.imap(funct, locations),
                                total=len(locations), file=sys.stdout))
            # TODO make optional output to sys.stdout if run on Condor
            # also in read-in
            # TODO is this more RAM intensive?
            for i, val in enumerate(results):
                res_labels[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = \
                    val[0]
                res_norm[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = val[1]
        write_timing_info('Predicted full data.', time.time() - since)
        # Write cluster labels to file
        cluster_info_dict = {
            'n clusters': n_clusters,
            'n samples': len(locations)*n_samples_per_loc,
            'n pcs': n_pcs,
            'labels [-]': res_labels,
            'cluster_mapping': cluster_mapping,
            'normalisation value [-]': res_norm,
            'training_data_info': training_labels_file['training_data_info'],
            'locations': locations,
            'n_samples_per_loc': n_samples_per_loc,
            }
        pickle.dump(cluster_info_dict, open(file_name_cluster_labels, 'wb'))
        write_timing_info('Predicted labels written. Done.',
                          time.time() - since)

    if not make_profiles and make_freq_distr:
        # TODO make this also parallel/serial, not all input at same time
        print('Exporting frequency distribution only')
        profiles_file = pd.read_csv(training_cluster_profiles, sep=";")
        scale_factors = []
        for i in range(n_clusters):
            scale_factors.append(profiles_file['scale factor{} [-]'
                                               .format(i+1)][0])
        with open(file_name_cluster_labels, 'rb') as f:
            labels_file = pickle.load(f)
        labels = labels_file['labels [-]']
        n_samples = len(labels)
        normalisation_values = labels_file['normalisation value [-]']
        locations = labels_file['locations']
        n_samples_per_loc = labels_file['n_samples_per_loc']
        write_timing_info('Input read.', time.time() - since)

        location_wise_frequency_distribution(locations, n_wind_speed_bins,
                                             labels, n_samples,
                                             n_samples_per_loc, scale_factors,
                                             normalisation_values)
        write_timing_info('Output written. Finished.', time.time() - since)

    elif make_profiles:
        print('Perform full clustering algorithm')

        data = get_wind_data(parallel=True)
        # TODO parallel from config in read data funct definition?
        write_timing_info('Input read.', time.time() - since)

        processed_data = preprocess_data(data)
        write_timing_info('Training data preprocessed.', time.time() - since)

        res = cluster_normalized_wind_profiles_pca(
            processed_data['training_data'], n_clusters, n_pcs=n_pcs)
        prl, prp = res['clusters_feature']['parallel'], \
            res['clusters_feature']['perpendicular']
        write_timing_info('Clustering trained.', time.time() - since)
        print('preprocessing refs:', sys.getrefcount(processed_data))
        # Free up some memory
        del processed_data
        # TODO can we remove stuff from memory here?
        processed_data_full = preprocess_data(data,
                                              remove_low_wind_samples=False)
        write_timing_info('Preprocessed full data.', time.time() - since)

        labels, frequency_clusters = predict_cluster(
            processed_data_full['training_data'], n_clusters,
            res['data_processing_pipeline'].predict, res['cluster_mapping'])
        write_timing_info('Predicted full data', time.time() - since)

        # Write cluster labels to file
        cluster_info_dict = {
            'n clusters': n_clusters,
            'n samples': len(labels),
            'n pcs': n_pcs,
            'labels [-]': labels,
            'cluster_mapping': res['cluster_mapping'],
            'normalisation value [-]':
                processed_data_full['normalisation_value'],
            'training_data_info': data_info,
            'locations': processed_data_full['locations'],
            'n_samples_per_loc': processed_data_full['n_samples_per_loc']
            }
        pickle.dump(cluster_info_dict, open(file_name_cluster_labels, 'wb'))

        # Write mapping pipeline to file
        pipeline = res['data_processing_pipeline']
        pickle.dump(pipeline, open(file_name_cluster_pipeline, 'wb'))

        scale_factors = export_wind_profile_shapes(data['altitude'],
                                                   prl, prp,
                                                   file_name_cluster_profiles)
        if make_freq_distr:
            location_wise_frequency_distribution(
                processed_data_full['locations'], n_wind_speed_bins, labels,
                processed_data_full['n_samples'],
                processed_data_full['n_samples_per_loc'], scale_factors,
                processed_data_full['normalisation_value'])
        write_timing_info('Output written. Finished.', time.time() - since)
