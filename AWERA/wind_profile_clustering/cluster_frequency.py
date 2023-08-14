import pandas as pd
import pickle
import numpy as np

# --------------------------- Cluster Frequency


def export_single_loc_frequency_distribution(config,
                                             labels_full,
                                             backscaling,
                                             n_samples,
                                             write_output=True,
                                             bounds=None):
    if bounds is None:
        cut_wind_speeds = pd.read_csv(
            config.IO.refined_cut_wind_speeds)
    freq_2d = np.zeros((config.Clustering.n_clusters,
                        config.Clustering.n_wind_speed_bins))
    v_bin_limits = np.zeros((config.Clustering.n_clusters,
                             config.Clustering.n_wind_speed_bins+1))
    for i_c in range(config.Clustering.n_clusters):
        if bounds is None:
            v = np.linspace(cut_wind_speeds['vw_100m_cut_in'][i_c],
                            cut_wind_speeds['vw_100m_cut_out'][i_c],
                            config.Clustering.n_wind_speed_bins+1)
        else:
            v = np.linspace(bounds[0],
                            bounds[1],
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
    print('Sum of freq:', np.sum(freq_2d))

    return freq_2d, v_bin_limits


def location_wise_frequency_distribution(config,
                                         locations,
                                         labels, n_samples, n_samples_per_loc,
                                         backscaling,
                                         bounds=None):
    #if len(locations) > 1:
    n_samples_per_loc = int(n_samples_per_loc)
    distribution_data = {
        'frequency': np.zeros((len(locations),
                               config.Clustering.n_clusters,
                               config.Clustering.n_wind_speed_bins)),
        'locations': locations
        }
    for i, loc in enumerate(locations):
        print(i, loc, n_samples_per_loc)
        distribution_data['frequency'][i, :, :], \
            wind_speed_bin_limits = \
            export_single_loc_frequency_distribution(
                config,
                labels[n_samples_per_loc*i: n_samples_per_loc*(i+1)],
                backscaling[n_samples_per_loc*i:
                            n_samples_per_loc*(i+1)],
                n_samples_per_loc,
                write_output=False,
                bounds=bounds)
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
    return distribution_data


def export_frequency_distr(config):
    # TODO make this also parallel/serial, not all input at same time
    print('Exporting frequency distribution only')
    profiles_file = pd.read_csv(
        config.IO.profiles, sep=";")
    scale_factors = []
    for i in range(config.Clustering.n_clusters):
        scale_factors.append(profiles_file['scale factor{} [-]'
                                           .format(i+1)][0])
    with open(config.IO.labels, 'rb') as f:
        labels_file = pickle.load(f)
    labels = labels_file['labels [-]']
    n_samples = len(labels)
    backscaling = labels_file['backscaling [m/s]']
    locations = labels_file['locations']
    n_samples_per_loc = labels_file['n_samples_per_loc']

    location_wise_frequency_distribution(config,
                                         locations,
                                         labels,
                                         n_samples,
                                         n_samples_per_loc,
                                         backscaling)
