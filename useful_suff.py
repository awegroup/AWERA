# Pickle
# Read the power curve constructor or other pickled dictionaries
import pickle
import pandas as pd
import numpy as np

file = 'test.pickle'
with open(file, 'rb') as f:
     res = pickle.load(f)
print(res)
# locations = res['locations']
# or res.kinematics


# Wind profile shapes
# u,v -> normalised at 100m altitude the abs wind speed is normalised to 1

# replace config stuff with file names etc.
def evaluate_aep(config):
    """Calculate the annual energy production for the requested cluster wind
    resource representation. Reads the wind speed distribution file, then
    the csv file of each power curve, post-processes the curve, and
    numerically integrates the product of the power and probability curves
    to determine the AEP."""
    n_clusters = config.Clustering.n_clusters
    with open(config.IO.freq_distr, 'rb') as f:
        freq_distr = pickle.load(f)
    freq_full = freq_distr['frequency']
    wind_speed_bin_limits = freq_distr['wind_speed_bin_limits']

    loc_aep = []
    loc_aep_sq = []
    p_n = []
    for i_loc, loc in enumerate(config.Data.locations):
        # Select location data
        freq = freq_full[i_loc, :, :]

        p_bins = np.zeros(freq.shape)

        for i in range(n_clusters):
            i_profile = i + 1
            # Read power curve file
            # TODO make optional trianing / normal
            df = pd.read_csv(config.IO.power_curve
                             .format(suffix='csv',
                                     i_profile=i_profile),
                             sep=";")
            # TODO drop? mask_faulty_point = get_mask_discontinuities(df)
            v = df['v_100m [m/s]'].values  # .values[~mask_faulty_point]
            p = df['P [W]'].values  # .values[~mask_faulty_point]
            if i_loc == 0:
                # Once extract nominal (maximal) power of cluster
                p_n.append(np.max(p))

                # assert v[0] == wind_speed_bin_limits[i, 0]
                # TODO differences at 10th decimal threw assertion error
                err_str = "Wind speed range of power curve {} is different"\
                    " than that of probability distribution: " \
                    "{:.2f} and {:.2f} m/s, respectively."
                if np.abs(v[0] - wind_speed_bin_limits[i, 0]) > 1e-6:
                    print(err_str.format(i_profile,
                                         wind_speed_bin_limits[i, 0], v[0]))
                if np.abs(v[-1] - wind_speed_bin_limits[i, -1]) > 1e-6:
                    print(err_str.format(i_profile,
                                         wind_speed_bin_limits[i, -1], v[-1]))
                # assert np.abs(v[-1] -
                #      wind_speed_bin_limits[i, -1]) < 1e-6, err_str
            # Determine wind speeds at bin centers and respective power output.
            v_bins = (wind_speed_bin_limits[i, :-1]
                      + wind_speed_bin_limits[i, 1:])/2.
            p_bins[i, :] = np.interp(v_bins, v, p, left=0., right=0.)

        # !!! p bins for all clusters combined with freq distribution gets average power curve
        avg_p_bins = p_bins * freq/100.

        # Weight profile energy production with the frequency of the cluster
        # sum(freq) < 100: non-operation times included
        aep_bins = p_bins * freq/100. * 24*365
        aep_sum = np.sum(aep_bins)*1e-6
        loc_aep.append(aep_sum)

        aep_bins_sq = (p_bins * freq/100. * 24*365) **2

        aep_sq_sum = np.sqrt(np.sum(aep_bins_sq))*1e-6
        loc_aep_sq.append(aep_sq_sum)
        if i_loc % 100 == 0:
            print("AEP: {:.2f} MWh".format(aep_sum),
                  'AEP squared in sum {:.2f} MWh, {:.2f}%'.format(
                      aep_sq_sum, aep_sq_sum/aep_sum*100))
            # plot_aep_matrix(config, freq, p_bins, aep_bins,
            #                 plot_info=(config.Data.data_info+str(i_loc)))
            # print(('AEP matrix plotted for location number'
            #       ' {} of {} - at lat {}, lon {}').format(i_loc,
            #                                               config.Data.n_locs,
            #                                               loc[0], loc[1]))
    # Estimate perfectly running & perfect conditions: nominal power
    # get relative cluster frequency:
    rel_cluster_freq = np.sum(freq_full, axis=(0, 2))/config.Data.n_locs
    print('Cluster frequency:', rel_cluster_freq)
    print('Cluster frequency sum:', np.sum(rel_cluster_freq))
    # Scale up to 1: run full time with same relative impact of clusters
    rel_cluster_freq = rel_cluster_freq/np.sum(rel_cluster_freq)
    aep_n_cluster = np.sum(np.array(p_n)*rel_cluster_freq)*24*365*1e-6
    aep_n_max = np.max(p_n)*24*365*1e-6
    print('Nominal aep [MWh]:', aep_n_cluster, aep_n_max)
    # TODO write aep to file
    return loc_aep, aep_n_max
    # TODO nominal AEP -> average power and nominal power



