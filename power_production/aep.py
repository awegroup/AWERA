import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from config import file_name_freq_distr, power_curve_output_file_name


def get_mask_discontinuities(df):
    """Identify discontinuities in the power curves. The provided approach is obtained by trial and error and should
    be checked carefully when applying to newly generated power curves."""
    mask = np.concatenate(((True,), (np.diff(df['P [W]']) > -5e2)))
    mask = np.logical_or(mask, df['v_100m [m/s]'] > 10)  # only apply mask on low wind speeds
    if df['P [W]'].iloc[-1] < 0 or df['P [W]'].iloc[-1] - df['P [W]'].iloc[-2] > 5e2:
        mask.iloc[-1] = False
    return ~mask


def plot_power_and_wind_speed_probability_curves(n_clusters=8, loc='mmc', post_process_curves=True):
    """Plot the power and wind speed probability curves for the requested cluster wind resource representation."""
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5.5, 4))
    plt.subplots_adjust(top=0.991, bottom=0.118, left=0.21, right=0.786)

    n_bins = 100
    with open(file_name_freq_distr, 'rb') as f:
        wind_speed_distribution = pickle.load(f)[n_bins]
    wind_speed_bin_freq = wind_speed_distribution['freq_2d']
    wind_speed_bin_limits = wind_speed_distribution['v_bin_limits']

    for i in range(n_clusters):
        # Plot power curve.
        i_profile = i + 1
        df_power_curve = pd.read_csv(power_curve_output_file_name.format(suffix='csv', i_profile=i_profile), sep=";")
        #mask discontinuities check
        if post_process_curves:
            mask_faulty_point = get_mask_discontinuities(df_power_curve)
        else:
            mask_faulty_point = np.array([False] * len(df_power_curve))

        lbl = "{}-{}".format(loc.upper(), i_profile)
        p = ax[0].plot(df_power_curve['v_100m [m/s]'][~mask_faulty_point],
                       df_power_curve['P [W]'][~mask_faulty_point] * 1e-3, '-', label=lbl)
        ax[0].plot(df_power_curve['v_100m [m/s]'][mask_faulty_point],
                   df_power_curve['P [W]'][mask_faulty_point] * 1e-3, 's', color=p[0].get_color())

        # Plot wind speed probability.
        aggregate_n_bins = 4

        v0 = wind_speed_bin_limits[i, :-1:aggregate_n_bins]
        v1 = wind_speed_bin_limits[i, aggregate_n_bins::aggregate_n_bins]
        if len(v0) != len(v1):
            v1 = np.append(v1, wind_speed_bin_limits[i, -1])
        bin_center = (v0 + v1)/2

        freq = np.zeros(len(bin_center))
        for j in range(len(bin_center)):
            freq[j] = np.sum(wind_speed_bin_freq[i, j*aggregate_n_bins:(j+1)*aggregate_n_bins])

        ax[1].step(bin_center, freq/100., where='mid')

    ax[0].set_ylim([0., 11])
    ax[0].grid()
    ax[0].set_ylabel('Mean cycle power [kW]')
    ax[0].legend(bbox_to_anchor=(1.02, 1.05), loc="upper left")
    ax[1].set_ylim([0., 0.0125])
    ax[1].grid()
    ax[1].set_ylabel('Normalised frequency [-]')
    ax[1].set_xlabel('$v_{100m}$ [m s$^{-1}$]')


def plot_aep_matrix(freq, power, aep):
    """Visualize the annual energy production contributions of each wind speed bin."""
    n_clusters = freq.shape[0]
    mask_array = lambda m: np.ma.masked_where(m == 0., m)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 3.5))
    plt.subplots_adjust(top=0.98, bottom=0.05, left=0.065, right=0.98)
    ax[0].set_ylabel("Cluster label [-]")
    ax[0].set_yticks(range(n_clusters))
    ax[0].set_yticklabels(range(1, n_clusters+1))

    for a in ax:
        a.set_xticks((0, freq.shape[1]-1))
        a.set_xticklabels(('cut-in', 'cut-out'))

    im0 = ax[0].imshow(mask_array(freq), aspect='auto')
    cbar0 = plt.colorbar(im0, orientation="horizontal", ax=ax[0], aspect=12, pad=.17)
    cbar0.set_label("Probability [%]")
    im1 = ax[1].imshow(mask_array(power)*1e-3, aspect='auto')
    cbar1 = plt.colorbar(im1, orientation="horizontal", ax=ax[1], aspect=12, pad=.17)
    cbar1.set_label("Power [kW]")
    im2 = ax[2].imshow(mask_array(aep)*1e-6, aspect='auto')
    cbar2 = plt.colorbar(im2, orientation="horizontal", ax=ax[2], aspect=12, pad=.17)
    cbar2.set_label("AEP contribution [MWh]")


def calculate_aep(n_clusters=8, loc='mmc'):
    """Calculate the annual energy production for the requested cluster wind resource representation. Reads the wind
    speed distribution file, then the csv file of each power curve, post-processes the curve, and numerically integrates
    the product of the power and probability curves to determine the AEP."""

    n_bins = 100
    with open(file_name_freq_distr, 'rb') as f:
        wind_speed_distribution = pickle.load(f)[n_bins]
    freq = wind_speed_distribution['freq_2d']
    wind_speed_bin_limits = wind_speed_distribution['v_bin_limits']

    p_bins = np.zeros(freq.shape)
    for i in range(n_clusters):
        i_profile = i + 1
        df = pd.read_csv(power_curve_output_file_name.format(suffix='csv', i_profile=i_profile), sep=";")
        mask_faulty_point = get_mask_discontinuities(df)
        v = df['v_100m [m/s]'].values[~mask_faulty_point]
        p = df['P [W]'].values[~mask_faulty_point]

        # assert v[0] == wind_speed_bin_limits[i, 0] #TODO differences at 10th decimal threw assertion error
        err_str = "Wind speed range of power curve {} is different than that of probability distribution: " \
                  "{:.2f} and {:.2f} m/s, respectively."
        if np.abs(v[0] - wind_speed_bin_limits[i, 0]) > 1e-6:
            print(err_str.format(i_profile, wind_speed_bin_limits[i, 0], v[0]))        
        if np.abs(v[-1] - wind_speed_bin_limits[i, -1]) > 1e-6:
            print(err_str.format(i_profile, wind_speed_bin_limits[i, -1], v[-1]))
        # assert np.abs(v[-1] - wind_speed_bin_limits[i, -1]) < 1e-6, err_str

        # Determine wind speeds at bin centers and corresponding power output.
        v_bins = (wind_speed_bin_limits[i, :-1] + wind_speed_bin_limits[i, 1:])/2.
        p_bins[i, :] = np.interp(v_bins, v, p, left=0., right=0.)

    # Weight profile energy production with the #TODO?? why not only n(cluster1, v_bin)/n_samples but n(cluster1, v_bin)/n(cluster1) 
    aep_bins = p_bins * freq/100. * 24*365
    aep_sum = np.sum(aep_bins)*1e-6
    print("AEP: {:.2f} MWh".format(aep_sum))

    return aep_sum, freq, p_bins, aep_bins


if __name__ == "__main__":
    plot_power_and_wind_speed_probability_curves()
    aep_sum, freq, p_bins, aep_bins = calculate_aep(8, loc='mmc')
    plot_aep_matrix(freq, p_bins, aep_bins)

    plt.show()
