import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pickle


def plot_power_and_frequency(config):
    n_profiles = config.Clustering.n_clusters
    fig, ax_pcs = plt.subplots(2, 1)
    for a in ax_pcs:
        a.grid()

    with open(config.IO.freq_distr, 'rb') as f:
        freq_distr = pickle.load(f)
    frequency = freq_distr['frequency']
    locations = config.Data.locations
    freq_sum = 0
    wind_speed_bin_limits = freq_distr['wind_speed_bin_limits']

    for i_profile in range(n_profiles):
        df_profile = pd.read_csv(config.IO.power_curve.format(
            i_profile=i_profile+1, suffix='csv'), sep=";")
        wind_speeds = df_profile['v_100m [m/s]']
        power = df_profile['P [W]']

        # Filter spikey results
        while True:
            mask_power_disc = [True] + list(np.diff(power) > -500)
            power = power[mask_power_disc]
            wind_speeds = wind_speeds[mask_power_disc]
            if sum(mask_power_disc) == len(mask_power_disc):
                # No more discontinuities
                break
            print('Power masked')
        # Plot power
        ax_pcs[0].plot(wind_speeds, power/1000, label=i_profile+1)

        # Frequency Plot for profile
        sel_loc_id = 685  # TODO make optional
        if sel_loc_id != -1:
            freq = frequency[sel_loc_id, i_profile, :]
            wind_speed_bins = wind_speed_bin_limits[sel_loc_id, i_profile, :]
        elif len(locations) > 1:
            # Mult locations
            freq = np.sum(frequency[:, i_profile, :], axis=0)/len(locations)
            wind_speed_bins = wind_speed_bin_limits[0, i_profile, :]
        else:
            freq = frequency[i_profile, :]
            wind_speed_bins = wind_speed_bin_limits[i_profile, :]

        wind_speed_bins = wind_speed_bins[:-1] + np.diff(wind_speed_bins)/2
        # Normalise frequency #TODO need this?
        # freq = freq / np.sum(freq)

        # combine 4 bins
        freq_step = np.zeros(int(len(freq)/4))
        for i in range(int(len(freq)/4)):
            freq_step[i] = np.sum(freq[i*4:i*4+4])
        wind_speed_step = wind_speed_bins[::4]
        # Plot frequency
        ax_pcs[1].step(wind_speed_step, freq_step, label=i_profile+1)
        freq_sum += sum(freq)

    print('Sum of frequencies: ', freq_sum)

    ax_pcs[1].legend()
    ax_pcs[0].tick_params(labelbottom=False)
    ax_pcs[1].set_xlabel('$v_{w,100m}$ [m/s]')
    ax_pcs[0].set_ylabel('Mean cycle Power [kW]')
    ax_pcs[1].set_ylabel('Cluster frequency [%]')
    # TODO paper: nomalised frequency?
    for ax in ax_pcs:
        ax.set_xlim([5, 21])
    # plt.show()


def plot_optimization_parameter_scatter(config, param_ids=[1, 2, 3]):
    n_profiles = config.Clustering.n_clusters
    optimizer_var_names = [
        'F_out [N]', 'F_in [N]',
        'theta_out [rad]', 'dl_tether [m]', 'l0_tether [m]']
    optimizer_var_labels = [
        '$F_{out}$ [N]', '$F_{in}$ [N]', '$\Theta$$_{out}$ [rad]',
        '$\Delta$ $L_{tether}$ [m]', '$L_{0,tether}$ [m]']
    fig = plt.figure()

    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    for i_profile in range(n_profiles):
        # Read optimization results
        df_profile = pd.read_csv(config.IO.power_curve.format(
            i_profile=i_profile+1, suffix='csv'), sep=";")
        x_profile = df_profile[optimizer_var_names[param_ids[0]]]
        y_profile = df_profile[optimizer_var_names[param_ids[1]]]
        z_profile = df_profile[optimizer_var_names[param_ids[2]]]
        ax.scatter(x_profile, y_profile, z_profile, marker='.')
    ax.set_xlabel(optimizer_var_labels[param_ids[0]])
    ax.set_ylabel(optimizer_var_labels[param_ids[1]])
    ax.set_zlabel(optimizer_var_labels[param_ids[2]])


if __name__ == '__main__':
    from ..config import config
    plot_power_and_frequency(config)
    # plot_optimization_parameter_scatter(config)
    # TODO include savefig
    if config.Plotting.plots_interactive:
        plt.show()
