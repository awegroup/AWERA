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
        if n_profiles > 1:
            cmap = plt.get_cmap("gist_ncar")
            if n_profiles > 25:
                if i_profile % 2 == 1:
                    if n_profiles % 2 == 1:
                        shift = -1
                    else:
                        shift = 0
                    i_c = - i_profile + shift
                else:
                    i_c = i_profile
            else:
                i_c = i_profile
            clrs = cmap(np.linspace(0.03, 0.97, n_profiles))
            color = clrs[i_c]
        else:
            color = 'orangered'
        df_profile = pd.read_csv(config.IO.power_curve.format(
            i_profile=i_profile+1, suffix='csv'), sep=";")
        wind_speeds = df_profile['v_100m [m/s]']
        power = df_profile['P [W]']

        # Plot power
        ax_pcs[0].plot(wind_speeds, power/1000, label=i_profile+1, color=color)

        # Frequency Plot for profile
        sel_loc_id = -1  # 343  # TODO make optional
        if sel_loc_id != -1:
            freq = frequency[sel_loc_id, i_profile, :]
            wind_speed_bins = wind_speed_bin_limits[i_profile, :]
        elif len(locations) > 1:
            # Mult locations
            freq = np.sum(frequency[:, i_profile, :], axis=0)/len(locations)
            wind_speed_bins = wind_speed_bin_limits[i_profile, :]
        else:  # TODO this is not right anymore?
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
        ax_pcs[1].step(wind_speed_step, freq_step, label=i_profile+1,
                       color=color)
        freq_sum += sum(freq)

    print('Sum of frequencies: ', freq_sum)

    ax_pcs[1].legend()
    ax_pcs[0].tick_params(labelbottom=False)
    x_label = '$v_{w,' + str(config.General.ref_height) + 'm}$ [m/s]'
    ax_pcs[1].set_xlabel(x_label)
    ax_pcs[0].set_ylabel('Mean cycle Power [kW]')
    ax_pcs[1].set_ylabel('Cluster frequency [%]')
    # TODO paper: nomalised frequency?
    for ax in ax_pcs:
        ax.set_xlim([5, 21])
    if not config.Plotting.plots_interactive:
        plt.savefig(config.IO.plot_output.format(
                title='power_curves_and_freq'))
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
    from ..config import Config
    config = Config()
    plot_power_and_frequency(config)
    # plot_optimization_parameter_scatter(config)
    # TODO include savefig
    if config.Plotting.plots_interactive:
        plt.show()
