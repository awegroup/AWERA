import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_frequency(config):
    n_profiles = config.Clustering.n_clusters
    fig, ax = plt.subplots(1, 1)
    ax.grid()

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

        # Frequency Plot for profile
        sel_loc_id = -1  # 343  # TODO make optional
        if sel_loc_id != -1:
            freq = frequency[sel_loc_id, i_profile, :]
            wind_speed_bins = wind_speed_bin_limits[i_profile, :]
        else:
            # Mult locations
            freq = np.sum(frequency[:, i_profile, :], axis=0)/len(locations)
            wind_speed_bins = wind_speed_bin_limits[i_profile, :]

        wind_speed_bins = wind_speed_bins[:-1] + np.diff(wind_speed_bins)/2
        # Normalise frequency #TODO need this?
        # freq = freq / np.sum(freq)

        # combine 4 bins
        freq_step = np.zeros(int(len(freq)/4))
        print(len(freq))
        for i in range(int(len(freq)/4)):
            freq_step[i] = np.sum(freq[i*4:i*4+4])
        wind_speed_step = wind_speed_bins[::4]

        if n_profiles > 17:
            # Cycle linestyles fpr better readability
            lines = ["-", "-", "--", "--", "-.", "-.", ":", ":",
                     (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5))]
            linestyle = lines[(i_profile+1) % 10]
        else:
            linestyle = '-'

        # Plot frequency
        ax.step(wind_speed_step, freq_step, label=i_profile+1,
                color=color, linestyle=linestyle)
        freq_sum += sum(freq)

        if n_profiles > 17 and \
                ((i_profile+1) % 10 == 0 or i_profile == n_profiles - 1):
            # Plot every 10 together
            ax.legend()
            x_label = '$v_{w,' + str(config.General.ref_height) + 'm}$ [m/s]'
            ax.set_xlabel(x_label)
            ax.set_ylabel('Cluster frequency [%]')
            # TODO paper: nomalised frequency?
            ax.set_xlim([0, 30])
            if not config.Plotting.plots_interactive:
                plt.savefig(config.IO.plot_output.format(
                        title='freq_up_to_{}'.format(i_profile+1)))

            fig, ax = plt.subplots(1, 1)
            ax.grid()

    print('Sum of frequencies: ', freq_sum)

    if n_profiles <= 17:
        ax.legend()
        x_label = '$v_{w,' + str(config.General.ref_height) + 'm}$ [m/s]'
        ax.set_xlabel(x_label)
        ax.set_ylabel('Cluster frequency [%]')
        # TODO paper: nomalised frequency?
        # TODO add custom
        ax.set_xlim([0, 30])
        if not config.Plotting.plots_interactive:
            plt.savefig(config.IO.plot_output.format(
                    title='freq'))