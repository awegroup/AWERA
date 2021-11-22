import numpy as np
import matplotlib.pyplot as plt

# !!! from config_production import plots_interactive
plots_interactive = True


def get_velocity_bin_mask(wind_speed,
                          split_velocities=[0, 1.5, 3, 5, 10, 20, 0]):
    """Find masks for the samples matching each velocity bin.

    Parameters
    ----------
    wind_speed : ndarray
        1D (or nD) array containing wind speed values for each sample
        (and each location, ...).
    split_velocities : list, optional
        List of wind speeds defining the velocity bin intervals. The last value
        is used as lower bound to inf. If the last value is lower than the
        second to last value, both are used as lower bound to inf.
        The default is [0, 1.5, 3, 5, 10, 20, 0].

    Returns
    -------
    masks : list(ndarray)
        List of mask array containing the masks for each wind speed bin.
    tags : list(string)
        List of name tags combined via '_' for each wind speed bin.

    """
    masks = [wind_speed]*len(split_velocities)
    tags = ['']*len(split_velocities)
    for vel_idx, vel in enumerate(split_velocities):
        # TODO similar structure here and in clustering validation possible?
        # Evaluate velocity bin steps
        if split_velocities[-1] < split_velocities[-2]:
            # Last two steps not increasing: take velocity-up for both
            idx_vel_up = [len(split_velocities)-2, len(split_velocities)-1]
        else:
            idx_vel_up = [len(split_velocities)-1]
        # Find mask
        if vel_idx in idx_vel_up:
            masks[vel_idx] = wind_speed < vel
            tags[vel_idx] = ' vel {} up'.format(vel)
        else:
            masks[vel_idx] = np.logical_or(
                (wind_speed < vel),
                (wind_speed >= split_velocities[vel_idx+1]))
            tags[vel_idx] = ' vel {} to {}'.format(
                vel, split_velocities[vel_idx+1])
    return masks, tags


#############################################
# ---------------- PLOTTING -----------------
def hist_data_eval_mean_std(txt_x_pos, data, unit):
    if np.ma.isMaskedArray(data):
        n_points = sum((~data.mask).flat)
    else:
        n_points = len(data.flat)
    # Print values properly formatted
    n_text = '{} entries | '.format(n_points)
    if np.abs(np.mean(data)) < 1e-3 or np.std(data) < 1e-3:
        mean_std_text = r'mean: ${:.2E}\pm{:.2E}$ {}'
    elif np.abs(np.mean(data)) > 1e3 or np.std(data) > 1e3:
        mean_std_text = r'mean: ${:.5}\pm{:.5}$ {}'
    else:
        mean_std_text = r'mean: ${:.3}\pm{:.3}$ {}'
    text = n_text + mean_std_text.format(np.mean(data), np.std(data), unit)
    return text


def plot_diff_pdf(data, diff_type, parameter,
                  unit, output_file_name='diff_pdf.pdf', title=''):
    """Plot pdf of differences.

    Parameters
    ----------
    data : list
        Sample data for one height/diff type/wind_orientation.
    wind_orientation : string
        Evaluated wind orientation (parallel, perpendicualar, absolute).
    diff_type : string
        Evaluated difference type (absolute, relative,
        if 'no' no 'diff' in plot label).
    parameter : string
        Parameter name
    unit : string
        Parameter unit
    output_file_name : string, optional
        Path to save pdf. The default is 'diff_pdf_height.pdf'.
    title : string, optional
        Plot title. The default is ''.

    Returns
    -------
    None.

    """
    plt.figure()
    # Fill histogram
    hist = plt.hist(data, bins=100)
    plt.grid()
    plt.title(title)
    # Plot labels
    if diff_type == 'no':
        plt.xlabel('{} {}'.format(parameter, unit))
    else:
        plt.xlabel('{} diff for {} {}'.format(diff_type, parameter, unit))
    plt.ylabel('frequency')
    # Add mean and standard deviation text
    # Find good text position
    x_vals, y_vals = hist[1], hist[0]
    max_index = np.argmax(y_vals)
    if max_index > len(x_vals)/2:
        txt_x_pos = x_vals[2]
    else:
        txt_x_pos = x_vals[int(len(x_vals)/2)]

    if np.ma.isMaskedArray(data):
        n_points = sum((~data.mask).flat)
    else:
        n_points = len(data.flat)
    # Print values properly formatted
    plt.text(txt_x_pos, plt.ylim()[1]*0.9,
             '{} entries'.format(n_points), color='tab:blue')
    if np.abs(np.mean(data)) < 1e-3 or np.std(data) < 1e-3:
        mean_std_text = r'mean: ${:.2E}\pm{:.2E}$ {}'
    elif np.abs(np.mean(data)) > 1e3 or np.std(data) > 1e3:
        mean_std_text = r'mean: ${:.5}\pm{:.5}$ {}'
    else:
        mean_std_text = r'mean: ${:.3}\pm{:.3}$ {}'
    plt.text(txt_x_pos, plt.ylim()[1]*0.85, mean_std_text.format(
        np.mean(data), np.std(data), unit), color='tab:blue')
    # Save output file
    if not plots_interactive:
        plt.savefig(output_file_name)


#TODO combine with diff_pdf
def plot_diff_pdf_mult_data(data_list, diff_type, parameter, unit,
                            data_type='', output_file_name='diff_pdf.pdf',
                            title=''):
    """Plot pdf of differences.

    Parameters
    ----------
    data_list : list
        Sample data.
    data_type : list(string)
        List of legend entries matching the data in data_list.
    diff_type : string
        Evaluated difference type (absolute, relative,
        if 'no': no 'diff' in plot label).
    parameter : string
        Parameter name
    unit : string
        Parameter unit
    output_file_name : string, optional
        Path to save pdf. The default is 'diff_pdf.pdf'.
    title : string, optional
        Plot title. The default is ''.

    Returns
    -------
    None.

    """
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive','tab:cyan']
    colors = colors[:len(data_list)]
    plt.figure()
    # Fill histogram
    n_bins = 100
    if len(data_list) > 1:
        filled = False
    else:
        filled = True

    hist = plt.hist(data_list, n_bins, histtype='step', color=colors,
                    label=data_type, fill=filled)
    for i, data in enumerate(data_list):
        # Find good text position
        if i == 0:
            x_vals, y_vals = hist[1], hist[0][i]
            max_index = np.argmax(y_vals)
            if max_index > len(x_vals)/2:
                txt_x_pos = x_vals[2]
            else:
                txt_x_pos = x_vals[int(len(x_vals)/2)+2]
        # Add mean and standard deviation text
        plt.text(txt_x_pos, plt.ylim()[1]*(0.9-0.05*i),
                 hist_data_eval_mean_std(txt_x_pos, data, unit),
                 color=colors[i])
    if len(data_list) > 1:
        # include legend
        plt.legend(loc='center right')

    plt.grid()
    plt.title(title)
    # Plot labels
    if diff_type == 'no':
        plt.xlabel('{} {}'.format(parameter, unit))
    else:
        plt.xlabel('{} diff for {} {}'.format(diff_type, parameter, unit))
    plt.ylabel('frequency')

    # Save output file
    if not plots_interactive:
        plt.savefig(output_file_name)


def plot_abs_rel_step_wise(x_vals, abs_res, rel_res, **plot_config):
    """Plot double y-axis for absolute and relative differences.

    Parameters
    ----------
    x_vals : list/1darray
        List of x-values.
    abs_res : list/1darray
        Absolute differences for each x-value.
    rel_res : list/1darray
        Relative differences for each x-value.
    **plot_config : dictionary
        Optional plotting parameter: x_ticks, x_label, output_file_name.

    Returns
    -------
    None.

    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Adding title
    plt.title(plot_config['title'])
    if 'x_ticks' in plot_config.keys():
        x = np.array(range(len(x_vals)))
    else:
        x = x_vals
    color = 'tab:blue'
    if 'x_label' in plot_config.keys():
        ax1.set_xlabel(plot_config['x_label'])
    ax1.set_ylabel('abs diff [m/s]', color=color)
    ax1.errorbar(x, abs_res[:, 0], yerr=abs_res[:, 1], fmt='+', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot relative difference in same plot
    ax2 = ax1.twinx()
    x = x + 0.2
    color = 'tab:orange'
    ax2.set_ylabel('rel diff [-]', color=color)
    ax2.errorbar(x, rel_res[:, 0], yerr=rel_res[:, 1], fmt='+', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if 'x_ticks' in plot_config.keys():
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_config['x_ticks'])
    # Symmetrize both y axes - matching 0 point in center
    y1 = ax1.get_ylim()
    if np.abs(y1[0]) > np.abs(y1[1]):
        ax1.set_ylim((y1[0], -y1[0]))
    else:
        ax1.set_ylim((-y1[1], y1[1]))
    y2 = ax2.get_ylim()
    if np.abs(y2[0]) > np.abs(y2[1]):
        ax2.set_ylim((y2[0], -y2[0]))
    else:
        ax2.set_ylim((-y2[1], y2[1]))
    ax1.axhline(0, linewidth=0.5, color='grey')

    if not plots_interactive:
        plt.savefig(plot_config['output_file_name'])
