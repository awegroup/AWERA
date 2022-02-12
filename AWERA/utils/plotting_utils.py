import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs

from ..resource_analysis.plot_maps import eval_contour_fill_levels, \
    plot_single_panel

def plot_discrete_map(config, values, title='', label='', log_scale=False):
    # Map range
    cm = 1/2.54
    # Plot Value
    plt.figure(figsize=(13*cm, 14.5*cm))
    mrc = ccrs.Mercator()
    ax = plt.axes(projection=mrc)

    ax.coastlines(zorder=4)
    # TODO resolution='50m', color='black', linewidth=1)
    ax.set_extent([config.Data.lon_range[0],
                   config.Data.lon_range[1],
                   config.Data.lat_range[0],
                   config.Data.lat_range[1]])
    plt.title(title)

    color_map = plt.get_cmap('YlOrRd')
    print(np.min(values), np.max(values))
    normalize = mpl.colors.Normalize(vmin=np.min(values),
                                     vmax=np.max(values))
    if log_scale:
        normalize = mpl.colors.LogNorm(vmin=np.min(values),
                                       vmax=np.max(values))
    else:
        normalize = mpl.colors.Normalize(vmin=np.min(values),
                                         vmax=np.max(values))

    lons_grid, lats_grid = np.meshgrid(config.Data.all_lons,
                                       config.Data.all_lats)
    # Compute map projection coordinates.
    # TODO does this work now? threw error too many values to unpack
    # - works locally
    ax.pcolormesh(lons_grid, lats_grid, values,
                  cmap=color_map, norm=normalize,
                  transform=cartopy.crs.PlateCarree(),
                  zorder=3)
    cbar_ax, _ = mpl.colorbar.make_axes(ax)
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=color_map, norm=normalize,
                              label=label)


def plot_single_map(data, title='', label='',
                    data_type='v',
                    fill_range=None,
                    line_levels=[2, 5, 15, 20],
                    n_decimals=0,
                    plot_item=None,
                    log_scale=False):
    plot_title = title
    if plot_item is None:
        if fill_range is None:
            # TODO make this work with masked arrays
            fill_range = [min(data.flat), max(data.flat)]
        linspace00 = np.linspace(fill_range[0], fill_range[1], 21)
        plot_item = {
            'data': data,
            'contour_fill_levels': linspace00,
            'contour_line_levels': line_levels,
            'contour_line_label_fmt': '%.{}f'.format(n_decimals),
            'colorbar_ticks': linspace00[::4],
            'colorbar_tick_fmt': '{:.1f}',
            'colorbar_label': label,
            'extend': 'max',
            'log_scale': log_scale,
        }
    eval_contour_fill_levels([plot_item])
    plot_single_panel(plot_item, plot_title=plot_title)


def plot_map(config, data, title='', label='', log_scale=False):
    # TODO cleanup - what is needed in the end?

    # TODO reinclude map plotting
    # Match locations with values - rest NaN
    n_lats = len(config.Data.all_lats)
    n_lons = len(config.Data.all_lons)
    data_loc = np.ma.array(np.zeros((n_lats, n_lons)), mask=True)
    # TODO right way around?
    for i, i_loc in enumerate(config.Data.i_locations):
        data_loc[i_loc[0], i_loc[1]] = data[i]
    if np.sum(data_loc.mask) == 0:
        # Plot continuous aep map
        print('Full map determined. Plot map:')
        plot_single_map(data_loc, title=title, label=label,
                        log_scale=log_scale)
        # TODO add other options
    else:
        plot_discrete_map(config,
                          data_loc,
                          title=title,
                          label=label,
                          log_scale=log_scale)
    return data_loc


##############


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


##############


def plot_diff_pdf(data, diff_type, parameter,
                  unit, output_file_name='diff_pdf.pdf', title='',
                  plots_interactive=False):
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
    if len(abs_res.shape) == 2:
        # Data and error given
        ax1.errorbar(x, abs_res[:, 0],
                     yerr=abs_res[:, 1], fmt='+', color=color)
    else:
        # Only data
        ax1.plot(x, abs_res, '+', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot relative difference in same plot
    ax2 = ax1.twinx()
    x = np.array(x) + 0.2
    color = 'tab:orange'
    ax2.set_ylabel('rel diff [-]', color=color)
    if len(rel_res.shape) == 2:
        # Data and error given
        ax2.errorbar(x, rel_res[:, 0],
                     yerr=rel_res[:, 1], fmt='+', color=color)
    else:
        # Only data
        ax2.plot(x, rel_res, '+', color=color)
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

    if not plot_config['plots_interactive']:
        plt.savefig(plot_config['output_file_name'])

