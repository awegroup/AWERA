import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs

from .convenience_utils import hour_to_date

from ..resource_analysis.plot_maps import eval_contour_fill_levels, \
    plot_single_panel

def plot_discrete_map(config, values, title='', label='', log_scale=False,
                      overflow=None, plot_colorbar=True):
    # Map range
    cm = 1/2.54
    # Plot Value
    plt.figure(figsize=(13*cm, 14.5*cm))
    mrc = ccrs.Mercator()
    ax = plt.axes(projection=mrc)

    ax.coastlines(zorder=4)
    # TODO resolution='50m', color='black', linewidth=1)
    # ax.set_extent([config.Data.lon_range[0],
    #                config.Data.lon_range[1],
    #                config.Data.lat_range[0],
    #                config.Data.lat_range[1]])
    plt.title(title)

    color_map = plt.get_cmap('YlOrRd')
    ticks = None
    if overflow is not None:
        ticks = []
        n_normal = 224
        n_over = 32
        top_overflow = overflow
        colors_underflow = []
        underflow_bounds = []
        min_val = np.min(values)
        ticks += [min_val]
        if isinstance(overflow, list):
            top_overflow = overflow[1]
            min_val = overflow[0]
            ticks += [min_val]
            n_over = int(n_over/2)
            colors_underflow = list(plt.get_cmap('coolwarm')(
                np.linspace(0, 0.21, n_over)))
            underflow_bounds = list(np.linspace(np.min(values), min_val,
                                                n_over+1))[:-1]
        colors_normal = list(color_map(np.linspace(0, .9, n_normal)))
        colors_overflow = list(
            plt.get_cmap('Greens')(np.linspace(0.5, 1, n_over)))
        all_colors = colors_underflow + colors_normal + colors_overflow
        color_map = mpl.colors.LinearSegmentedColormap.from_list(
            'color_map', all_colors)
        normal_bounds = list(np.linspace(min_val,
                                         top_overflow, n_normal+1))[:-1]
        ticks += list(np.linspace(min_val,
                                  top_overflow, 7))[1:]
        ticks += [np.max(values)]
        overflow_bounds = list(np.linspace(top_overflow,
                                           np.max(values), n_over))
        bounds = underflow_bounds + normal_bounds + overflow_bounds
        normalize = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    elif log_scale:
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
    if plot_colorbar:
        cbar_ax, _ = mpl.colorbar.make_axes(ax)
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=color_map, norm=normalize,
                                  label=label, ticks=ticks)


def match_loc_data_map_data(config, data):
    # Match locations with values - rest NaN
    n_lats = len(config.Data.all_lats)
    n_lons = len(config.Data.all_lons)
    data_loc = np.ma.array(np.zeros((n_lats, n_lons)), mask=True)
    # TODO right way around?
    for i, i_loc in enumerate(config.Data.i_locations):
        data_loc[i_loc[0], i_loc[1]] = data[i]
    return data_loc


def plot_single_map(data, title='', label='',
                    fill_range=None,
                    # TODO also make default None line levels?
                    line_levels=None,
                    n_decimals=0,
                    plot_item=None,
                    overflow=None,
                    log_scale=False):
    plot_title = title
    if plot_item is None:
        if fill_range is None:
            fill_range = [np.min(data), np.max(data)]
            print(fill_range)
        linspace00 = np.linspace(fill_range[0], fill_range[1], 21)

        if overflow is not None:
            ticks = []
            # TODO what if overflow less than min val?
            min_val = np.min(data)
            if overflow < min_val:
                raise ValueError('Overflow smaller than minimal data value')
            ticks += [min_val]
            top_overflow = overflow
            linspace00 = [min_val]
            if isinstance(overflow, list):
                min_val = overflow[0]
                top_overflow = overflow[1]
                ticks += [min_val]
                linspace00 += list(np.linspace(ticks[0], ticks[1], 3))[1:]
            ticks += list(np.linspace(min_val,
                                      top_overflow, 5))[1:]
            linspace00 += list(np.linspace(min_val,
                                           top_overflow, 16))[1:]
            linspace00 += list(np.linspace(top_overflow, np.max(data), 4))[1:]
            ticks += [np.max(data)]
        else:
            ticks = linspace00[::4]
        print('Ticks: ', ticks)
        if line_levels is None:
            line_levels = ticks
            if overflow is not None:
                line_levels = line_levels[1:-2]

        plot_item = {
            'data': data,
            'contour_fill_levels': linspace00,
            'contour_line_levels': line_levels,
            'contour_line_label_fmt': '%.{}f'.format(n_decimals),
            'colorbar_ticks': ticks,
            'colorbar_tick_fmt': '{' + ':.{}f'.format(n_decimals) + '}',
            'colorbar_label': label,
            'extend': 'max',
            'log_scale': log_scale,
        }
    eval_contour_fill_levels([plot_item])
    plot_single_panel(plot_item, plot_title=plot_title,
                      overflow=overflow)


def plot_map(config, data, title='', label='', log_scale=False,
             plot_continuous=False, line_levels=None,  # [2, 5, 15, 20],
             fill_range=None, n_decimals=0,
             overflow=None,
             output_file_name='map.pdf',
             plot_colorbar=True):
    # TODO cleanup - what is needed in the end?

    # TODO reinclude map plotting
    # Match locations with values - rest masked
    if data.shape != (len(config.Data.all_lats), len(config.Data.all_lons)):
        data_loc = match_loc_data_map_data(config, data)
    else:
        data_loc = data
    if np.ma.isMaskedArray(data_loc):
        data_continuous = np.sum(data_loc.mask) == 0
    else:
        data_continuous = True
    if data_continuous or plot_continuous:
        # Plot continuous map
        print('Full map determined. Plot map:')
        plot_single_map(data_loc, title=title, label=label,
                        line_levels=line_levels,
                        fill_range=fill_range,
                        log_scale=log_scale,
                        n_decimals=n_decimals,
                        overflow=overflow)
        # TODO add other options
    else:
        plot_discrete_map(config,
                          data_loc,
                          title=title,
                          label=label,
                          log_scale=log_scale,
                          overflow=overflow,
                          plot_colorbar=plot_colorbar)
    if not config.Plotting.plots_interactive:
        plt.savefig(output_file_name)
    return data_loc


##############
def plot_percentiles(perc_5, perc_32, perc_50, output_file_name=None):
    """" Generate power density percentile plot. """
    from ..resource_analysis.plot_maps import eval_contour_fill_levels, \
        plot_panel_1x3_seperate_colorbar
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]

    linspace0 = np.linspace(0, .033, 21)
    plot_item0 = {
        'data': perc_5*1e-3,
        'contour_fill_levels': linspace0,
        'contour_line_levels': sorted([.003]+list(linspace0[::5])),
        'contour_line_label_fmt': '%.3f',
        'colorbar_ticks': linspace0[::5],
        'colorbar_tick_fmt': '{:.3f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    linspace1 = np.linspace(0, .45, 21)
    plot_item1 = {
        'data': perc_32*1e-3,
        'contour_fill_levels': linspace1,
        'contour_line_levels': sorted([.04]+list(linspace1[::4])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace1[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    linspace2 = np.linspace(0, 1, 21)
    plot_item2 = {
        'data': perc_50*1e-3,
        'contour_fill_levels': linspace2,
        'contour_line_levels': sorted([.1]+list(linspace2[::4])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace2[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)
    if output_file_name is not None:
        plt.savefig(output_file_name)


def plot_percentile_ratios(perc_5, perc_32, perc_50,
                           plot_range=[1, 2.7],
                           output_file_name=None):
    """" Generate power density percentile plot. """
    # TODO move functions here? / to plot maps utils file?
    from ..resource_analysis.plot_maps import eval_contour_fill_levels, \
        plot_panel_1x3_seperate_colorbar
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]

    linspace0 = np.linspace(plot_range[0], plot_range[1], 21)
    plot_item0 = {
        'data': perc_5,
        'contour_fill_levels': linspace0,
        'contour_line_levels': [1.2, 1.35],  #sorted(list(linspace0[::5])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace0[::5],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density ratio [-]',
    }
    linspace1 = np.linspace(plot_range[0], plot_range[1], 21)
    plot_item1 = {
        'data': perc_32,
        'contour_fill_levels': linspace1,
        'contour_line_levels': [1.2, 1.5],  # sorted(list(linspace1[::4])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace1[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density ratio [-]',
    }
    linspace2 = np.linspace(plot_range[0], plot_range[1], 21)
    plot_item2 = {
        'data': perc_50,
        'contour_fill_levels': linspace2,
        'contour_line_levels': [1.2, 1.5, 1.8],  # sorted(list(linspace2[::4])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace2[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density ratio [-]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)
    if output_file_name is not None:
        plt.savefig(output_file_name)
##################
# SINGLE LOCATION
def plot_timeline(hours, data,
                  ylabel='Power [W]',
                  #heights_of_interest, ceiling_id, floor_id,
                  #data_bounds=[50, 500],
                  show_n_hours=24*7,
                  output_file_name='power_timeline.pdf',
                  plots_interactive=False):
    # TODO rename, docstring
    # TODO check passing plots interactive
    """Plot optimal height and wind speed time series for the first week of data.

    Args:
        hours (list): Hour timestamps.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        optimal_heights (list): Time series of optimal heights corresponding to `v_ceiling`.
        heights_of_interest (list): Heights above the ground at which the wind speeds are evaluated.
        ceiling_id (int): Id of the ceiling height in `heights_of_interest`, as used in the variable-height analysis.
        floor_id (int): Id of the floor height in `heights_of_interest`, as used in the variable-height analysis.

    """
    shift = int(1e4)
    # TODO update docstring
    # TODO optional time range, not only from beginning
    # TODO heights_of_interest use cases fix -> height_bounds
    data = data[shift:shift+show_n_hours]
    dates = [hour_to_date(h) for h in hours[shift:shift+show_n_hours]]

    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=.2)

    # Plot the height limits.
    dates_limits = [dates[0], dates[-1]]
    # ceiling_height = height_bounds[1]  # heights_of_interest[ceiling_id]
    # floor_height = height_bounds[0]  # heights_of_interest[floor_id]
    # ax[0].plot(dates_limits, [ceiling_height]*2, 'k--', label='height bounds')
    # ax[0].plot(dates_limits, [floor_height]*2, 'k--')

    # Plot the optimal height time series.
    ax.plot(dates, data, color='darkcyan')

    # Plot the markers at the points for which the wind profiles are plotted
    # in figure 5b.
    # TODO make optional
    # marker_ids = [list(hours).index(h) for h in hours_wind_profile_plots]
    # for i, h_id in enumerate(marker_ids):
    #    ax[0].plot(dates[h_id], optimal_heights[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=8,
    #               markeredgewidth=2, markerfacecolor='None')

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time')
    # ax.set_ylim([0, 800])
    ax.grid()
    # ax.legend()

    ax.set_xlim(dates_limits)

    # plt.axes(ax[1])
    plt.xticks(rotation=70)
    if not plots_interactive:
        plt.savefig(output_file_name)



def plot_optimal_height_and_wind_speed_timeline(
        hours,
        v_ceiling,
        optimal_heights,
        #heights_of_interest, ceiling_id, floor_id,
        height_range=None,
        ref_velocity=None,
        height_bounds=[200, 500],
        v_bounds=[None, None],
        show_n_hours=24*7,
        shift=int(1e4),
        plots_interactive=False,
        output_file_name='optimal_height_and_wind_speed_timeline.pdf'):
    # TODO rename, update Docstring
    """Plot optimal height and wind speed time series for the first week of data.

    Args:
        hours (list): Hour timestamps.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        optimal_heights (list): Time series of optimal heights corresponding to `v_ceiling`.
        heights_of_interest (list): Heights above the ground at which the wind speeds are evaluated.
        ceiling_id (int): Id of the ceiling height in `heights_of_interest`, as used in the variable-height analysis.
        floor_id (int): Id of the floor height in `heights_of_interest`, as used in the variable-height analysis.

    """
    # TODO optional time range, not only from beginning
    # TODO heights_of_interest use cases fix -> height_bounds
    optimal_heights = optimal_heights[shift:shift+show_n_hours]

    if not isinstance(hours[0], np.datetime64):
        dates = [hour_to_date(h) for h in hours[shift:shift+show_n_hours]]
    else:
        dates = hours[shift:shift+show_n_hours]
    v_ceiling = v_ceiling[shift:shift+show_n_hours]

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    plt.subplots_adjust(bottom=.2)

    # Plot the height limits.
    dates_limits = [dates[0], dates[-1]]
    ceiling_height = height_bounds[1]  # heights_of_interest[ceiling_id]
    floor_height = height_bounds[0]  # heights_of_interest[floor_id]
    if ceiling_height > 0:
        ax[0].plot(dates_limits,
                   [ceiling_height]*2, 'k--',
                   label='height bounds')
    if floor_height > 0:
        ax[0].plot(dates_limits, [floor_height]*2, 'k--')

    # Plot the optimal height time series.
    ax[0].plot(dates, optimal_heights, color='darkcyan', label='AWES height')
    if height_range is not None:
        ax[0].plot(dates, height_range['min'][shift:shift+show_n_hours],
                   color='darkcyan', alpha=0.25)
        ax[0].plot(dates, height_range['max'][shift:shift+show_n_hours],
                   color='darkcyan', alpha=0.25, label='max/min AWES height')
    print('heights plotted...')
    # Plot the markers at the points for which the wind profiles are plotted
    # in figure 5b.
    # TODO make optional
    #marker_ids = [list(hours).index(h) for h in hours_wind_profile_plots]
    #for i, h_id in enumerate(marker_ids):
    #    ax[0].plot(dates[h_id], optimal_heights[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=8,
    #               markeredgewidth=2, markerfacecolor='None')

    ax[0].set_ylabel('Height [m]')
    # TODO automatize ylim
    ax[0].set_ylim([0, 600])
    ax[0].grid()
    ax[0].legend()

    if ref_velocity is not None:
        print(ref_velocity.shape)
        ref_velocity = ref_velocity[shift:shift+show_n_hours]
        ax[1].plot(dates, ref_velocity, alpha=0.5, label='@ ref. height')
        print('ref velocity plotted')
    if v_bounds[0] is not None:
        ax[1].plot(dates_limits,
                   [v_bounds[0]]*2, 'k--',
                   label='wind speed bounds')
    if v_bounds[1] is not None:
        ax[1].plot(dates_limits,
                   [v_bounds[1]]*2, 'k--')
    # Plot the optimal wind speed time series.
    ax[1].plot(dates, v_ceiling, label='@ AWES height', color='darkcyan')

    ax[1].legend()
    #for i, h_id in enumerate(marker_ids):
    #    ax[1].plot(dates[h_id], v_ceiling[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=8,
    #               markeredgewidth=2, markerfacecolor='None')
    ax[1].set_ylabel('Wind speed [m/s]')
    ax[1].grid()
    ax[1].set_xlim(dates_limits)
    print('wind speeds plotted...')
    plt.axes(ax[1])
    plt.xticks(rotation=70)
    if not plots_interactive:
        plt.savefig(output_file_name)
    return dates

###########################################

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
                            title='', plots_interactive=False):
    # TODO check passing plots interactive?
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
    if 'figsize' in plot_config.keys():
        figsize = plot_config['figsize']
    else:
        figsize = (5.5, 3.7)
    fig, ax1 = plt.subplots(figsize=figsize)

    # Adding title
    plt.title(plot_config['title'])
    if 'x_ticks' in plot_config.keys():
        x = np.array(range(len(x_vals)))
    else:
        x = x_vals
    color = 'tab:blue'
    if 'x_label' in plot_config.keys():
        ax1.set_xlabel(plot_config['x_label'])
    if 'y_label' in plot_config.keys():
        ax1.set_ylabel(plot_config['y_label'], color=color)
    else:
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
    if 'y_lim' not in plot_config:
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
    else:
        ax1.set_ylim(plot_config['y_lim'][0])
        ax2.set_ylim(plot_config['y_lim'][1])
    ax1.axhline(0, linewidth=0.5, color='grey')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if not plot_config['plots_interactive']:
        plt.savefig(plot_config['output_file_name'])


def plot_diff_step_wise(x_vals, diff,
                        eval_components=[8],
                        diff_type='absolute',
                        eval_type='clusters',
                        **plot_config):

    x = np.array(range(len(x_vals)))

    if not isinstance(eval_components, list):
        eval_components = [eval_components]

    fig, ax = plt.subplots()

    if 'x_label' in plot_config.keys():
        ax.set_xlabel(plot_config['x_label'])

    if 'y_label' in plot_config.keys():
        ax.set_ylabel(plot_config['y_label'])
    else:
        if diff_type == 'absolute':
            plt.ylabel('absolute diff [m/s]')
        else:
            plt.ylabel('relative diff [-]')

    if 'y_lim' in plot_config.keys():
        ax.set_ylim(plot_config['y_lim'])

    plot_dict = {}
    for idx, n in enumerate(eval_components):
        y = diff[idx][:, 0]
        dy = diff[idx][:, 1]
        if len(eval_components) > 1:
            shift = -0.25 + 0.5/(len(eval_components)-1) * idx
        else:
            shift = 0
        shifted_x = x+shift
        if np.ma.isMaskedArray(y):
            y = y[~y.mask]
            dy = dy[~y.mask]
            shifted_x = shifted_x[~y.mask]
        plot_dict[n] = plt.errorbar(shifted_x, y, yerr=dy, fmt='+')

    if 'x_ticks' in plot_config.keys():
        ax.set_xticks(x)
        ax.set_xticklabels(plot_config['x_ticks'])

    legend_list = [plot_item for key, plot_item in plot_dict.items()]
    legend_names = ['{} {}'.format(key, eval_type) for key in plot_dict]
    plt.legend(legend_list, legend_names)

    if not plot_config['plots_interactive']:
        plt.savefig(plot_config['output_file_name'])
