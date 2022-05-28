#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate map plots.
Example::

    $ python plot_maps.py -c           : plot from files from combined
                                         output file
    $ python plot_maps.py -m max_id    : plot from files with maximal subset
                                         id of max_id
    $ python plot_maps.py -h           : display this help

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter

import cartopy
import cartopy.crs as ccrs

# General plot settings.
cline_label_format_default = '%.1f'
n_fill_levels_default = 14
n_line_levels_default = 6
color_map = plt.get_cmap('coolwarm')  # 'coolwarm' 'RdBu' 'bwr'  # plt.get_cmap('YlOrRd')

if __name__ == "__main__":
    # TODO this has to be changed to work via class!
    # from ..utils.convenience_utils import hour_to_date_str
    from plotting_utils import read_dataset_user_input

    # Load the processed data from the NetCDF files specified in the input.
    nc = read_dataset_user_input()
    # TODO remove - use config for this!
    lons = nc['longitude'].values
    lats = nc['latitude'].values

    height_range_floor = 50.
    height_range_ceilings = list(nc['height_range_ceiling'].values)
    fixed_heights = list(nc['fixed_height'].values)
    integration_range_ids = list(nc['integration_range_id'].values)
    p_integral_mean = nc['p_integral_mean'].values
    # Hours since 1900-01-01 00:00:00, see: print(nc['time'].values).
    hours = nc['time'].values
    # print("Analyzing " + hour_to_date_str(hours[0]) + " till "
    #       + hour_to_date_str(hours[-1]))
    # TODO fix from config
else:
    lons = list(np.arange(-20, 20.25, .25))
    lats = list(np.arange(65, 29.75, -.25))

#     #lons = np.arange(-12, -5.0, .25)  # config.Data.all_lons
#     #lats = np.arange(51, 56.25, .25)  # config.Data.all_lats
# else:
#     # TODO make more understandable
#     # TODO make into utils -> use for map plots in production
#     # TODO fix from config

#     # Ireland
#     # lons = list(np.arange(-12, -5.0, .25))  # -5.75, .25))
#     # lats = list(np.arange(51, 56.25, .25))
#     # Europe map
#     lons = list(np.arange(-20, 20.25, .25))
#     lats = list(np.arange(65, 29.75, -.25))

# Plotting map - region selection # TODO rework -> config
plot_northern_germany = False
label_cities = False


map_plot_aspect_ratio = 9 / 12.5  # len(lons)/len(lats) # TODO this makes sense - adapt fixed number later on -> adaptable

mrc = ccrs.Mercator()


def calc_fig_height(fig_width, subplot_shape, plot_frame_top,
                    plot_frame_bottom, plot_frame_left, plot_frame_right):
    """"Calculate figure height, such that all maps have the same resolution.

    Args:
        fig_width (float): Figure width in inches.
        subplot_shape (tuple of int): Containing number of rows and columns of
            subplot.
        plot_frame_top (float): Top of plot as a fraction of the figure window
            height w.r.t. bottom.
        plot_frame_bottom (float): Bottom of plot as a fraction of the figure
            window height w.r.t. bottom.
        plot_frame_left (float): Left side of plot as a fraction of the figure
            window width w.r.t. left.
        plot_frame_right (float): Right side of plot as a fraction of the
            figure window width w.r.t. left.

    Returns:
        float: Figure height in inches.

    """
    plot_frame_width = fig_width*(plot_frame_right - plot_frame_left)
    plot_frame_height = plot_frame_width/(map_plot_aspect_ratio *
                                          subplot_shape[1] / subplot_shape[0])
    fig_height = plot_frame_height/(plot_frame_top - plot_frame_bottom)
    return fig_height


def eval_contour_fill_levels(plot_items):
    """"Evaluate the plot data, e.g. if values are within contour fill
        levels limits.

    Args:
        plot_items (list of dict): List containing the plot property dicts.

    """
    for i, item in enumerate(plot_items):
        max_value = np.amax(item['data'])
        min_value = np.amin(item['data'])
        print("Max and min value of plot"
              " {}: {:.3f} and {:.3f}".format(i, max_value, min_value))
        if item['contour_fill_levels'][-1] < max_value:
            print("Contour fills "
                  "(max={:.3f}) do not cover max value of plot {}"
                  .format(item['contour_fill_levels'][-1], i))
        if item['contour_fill_levels'][0] > min_value:
            print("Contour fills "
                  "(min={:.3f}) do not cover min value of plot {}"
                  .format(item['contour_fill_levels'][0], i))


def individual_plot(z, cf_lvls, cl_lvls,
                    cline_label_format=cline_label_format_default,
                    log_scale=False,
                    extend="neither",
                    overflow=None):
    """"Individual plot of coastlines and contours.

    Args:
        z (ndarray): 2D array containing contour plot data.
        cf_lvls (list): Contour fill levels.
        cl_lvls (list): Contour line levels.
        cline_label_format (str, optional): Contour line label format string.
            Defaults to `cline_label_format_default`.
        log_scale (bool): Logarithmic scaled contour levels are used if True,
            linearly scaled if False.
        extend (str): Setting for extension of contour fill levels.

    Returns:
        QuadContourSet: Contour fills object.

    """
    # Care if colorbar ticks are set beforehand, see plot_single_map
    # colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
    # colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))

    # # combine them and build a new colormap
    # colors_stack = np.vstack((colors_undersea, colors_land))
    # color_map = colors.LinearSegmentedColormap.from_list('color_map',
    #                                                      colors_stack)
    color_map = plt.get_cmap('coolwarm')  # YlOrRd
    if overflow is not None:
        n_normal = 224
        n_over = 32
        top_overflow = overflow
        colors_underflow = []
        underflow_bounds = []
        min_val = np.min(z)
        if isinstance(overflow, list):
            top_overflow = overflow[1]
            min_val = overflow[0]
            n_over = int(n_over/2)
            colors_underflow = list(plt.get_cmap('coolwarm')(
                np.linspace(0, 0.21, n_over)))
            underflow_bounds = list(np.linspace(np.min(z), min_val,
                                                n_over+1))[:-1]
        colors_normal = list(plt.get_cmap('YlOrRd')(
            np.linspace(0, .9, n_normal)))
        colors_overflow = list(
            plt.get_cmap('Greens')(np.linspace(0.5, 1, n_over)))
        all_colors = colors_underflow + colors_normal + colors_overflow
        color_map = mpl.colors.LinearSegmentedColormap.from_list(
            'color_map', all_colors)
        normal_bounds = list(np.linspace(min_val,
                                         top_overflow, n_normal+1))[:-1]
        overflow_bounds = list(np.linspace(top_overflow,
                                           np.max(z), n_over))
        bounds = underflow_bounds + normal_bounds + overflow_bounds
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    elif log_scale:
        norm = colors.LogNorm(vmin=cf_lvls[0], vmax=cf_lvls[-1])
    else:
        norm = None

    if extend == 'neither':
        # plot with appropriate parameters
        # zorder: put the filled-contour below coastlines
        contour_fills = plt.contourf(lons, lats, z, cf_lvls,
                                     transform=cartopy.crs.PlateCarree(),
                                     zorder=0.5,
                                     cmap=color_map,
                                     norm=norm)
    else:
        contour_fills = plt.contourf(lons, lats, z, cf_lvls,
                                     transform=cartopy.crs.PlateCarree(),
                                     zorder=0.5,
                                     cmap=color_map,
                                     norm=norm,
                                     extend=extend)
    contour_lines = plt.contour(lons, lats, z, cl_lvls, colors='0.1',
                                transform=cartopy.crs.PlateCarree(),
                                linewidths=1)

    # Label levels with specially formatted floats
    plt.rcParams['font.weight'] = 'bold'
    plt.clabel(contour_lines, fmt=cline_label_format, inline=1, fontsize=9,
               colors='k')
    plt.rcParams['font.weight'] = 'normal'

    if label_cities:  # TODO remove/ better: test locations
        HH = (53.551086, 9.993682)
        Hannover = (52.373954, 9.741647)
        Bremen = (53.075176, 8.801850)
        city_labels = ['Hamburg', 'Hannover', 'Bremen']
        x_cities, y_cities = plt([HH[1], Hannover[1], Bremen[1]],
                                 [HH[0], Hannover[0], Bremen[0]])
        plt.plot(x_cities, y_cities, 'o', color='darkslategrey',
                 markersize=4)
        for label, xpt, ypt in zip(city_labels, x_cities, y_cities):
            plt.text(xpt+0.5, ypt+0.01, label, color='darkslategrey',
                     fontsize=6)

    return contour_fills


def plot_single_panel(plot_item, plot_title='',
                      overflow=None):
    """"Plot panel with one individual plot.

    Args:
        plot_item (dict): Individual properties of the plots.
        plot_title (string, optional): Title to be written above the plot.
    """
    # Set up figure, calculate figure height corresponding to desired width.
    plot_frame_top, plot_frame_bottom, plot_frame_left, \
        plot_frame_right = .95, 0.15, 0., 1.

    bottom_pos_colorbar = .09
    fig_width = 3
    if plot_title == '':
        plot_frame_top = 1.

    plot_frame_width = plot_frame_right - plot_frame_left
    width_colorbar = plot_frame_width*0.9
    fig_height = calc_fig_height(fig_width, (1, 1), plot_frame_top,
                                 plot_frame_bottom, plot_frame_left,
                                 plot_frame_right)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=150,
                           subplot_kw={'projection': mrc})
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom,
                        left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    ax.coastlines(color='darkslategrey')  # TODO resolution='50m', color='black', linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5, edgecolor='darkslategrey')
    # Plot the data.

    # Mapping individual properties of the plots.
    z = plot_item['data']
    cf_lvls = plot_item['contour_fill_levels']
    cl_lvls = plot_item['contour_line_levels']
    cb_ticks = plot_item['colorbar_ticks']
    cb_tick_fmt = plot_item['colorbar_tick_fmt']
    apply_log_scale = plot_item.get('log_scale', False)
    extend = plot_item.get('extend', "neither")
    cl_label_fmt = plot_item.get('contour_line_label_fmt', None)
    if cl_label_fmt is None:
        cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")

    plt.title(plot_title)
    contour_fills = individual_plot(z, cf_lvls, cl_lvls,
                                    cline_label_format=cl_label_fmt,
                                    log_scale=apply_log_scale,
                                    extend=extend,
                                    overflow=overflow)

    # Add axis for colorbar.
    i = 0
    left_pos_colorbar = plot_frame_width*i + \
        (plot_frame_width-width_colorbar)/2 + plot_frame_left
    cbar_ax = fig.add_axes([left_pos_colorbar, bottom_pos_colorbar,
                            width_colorbar, 0.035])
    if apply_log_scale:
        formatter = LogFormatter(10, labelOnlyBase=False)
    else:
        formatter = None
    cbar = plt.colorbar(contour_fills, orientation="horizontal",
                        cax=cbar_ax, ticks=cb_ticks, format=formatter)
    cbar.ax.set_xticklabels([cb_tick_fmt.format(t) for t in cb_ticks])
    cbar.set_label(plot_item['colorbar_label'])


def plot_panel_1x3(plot_items, column_titles, row_item):
    """"Plot panel with 3 columns of individual plots.

    Args:
        plot_items (list of dict): Individual properties of the plots.
        column_titles (list): Plot titles per column.
        row_item (dict): General properties of the plots.

    """
    # Set up figure, calculate figure height corresponding to desired width.
    bottom_pos_colorbar = .09
    fig_width = 9.
    plot_frame_top, plot_frame_bottom, plot_frame_left, \
        plot_frame_right = .95, 0, .035, 0.88
    fig_height = calc_fig_height(fig_width, (1, 3), plot_frame_top,
                                 plot_frame_bottom, plot_frame_left,
                                 plot_frame_right)

    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=150,
                            subplot_kw={'projection': mrc})
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom,
                        left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    # Mapping general properties of the plots.
    cf_lvls = row_item['contour_fill_levels']
    cb_tick_fmt = row_item.get('colorbar_tick_fmt', "{:.1f}")
    cl_label_fmt = row_item.get('contour_line_label_fmt', None)
    if cl_label_fmt is None:
        cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")

    # Plot the data.
    for ax, title, plot_item in zip(axs, column_titles, plot_items):
        # Mapping individual properties of the plots.
        z = plot_item['data']
        cl_lvls = plot_item['contour_line_levels']

        plt.axes(ax)
        ax.coastlines(color='darkslategrey')  # TODO resolution='50m', color='black', linewidth=1)
        plt.title(title)
        contour_fills = individual_plot(z, cf_lvls, cl_lvls,
                                        cline_label_format=cl_label_fmt)

    # Add axis for colorbar.
    height_colorbar = .85
    bottom_pos_colorbar = (plot_frame_top - height_colorbar)/2
    cbar_ax = fig.add_axes([0.91, bottom_pos_colorbar, 0.02, height_colorbar])
    cbar = fig.colorbar(contour_fills, cax=cbar_ax,
                        ticks=row_item['colorbar_ticks'])
    cbar.ax.set_yticklabels([cb_tick_fmt.format(t)
                             for t in row_item['colorbar_ticks']])
    cbar.set_label(row_item['colorbar_label'])


def plot_panel_1x3_seperate_colorbar(plot_items, column_titles):
    """"Plot panel with 3 columns of individual plots using solely seperate
        plot properties.

    Args:
        plot_items (list of dict): Individual properties of the plots.
        column_titles (list): Plot titles per column.

    """
    # Set up figure, calculate figure height corresponding to desired width.
    plot_frame_top, plot_frame_bottom, plot_frame_left, \
        plot_frame_right = .95, 0.17, 0., 1.
    width_colorbar = .27
    bottom_pos_colorbar = .1
    fig_width = 9.*(0.88-.035)
    if column_titles is None:
        plot_frame_top = 1.
        column_titles = [None]*3
    plot_frame_width = plot_frame_right - plot_frame_left

    fig_height = calc_fig_height(fig_width, (1, 3), plot_frame_top,
                                 plot_frame_bottom, plot_frame_left,
                                 plot_frame_right)

    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=150,
                            subplot_kw={'projection': mrc})
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom,
                        left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    # Plot the data.
    for i, (ax, title, plot_item) in enumerate(zip(axs, column_titles,
                                                   plot_items)):
        # Mapping individual properties of the plots.
        z = plot_item['data']
        cf_lvls = plot_item['contour_fill_levels']
        cl_lvls = plot_item['contour_line_levels']
        cb_ticks = plot_item['colorbar_ticks']
        cb_tick_fmt = plot_item['colorbar_tick_fmt']
        apply_log_scale = plot_item.get('log_scale', False)
        extend = plot_item.get('extend', "neither")
        cl_label_fmt = plot_item.get('contour_line_label_fmt', None)
        if cl_label_fmt is None:
            cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")

        plt.axes(ax)
        ax.coastlines(color='darkslategrey')  # TODO resolution='50m', color='black', linewidth=1)
        plt.title(title)
        contour_fills = individual_plot(z, cf_lvls, cl_lvls,
                                        cline_label_format=cl_label_fmt,
                                        log_scale=apply_log_scale,
                                        extend=extend)

        # Add axis for colorbar.
        left_pos_colorbar = plot_frame_width/3*i + \
            (plot_frame_width/3-width_colorbar)/2 + plot_frame_left
        cbar_ax = fig.add_axes([left_pos_colorbar, bottom_pos_colorbar,
                                width_colorbar, 0.035])
        if apply_log_scale:
            formatter = LogFormatter(10, labelOnlyBase=False)
        else:
            formatter = None
        cbar = plt.colorbar(contour_fills, orientation="horizontal",
                            cax=cbar_ax, ticks=cb_ticks, format=formatter)
        cbar.ax.set_xticklabels([cb_tick_fmt.format(t) for t in cb_ticks])
        cbar.set_label(plot_item['colorbar_label'])


def plot_panel_2x3(plot_items, column_titles, row_items):
    """"Plot panel with 2 rows and 3 columns of individual plots.

    Args:
        plot_items (list of dict): Individual properties of the plots.
        column_titles (list): Plot titles per column.
        row_items (list of dict): Properties of the plots shared per row.

    """
    # Set up figure, calculate determine figure height corresponding to
    # desired width.
    plot_frame_top, plot_frame_bottom, plot_frame_left, \
        plot_frame_right = .96, 0.0, .035, 0.88
    fig_width = 9.
    fig_height = calc_fig_height(fig_width, (2, 3), plot_frame_top,
                                 plot_frame_bottom, plot_frame_left,
                                 plot_frame_right)

    fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height), dpi=150,
                            subplot_kw={'projection': mrc})
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom,
                        left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    # Positioning of colorbars.
    height_colorbar = .4
    right_pos_colorbar = .9

    for i_row, row_item in enumerate(row_items):
        # Mapping properties of the plots shared per row.
        cb_tick_fmt = row_item.get('colorbar_tick_fmt', "{:.1f}")
        extend = row_item.get('extend', "neither")
        cl_label_fmt = row_item.get('contour_line_label_fmt', None)
        if cl_label_fmt is None:
            cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")
        cf_lvls = row_items[i_row]['contour_fill_levels']

        # First row of plots.
        for ax, plot_item in zip(axs[i_row, :], plot_items[i_row]):
            # Mapping individual properties of the plots.
            z = plot_item['data']
            cl_lvls = plot_item['contour_line_levels']

            plt.axes(ax)
            ax.coastlines(color='darkslategrey')  # TODO resolution='50m', color='black', linewidth=1)
            contour_fills = individual_plot(z, cf_lvls, cl_lvls,
                                            cline_label_format=cl_label_fmt,
                                            extend=extend)

        # Add axis for colorbar.
        bottom_pos_colorbar = (1-i_row)*plot_frame_top/2 + \
            (plot_frame_top/2-height_colorbar)/2
        cbar_ax = fig.add_axes([right_pos_colorbar, bottom_pos_colorbar,
                                0.02, height_colorbar])
        cbar = fig.colorbar(contour_fills, cax=cbar_ax,
                            ticks=row_item['colorbar_ticks'])
        cbar.ax.set_yticklabels([cb_tick_fmt.format(t)
                                 for t in row_item['colorbar_ticks']])
        cbar.set_label(row_item['colorbar_label'])

    # Add subplot row and column labels.
    row_titles = [r['title'] for r in row_items]
    for ax, col in zip(axs[0], column_titles):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 5.),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:, 0], row_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad + 2., 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)


def percentile_plots(plot_var, i_case, plot_settings):
    """" Reading processed data and plotting the 5th, 32nd, 50th percentile
        maps. Used for figure 3.

    Args:
        plot_var (str): Name of plotting variable in netCDF source file.
        i_case (int): Id of plotted case.
        plot_settings (dict): Individual and shared properties of the plots.

    """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    plot_var_suffix = ["_perc5", "_perc32", "_perc50"]

    # Read data from NetCDF source file.
    plot_items = []
    plot_data_max = 0
    for s in plot_var_suffix:
        d = nc[plot_var+s].values[i_case, :, :]
        if plot_var[0] == "p":
            d *= 1e-3
        plot_items.append({'data': d})
        if np.amax(d) > plot_data_max:
            plot_data_max = np.amax(d)

    # Mapping plot properties and splitting up into individual and
    # shared properties.
    plot_handling = plot_settings["plot_handling"]
    contour_fill_levels = plot_handling["contour_fill_levels"]
    contour_line_levels = plot_handling.get("contour_line_levels", 3 *
                                            [contour_fill_levels])
    colorbar_ticks = plot_handling.get("colorbar_ticks", contour_fill_levels)
    colorbar_label = plot_settings["color_label"]

    # Write the contour handling to plot_items.
    for i, plot_item in enumerate(plot_items):
        plot_item['contour_line_levels'] = contour_line_levels[i]

    # Write the row dependent settings to row_items.
    row_item = {
        'colorbar_ticks': colorbar_ticks,
        'colorbar_label': colorbar_label,
        'contour_fill_levels': contour_fill_levels,
    }
    if 'colorbar_tick_fmt' in plot_handling:
        row_item['colorbar_tick_fmt'] = plot_handling["colorbar_tick_fmt"]
    if 'contour_line_label_fmt' in plot_handling:
        row_item['contour_line_label_fmt'] = \
            plot_handling["contour_line_label_fmt"]

    plot_panel_1x3(plot_items, column_titles, row_item)


def percentile_plots_ref(plot_var, i_case, plot_var_ref, i_case_ref,
                         plot_settings_abs, plot_settings_rel):
    """" Reading processed data and plotting the 5th, 32nd, 50th percentile
        maps on the first row and the relative
        increase w.r.t the reference case on the second row. Used for figure 7.

    Args:
        plot_var (str): Name of plotting variable in netCDF source file.
        i_case (int): Id of plotted case.
        plot_var_ref (str): Name of reference variable in netCDF source file.
        i_case_ref (int): Id of reference case
        plot_settings_abs (dict): Individual and shared properties of the top
            row plots.
        plot_settings_rel (dict): Individual and shared properties of the
            bottom row plots.

    """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    row_titles = ['Absolute value', 'Relative to reference case']
    plot_var_suffix = ["_perc5", "_perc32", "_perc50"]

    # Read data from NetCDF source file.
    plot_items = [[], []]
    plot_data_max, plot_data_relative_max = 0, 0
    for s in plot_var_suffix:
        d = nc[plot_var+s].values[i_case, :, :]
        if plot_var[0] == "p":
            d *= 1e-3
        plot_items[0].append({'data': d})
        if np.amax(d) > plot_data_max:
            plot_data_max = np.amax(d)

        d_ref = nc[plot_var_ref+s].values[i_case_ref, :, :]
        if plot_var[0] == "p":
            d_ref *= 1e-3
        d_relative = d/d_ref
        plot_items[1].append({'data': d_relative})
        if np.amax(d_relative) > plot_data_relative_max:
            plot_data_relative_max = np.amax(d_relative)

    print("Max absolute and relative value are respectively {:.2f} and {:.2f}"
          .format(plot_data_max, plot_data_relative_max))

    # Mapping plot properties and splitting up into individual and shared properties.
    plot_handling = plot_settings_abs["plot_handling"]
    contour_fill_levels = plot_handling["contour_fill_levels"]
    contour_line_levels = plot_handling.get("contour_line_levels", 3*[contour_fill_levels])
    colorbar_ticks = plot_handling.get("colorbar_ticks", contour_fill_levels)

    contour_fill_levels_rel = plot_settings_rel["contour_fill_levels"]
    contour_line_levels_rel = plot_settings_rel.get("contour_line_levels", 3*[contour_fill_levels_rel])
    colorbar_ticks_rel = plot_settings_rel.get("colorbar_ticks", contour_fill_levels_rel)

    # Write the contour handling to plot_items.
    for i, plot_item in enumerate(plot_items[0]):
        plot_item['contour_line_levels'] = contour_line_levels[i]
    for i, plot_item in enumerate(plot_items[1]):
        plot_item['contour_line_levels'] = contour_line_levels_rel[i]

    # Write the row dependent settings to row_items.
    row_items = []
    for i in range(2):
        row_items.append({
            'title': row_titles[i],
        })
    row_items[0]['colorbar_ticks'] = colorbar_ticks
    row_items[0]['colorbar_label'] = plot_settings_abs["color_label"]
    row_items[0]['contour_fill_levels'] = contour_fill_levels
    if 'colorbar_tick_fmt' in plot_handling:
        row_items[0]['colorbar_tick_fmt'] = plot_handling["colorbar_tick_fmt"]
    row_items[0]['contour_line_label_fmt'] = '%.1f'

    row_items[1]['colorbar_ticks'] = colorbar_ticks_rel
    row_items[1]['colorbar_label'] = "Increase factor [-]"
    row_items[1]['contour_fill_levels'] = contour_fill_levels_rel
    if 'colorbar_tick_fmt' in plot_settings_rel:
        row_items[1]['colorbar_tick_fmt'] = plot_settings_rel["colorbar_tick_fmt"]
    row_items[1]['extend'] = plot_settings_rel.get('extend', "neither")

    plot_panel_2x3(plot_items, column_titles, row_items)


def plot_figure5():
    """" Generate integrated mean power plot. """
    column_titles = ["50 - 150m", "10 - 500m", "Ratio"]

    linspace0 = np.linspace(0, .31, 21)
    plot_item0 = {
        'data': p_integral_mean[0, :, :]*1e-6,
        'contour_line_levels': linspace0[::4],
        'contour_fill_levels': linspace0,
        'colorbar_ticks': linspace0[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': '[$MWm/m^2$]',
    }
    linspace1 = np.linspace(0, 1.5, 21)
    plot_item1 = {
        'data': p_integral_mean[1, :, :]*1e-6,
        'contour_line_levels': linspace1[::4],
        'contour_fill_levels': linspace1,
        'colorbar_ticks': linspace1[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': '[$MWm/m^2$]',
    }
    logspace2 = np.logspace(np.log10(4), np.log10(28.0), num=17)
    plot_item2 = {
        'data': plot_item1['data']/plot_item0['data'],
        'contour_line_levels': [10, 15],
        'contour_fill_levels': logspace2,
        'colorbar_ticks': logspace2[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Increase factor [-]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure3():
    """" Generate fixed height wind speed plot. """
    plot_settings = {
        "color_label": 'Wind speed [m/s]',
        "plot_handling": {
            "contour_fill_levels": np.arange(0, 15.1, 0.5),  # 13.1, 1),
            "contour_line_levels": [
                [1., 2., 3., 4.],
                [3., 5., 7., 9.],
                [5., 7., 9., 11.],
            ],
            "colorbar_ticks": np.arange(0, 15, 2),  # 13, 2),
            "colorbar_tick_fmt": "{:.0f}",
            'contour_line_label_fmt': '%.1f',
        },
    }

    percentile_plots("v_fixed", 0, plot_settings)


def plot_figure4():
    """" Generate fixed height power density plot. """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]

    fixed_height_ref = 100.
    fixed_height_id = list(fixed_heights).index(fixed_height_ref)

    linspace0 = np.linspace(0, 0.027, 21)  # np.linspace(0, .033, 21)
    plot_item0 = {
        'data': nc["p_fixed_perc5"].values[fixed_height_id, :, :]*1e-3,
        'contour_fill_levels': linspace0,
        'contour_line_levels': sorted([.003]+list(linspace0[::5])),
        'contour_line_label_fmt': '%.3f',
        'colorbar_ticks': linspace0[::5],
        'colorbar_tick_fmt': '{:.3f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    linspace1 = np.linspace(0, 0.45, 21)  # np.linspace(0, .45, 21)
    plot_item1 = {
        'data': nc["p_fixed_perc32"].values[fixed_height_id, :, :]*1e-3,
        'contour_fill_levels': linspace1,
        'contour_line_levels': sorted([.04]+list(linspace1[::4])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace1[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    linspace2 = np.linspace(0, 0.95, 21)  # np.linspace(0, 1, 21)
    plot_item2 = {
        'data': nc["p_fixed_perc50"].values[fixed_height_id, :, :]*1e-3,
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


def plot_figure8():
    """" Generate baseline comparison wind speed plot. """
    linspace_absolute = np.linspace(0, 15, 21)  # np.arange(0, 15.1, 1)
    plot_settings_absolute_row = {
        "color_label": 'Wind speed [m/s]',
        "plot_handling": {
            "contour_fill_levels": linspace_absolute,
            "colorbar_ticks": linspace_absolute[::2],
            "contour_line_levels": [
                linspace_absolute,
                [5., 7., 9., 10.],
                [7., 9., 11., 13.],
            ],
            "colorbar_tick_fmt": "{:.0f}",
        },
    }
    linspace_relative = np.linspace(0, 2, 21)  # np.linspace(1., 2.2, 21)
    plot_settings_relative_row = {
        "contour_fill_levels": linspace_relative,
        "colorbar_ticks": linspace_relative[::4],
        "contour_line_levels": [
            [1.1, 1.4, 1.7],
            [1.1, 1.4, 1.7],
            [1.1, 1.4, 1.7],
        ],
        'extend': 'max',
    }
    percentile_plots_ref("v_ceiling", height_range_ceilings.index(500),
                         "v_fixed", fixed_heights.index(100),
                         plot_settings_absolute_row, plot_settings_relative_row)


def plot_figure9_upper():
    """" Generate baseline comparison wind power plot - upper part. """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]

    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    linspace0 = np.linspace(0, .04, 21)
    plot_item0 = {
        'data': nc["p_ceiling_perc5"].values[height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': linspace0,
        'contour_line_levels': linspace0[::5],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace0[::5],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    linspace1 = np.linspace(0, .6, 21)
    plot_item1 = {
        'data': nc["p_ceiling_perc32"].values[height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': linspace1,
        'contour_line_levels': linspace1[::4],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace1[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    linspace2 = np.linspace(0, 1.3, 21)
    plot_item2 = {
        'data': nc["p_ceiling_perc50"].values[height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': linspace2,
        'contour_line_levels': linspace2[::4],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': linspace2[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure9_lower():
    """" Generate baseline comparison wind power plot - lower part. """
    column_titles = None

    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    fixed_height_ref = 100.
    fixed_height_id = list(fixed_heights).index(fixed_height_ref)

    linspace0 = np.linspace(0, 20, 21)
    plot_item0 = {
        'data': nc["p_ceiling_perc5"].values[height_ceiling_id, :, :]
                / nc["p_fixed_perc5"].values[fixed_height_id, :, :],
        'contour_fill_levels': linspace0,
        'contour_line_levels': np.arange(2., 5., 1.),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace0[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
        'extend': 'max',
    }
    linspace1 = np.linspace(0, 10, 21)
    plot_item1 = {
        'data': nc["p_ceiling_perc32"].values[height_ceiling_id, :, :]
                / nc["p_fixed_perc32"].values[fixed_height_id, :, :],
        'contour_fill_levels': linspace1,
        'contour_line_levels': linspace1[::4],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace1[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
        'extend': 'max',
    }
    linspace2 = np.linspace(0, 10, 21)
    plot_item2 = {
        'data': nc["p_ceiling_perc50"].values[height_ceiling_id, :, :]
                / nc["p_fixed_perc50"].values[fixed_height_id, :, :],
        'contour_fill_levels': linspace2,
        'contour_line_levels': linspace2[::4],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace2[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
        'extend': 'max',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure10():
    """" Generate power availability plot. """
    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    linspace00 = np.linspace(0, 100, 21)
    plot_item00 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_id, :, :],
        'contour_fill_levels': linspace00,
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace00[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    linspace01 = np.linspace(0, 100, 21)
    plot_item01 = {
        'data': 100.-nc["p_ceiling_rank300"].values[height_ceiling_id, :, :],
        'contour_fill_levels': linspace01,
        'contour_line_levels': linspace01[::4][2:],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace01[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
    }
    linspace02 = np.linspace(0, 70, 21)
    plot_item02 = {
        'data': 100.-nc["p_ceiling_rank1600"].values[height_ceiling_id, :, :],
        'contour_fill_levels': linspace02,
        'contour_line_levels': linspace02[::4][2:],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace02[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
    }

    column_titles = ["40 $W/m^2$", "300 $W/m^2$", "1600 $W/m^2$"]
    plot_items = [plot_item00, plot_item01, plot_item02]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)

    linspace10 = np.linspace(0., 50., 21)
    plot_item10 = {
        'data': (100.-nc["p_ceiling_rank40"].values[height_ceiling_id, :, :]) -
                (100.-nc["p_fixed_rank40"].values[0, :, :]),
        'contour_fill_levels': linspace10,
        'contour_line_levels': sorted([1.1, 2.2]+list(linspace10[::4][:-2])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace10[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    linspace11 = np.linspace(0., 55., 21)
    plot_item11 = {
        'data': (100.-nc["p_ceiling_rank300"].values[height_ceiling_id, :, :]) -
                (100.-nc["p_fixed_rank300"].values[0, :, :]),
        'contour_fill_levels': linspace11,
        'contour_line_levels': linspace11[::4][:-2],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace11[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    linspace12 = np.linspace(0., 45., 21)
    plot_item12 = {
        'data': (100.-nc["p_ceiling_rank1600"].values[height_ceiling_id, :, :]) -
                (100.-nc["p_fixed_rank1600"].values[0, :, :]),
        'contour_fill_levels': linspace12,
        'contour_line_levels': linspace12[::4][:-2],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace12[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }

    column_titles = None
    plot_items = [plot_item10, plot_item11, plot_item12]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure11():
    """" Generate 40 W/m^2 power availability plot for alternative height ceilings. """
    height_ceilings = [300., 1000., 1250.]
    height_ceiling_ids = [list(height_range_ceilings).index(height_ceiling) for height_ceiling in height_ceilings]
    baseline_height_ceiling = 500.
    baseline_height_ceiling_id = list(height_range_ceilings).index(baseline_height_ceiling)

    linspace00 = np.linspace(0, 100, 21)
    plot_item00 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[0], :, :],
        'contour_fill_levels': linspace00,
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace00[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    linspace01 = np.linspace(10, 100, 21)
    plot_item01 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[1], :, :],
        'contour_fill_levels': linspace01,
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace01[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    linspace02 = np.linspace(10, 100, 21)
    plot_item02 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[2], :, :],
        'contour_fill_levels': linspace02,
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace02[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }

    column_titles = ["300 m", "1000 m", "1250 m"]
    plot_items = [plot_item00, plot_item01, plot_item02]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)

    linspace10 = np.linspace(0., 22., 21)
    plot_item10 = {
        'data': -(100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[0], :, :]) +
                (100.-nc["p_ceiling_rank40"].values[baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace10,
        'contour_line_levels': sorted([1.1]+list(linspace10[::4])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace10[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability decrease [%]',
    }
    linspace11 = np.linspace(0., 38., 21)
    plot_item11 = {
        'data': (100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[1], :, :]) -
                (100.-nc["p_ceiling_rank40"].values[baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace11,
        'contour_line_levels': sorted([2.3]+list(linspace11[::4])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace11[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    linspace12 = np.linspace(0., 50., 21)
    plot_item12 = {
        'data': (100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[2], :, :]) -
                (100.-nc["p_ceiling_rank40"].values[baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace12,
        'contour_line_levels': sorted([3.8]+list(linspace12[::4])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace12[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }

    column_titles = None
    plot_items = [plot_item10, plot_item11, plot_item12]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_mean_and_ratio(data_type='v',
                        fill_range=[0, 20],
                        ratio_range=[0, 2],
                        line_levels=[2, 5, 15, 20],
                        n_decimals=0):
    if data_type == 'v':
        label = r'v [m/s]'
        scale = 1
        ratio_levels = [1.1, 1.3, 1.6]
    elif data_type == 'p':
        label = r'Power density [$kW/m^2$]'
        scale = 10**(-3)
        ratio_levels = [1, 3, 4.5]

    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    fixed_height_ref = 100.
    fixed_height_id = list(fixed_heights).index(fixed_height_ref)

    # TODO automatize with data?
    plot_title = '500m ceiling'
    linspace00 = np.linspace(fill_range[0], fill_range[1], 21)
    plot_item = {
        'data': nc['{}_ceiling_mean'.format(data_type)].values[height_ceiling_id, :, :]*scale,
        'contour_fill_levels': linspace00,
        'contour_line_levels': line_levels,
        'contour_line_label_fmt': '%.{}f'.format(n_decimals),
        'colorbar_ticks': linspace00[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': label,
        'extend': 'max',
    }
    eval_contour_fill_levels([plot_item])
    plot_single_panel(plot_item, plot_title=plot_title)

    plot_title = '100m fixed'
    linspace00 = np.linspace(fill_range[0], fill_range[1], 21)
    plot_item = {
        'data': nc['{}_fixed_mean'.format(data_type)].values[fixed_height_id, :, :]*scale,
        'contour_fill_levels': linspace00,
        'contour_line_levels': line_levels,
        'contour_line_label_fmt': '%.{}f'.format(n_decimals),
        'colorbar_ticks': linspace00[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': label,
        'extend': 'max',
    }
    eval_contour_fill_levels([plot_item])
    plot_single_panel(plot_item, plot_title=plot_title)

    plot_title = 'Ratio using 100m'
    linspace00 = np.linspace(ratio_range[0], ratio_range[1], 25)
    plot_item = {
        'data': nc['{}_ceiling_mean'.format(data_type)].values[height_ceiling_id, :, :]/nc[
            '{}_fixed_mean'.format(data_type)].values[fixed_height_id, :, :],
        'contour_fill_levels': linspace00,
        'contour_line_levels': ratio_levels,
        'contour_line_label_fmt': '%.{}f'.format(1),
        'colorbar_ticks': linspace00[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': '{}/{}_ref [-]'.format(data_type, data_type),
        'extend': 'max',
    }
    eval_contour_fill_levels([plot_item])
    plot_single_panel(plot_item, plot_title=plot_title)

def plot_surface_elevation_from_geopotential():
    from process_data_paper import get_surface_elevation
    data = get_surface_elevation(lats, lons, remove_neg=False,
                                 revert_lat=True)
    data[np.logical_and(data < 20, data > 0)] = 0

    plot_title = 'Topography'
    # color_map = plt.get_cmap('terrain')
    # Set range such that 0 is at blue part
    min_range_data, max_range = np.min(data), np.max(data)
    blue = 56/256.
    min_range = blue/(1 - blue) * max_range
    if -min_range > min_range_data:
        print('Min range does not cover full min range.')
    linspace00 = np.linspace(-min_range, max_range, 42)
    plot_item = {
        'data': data,
        'contour_fill_levels': linspace00,
        'contour_line_levels': [-300, 300, 700, 1500],
        'contour_line_label_fmt': '%.{}f'.format(0),
        'colorbar_ticks': linspace00[::8],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'surface elevation [m]',
    }
    eval_contour_fill_levels([plot_item])
    plot_single_panel(plot_item, plot_title=plot_title)

def plot_all():
    plot_mean_and_ratio(data_type='v',
                        fill_range=[0, 15], # [6, 13],
                        line_levels=[7, 9, 11],
                        ratio_range=[0, 2.5],
                        n_decimals=0)
    # plot_mean_and_ratio(data_type='p',
    #                     fill_range=[0, 2.7],
    #                     line_levels=[0.3, 1.1, 1.5, 2],
    #                     ratio_range=[1, 17],
    #                     n_decimals=1)
    # plot_surface_elevation_from_geopotential()
    # plot_figure3()
    # plot_figure4()
    # plot_figure5()
    # plot_figure8()
    # plot_figure9_upper()
    # plot_figure9_lower()
    # plot_figure10()
    # plot_figure11()
    plt.show()

if __name__ == "__main__":
    plot_all()