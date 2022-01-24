import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs

from ..resource_analysis.plot_maps import eval_contour_fill_levels, \
    plot_single_panel

def plot_discrete_map(config, values, title='', label=''):
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
                    plot_item=None):
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
        }
    eval_contour_fill_levels([plot_item])
    plot_single_panel(plot_item, plot_title=plot_title)


def plot_map(config, data, title='', label=''):
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
        plot_single_map(data_loc, title=title, label=label)
        # TODO add other options
    else:
        plot_discrete_map(config,
                          data_loc,
                          title=title,
                          label=label)
    return data_loc