#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO function descriptions docstrings
import matplotlib.pyplot as plt
import numpy as np
from config_Lavi import start_year, final_year, output_dir
from plotting_utils import read_dataset_user_input
from process_data_paper import get_altitude_from_level
plt.style.use('seaborn-whitegrid')


def plot_histogram(level, data_entries, bin_range=[], hist_type='ml',
                   x_label=r'Calculated Height in m',
                   mean_label=r'Average Calculated Height',
                   ml_label=r'Model Level Height',
                   title='', info='', ml=True):
    """
    Histogram specific plotting.

    Parameters
    ----------
    level : TYPE
        DESCRIPTION.
    data_entries : TYPE
        DESCRIPTION.
    bin_range : TYPE
        DESCRIPTION.
    hist_type : TYPE, optional
        DESCRIPTION. The default is 'ml'.
    x_label : TYPE, optional
        DESCRIPTION. The default is r'Calculated Height in m'.
    mean_label : TYPE, optional
        DESCRIPTION. The default is r'Average Calculated Height'.
    ml_label : TYPE, optional
        DESCRIPTION. The default is r'Model Level Height'.
    title : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.

    """
    if len(bin_range) != 0:  # Preset bin range filled according to data:
        normed_entries = data_entries/sum(data_entries)
        mean = sum(bin_range*normed_entries)
        # rms = np.sqrt(normed_entries*np.square(bin_range))  # TODO check rms calc
        plt.step(bin_range, normed_entries, color='skyblue')
    else:
        hist_data = data_entries
        mean = np.mean(hist_data)
        rms = np.sqrt(np.mean(np.square(mean-hist_data)))
        plt.hist(hist_data, density=True, bins=100, color='skyblue')
        plt.axvline(x=mean, color='steelblue', linewidth=rms*2, alpha=0.2)
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')


    if ml:
        plt.axvline(x=get_altitude_from_level(level), color='red', label=ml_label)
    plt.axvline(x=mean, color='steelblue', label=mean_label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(r"Probability")
    plt.grid(False)
    plt.legend()
    plt.savefig(output_dir + 'hists_model_level_height/hist_'+hist_type+'_{:.0f}_years_{}_{}'.format(
            level, start_year, final_year) + info + '.pdf')
    plt.close()


def plot_all_histograms(hists_dict):
    # Fill Histograms with height information for each model level
    print("Plotting histograms for each level:")
    for level_idx, level in enumerate(levels):
        print(level)
        data_entries = hists_dict['height'][level]['data']
        bin_range = hists_dict['height'][level]['bins']
        plot_histogram(level, data_entries, bin_range=bin_range,
                       title=r'Model Level {:.0f} ({:.2f}m)'.format(level, get_altitude_from_level(level)))

        # Difference to constant height - surface elevation
        data_entries = hists_dict['diff'][level]['data']
        bin_range = hists_dict['diff'][level]['bins']
        plot_histogram(level, data_entries, bin_range=bin_range, hist_type='difference_ml',
                       mean_label=r'Average Height Difference',
                       x_label=r'Difference Calculated Height and (ML height minus surface elevation) in m',
                       title=r'Model Level {:.0f} ({:.2f}m)'.format(level, get_altitude_from_level(level)),
                       ml=False)

    # Velocity distributions for 500m ceiling height
    data_entries = hists_dict['500m_ceil']['data']
    bin_range = hists_dict['500m_ceil']['bins']
    plot_histogram(level, data_entries, bin_range=bin_range, hist_type='velocity_500m_ceil',
                   mean_label=r'Average wind speed',
                   x_label=r'wind speed in m/s',
                   title=r'Maximal wind speeds to be reached in a 500m ceiling height setting',
                   ml=False)

    # Velocity distributions for 100m fixed height
    data_entries = hists_dict['100m_fixed']['data']
    bin_range = hists_dict['100m_fixed']['bins']
    plot_histogram(level, data_entries, bin_range=bin_range, hist_type='velocity_100m_fixed',
                   mean_label=r'Average wind speed',
                   x_label=r'wind speed in m/s',
                   title=r'Maximal wind speeds to be reached in a 100m fixed height setting',
                   ml=False)


def def_hist_data_dict(levels):
    hists_dict = {}
    keys = ['height', 'diff', '100m_fixed', '500m_ceil']
    for key in keys:
        hists_dict[key] = {}
        if key in ['height', 'diff']:
            for level in levels:
                hists_dict[key][level] = {}
    return hists_dict


if __name__ == '__main__':
    file_name = output_dir +\
        "hists_model_level_height/plot_data/hist_data" +\
        "_{start_year:d}_{final_year:d}_subset_{lat_subset_id:04d}_of_{max_lat_subset_id:04d}.nc"
    ds = read_dataset_user_input(output_file_subset=file_name, concat_via='lat_subset')
    levels = ds['level'].values
    subset_ids = ds['lat_subset'].values
    hists_dict = def_hist_data_dict(levels)
    level_heights = ds['level_heights'].values
    level_height_diffs = ds['level_height_diffs'].values
    bins_level_heights = ds['bins_level_heights'].values
    bins_level_height_diffs = ds['bins_level_height_diffs'].values
    v_100m_fixed = ds['v_100m_fixed'].values
    v_500m_ceil = ds['v_500m_ceil'].values
    print("Combine subset input")
    for subset_id in subset_ids:
        if subset_id % 20 == 0:
            print("    {:04d} subsets evaluated".format(subset_id))
        for level_idx, level in enumerate(levels):
            if subset_id == 0:
                hists_dict['height'][level]['data'] = level_heights[subset_id, level_idx, :]
                hists_dict['height'][level]['bins'] = bins_level_heights[subset_id, level_idx, :]
                hists_dict['diff'][level]['data'] = level_height_diffs[subset_id, level_idx, :]
                hists_dict['diff'][level]['bins'] = bins_level_height_diffs[subset_id, level_idx, :]
            else:
                hists_dict['height'][level]['data'] += level_heights[subset_id, level_idx, :]
                hists_dict['diff'][level]['data'] += level_height_diffs[subset_id, level_idx, :]
        if subset_id == 0:
            hists_dict['100m_fixed']['data'] = v_100m_fixed[subset_id, :]
            hists_dict['100m_fixed']['bins'] = ds['v_bin'].values[:]
            hists_dict['500m_ceil']['data'] = v_500m_ceil[subset_id, :]
            hists_dict['500m_ceil']['bins'] = ds['v_bin'].values[:]
        else:
            hists_dict['100m_fixed']['data'] += v_100m_fixed[subset_id, :]
            hists_dict['500m_ceil']['data'] += v_500m_ceil[subset_id, :]
    plot_all_histograms(hists_dict)
