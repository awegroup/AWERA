import numpy as np
import pickle
import pandas as pd

from config_clustering import data_info, result_dir, \
    cut_wind_speeds_file, n_clusters
from config_production import get_loc_brute_force_name_and_locs, \
    sample_selection, brute_force_files, step

from utils_validation import plot_diff_pdf, get_velocity_bin_mask, \
    plot_diff_pdf_mult_data, plot_abs_rel_step_wise
from plot_diff_map import plot_diff_map

# !!! from config_production import plots_interactive
plots_interactive = True
if plots_interactive:
    import matplotlib.pyplot as plt
else:
    import matplotlib as mpl
    mpl.use('Pdf')
    import matplotlib.pyplot as plt

# Utility

def get_file_name(out_info, extension='pdf'):
    return result_dir + out_info \
        + brute_force_files.format(sample_selection, data_info, extension)


def get_central_data(data, lower=0.02, upper=99.98, return_mask=False):
    mask_outer = np.logical_or(data < np.percentile(data, lower),
                               data > np.percentile(data, upper))
    if return_mask:
        return mask_outer
    else:
        central_data = np.ma.array(data, mask=mask_outer)
        return central_data


# Processing steps

def read_single_sample_power_output(n_locs=1000,
                                    skip_locs=[469, 805, 863, 975]):
    skip_locs = [i_loc for i_loc in skip_locs if i_loc < n_locs]
    failed = []
    errs = []
    # TODO combine separately, save again
    for location_number in range(n_locs):
        if location_number == 0:
            n_locs = n_locs - len(skip_locs)
        if location_number in skip_locs:
            continue  # Skip failed simulation

        data_info_loc, locs = get_loc_brute_force_name_and_locs(
            location_number*step, data_info, mult_loc=False)
        # skipped location numbers not counted in arrays to be filled
        i_loc = location_number - sum(location_number > np.array(skip_locs))\
            - len(failed)

        brute_force_testing_file_name = result_dir + brute_force_files.format(
            sample_selection, data_info_loc, 'pickle')
        try:
            with open(brute_force_testing_file_name, 'rb') as f:
                res_n = pickle.load(f)
            if i_loc == 0:
                sample_ids = res_n['sample_ids']
                res = {
                    # Brute force optimization on sample w/o clustering input
                    'power_sample': np.zeros([n_locs, len(sample_ids)]),
                    'x_opt_sample': np.zeros([n_locs, len(sample_ids), 5]),
                    # Power and  controls output from clustering
                    'power_cluster': np.zeros([n_locs, len(sample_ids)]),
                    'x_opt_cluster': np.zeros([n_locs, len(sample_ids), 5]),
                    # QSM simulation using clustering controls (cc) on sample
                    'power_sample_cc': np.zeros([n_locs, len(sample_ids)]),
                    # Optimization starting at the resulting x_opt from
                    # clustering and closely around
                    'power_cc_opt': np.zeros([n_locs, len(sample_ids)]),
                    'x_opt_cc_opt': np.zeros([n_locs, len(sample_ids), 5]),
                    # Locations
                    'locs': [],
                    'n_locs': n_locs,
                    # Clustering info
                    'cluster_id': np.zeros([n_locs, len(sample_ids)]),
                    # Velocity at 100m - used for backscaling
                    # the normalised wind profile from clustering
                    'backscaling': np.zeros([n_locs, len(sample_ids)])
                    }
                # Sample ids ( -> Times) analyzed
                res['sample_ids'] = sample_ids
            # Read 0th value from file - single location files
            res['power_sample'][i_loc, :] = res_n['power'][0, :]
            res['x_opt_sample'][i_loc, :, :] = res_n['sample_x_opt'][0]

            res['power_cluster'][i_loc, :] = res_n['cluster_power'][0, :]
            res['x_opt_cluster'][i_loc, :, :] = res_n['cluster_x_opt'][0]

            res['power_sample_cc'][i_loc, :] = res_n['sample_cluster_power'][0,
                                                                             :]

            res['power_cc_opt'][i_loc, :] = res_n['power_cluster_opt'][0, :]
            res['x_opt_cc_opt'][i_loc, :, :] = res_n['x_opt_cluster_opt'][0]

            res['locs'].append(res_n['locs'][0])

            res['cluster_id'][i_loc, :] = res_n['cluster_id'][0, :]
            res['backscaling'][i_loc, :] = res_n['backscaling'][0, :]
        except (KeyError, FileNotFoundError) as e:
            print('Failed: ', location_number)
            errs.append(e)
            failed.append(location_number)
    if len(failed) > 0:
        print('Failing to read input for {} locations with ids: '.format(
            len(failed)), failed)
        raise errs[-1]

    # read matching cut_in / cut_out wind speeds
    cluster_ids = res['cluster_id']
    res['matching_vw_cut_in'] = np.zeros(cluster_ids.shape)
    res['matching_vw_cut_out'] = np.zeros(cluster_ids.shape)

    limit_estimates = pd.read_csv(cut_wind_speeds_file)
    for cluster_id in range(1, n_clusters+1):
        res['matching_vw_cut_in'][cluster_ids == cluster_id] = \
            limit_estimates.iloc[cluster_id-1]['vw_100m_cut_in']
        res['matching_vw_cut_out'][cluster_ids == cluster_id] = \
            limit_estimates.iloc[cluster_id-1]['vw_100m_cut_out']

    return res


def eval_percentages_succ_vel(name, total, data, select_v):
    in_v_sel = sum((select_v).flat)
    succ = sum((data > -1).flat)
    succ_in_v_sel = sum(np.logical_and(select_v, (data > -1)).flat)
    print('Evaluate number of successful samples and cut-in/out range:')
    text_abs = '{}   total: {} | successful: {} | within cut-in/out: '\
        '{} | successf. and within cut-in/out: {}'
    text_rel = '{}   total: {:.2f} % | successful: {:.2f} % | within '\
        'cut-in/out: {:.2f} % | successf. and within cut-in/out: '\
        '{:.2f} % '
    print(text_abs.format(
          name, total, succ, in_v_sel, succ_in_v_sel))
    print(text_rel.format(
              name, 100, succ/total*100, in_v_sel/total*100,
              succ_in_v_sel/total*100))
    print('--------')


def get_eval_diffs_and_data(test_diffs, res, config, title_names):
    # Calculate differences, select only positive power and
    # successful simulations/optimizations (!=-1)
    # Cluster: select only within clustering velocitiy range (cut-in/out)
    # config: outliers_only, eval_specific_outliers, main_diff
    power_diffs, power_rel_diffs, x_opt_diffs, titles = {}, {}, {}, {}
    for diff in test_diffs:
        diff_type = '{}_vs_{}'.format(diff[0], diff[1])
        titles[diff_type] = 'Diff {} vs {}'.format(
            title_names[diff[0]], title_names[diff[1]])
        # Power differences
        power_diffs[diff_type] = np.ma.array((res['power_'+diff[0]]
                                              - res['power_'+diff[1]]))
        power_diffs[diff_type].mask = np.logical_or(
            res['power_'+diff[0]] <= -1, res['power_'+diff[1]] <= -1)
        if 'cluster' in diff:
            power_diffs[diff_type].mask = np.logical_or(
               power_diffs[diff_type].mask, ~select_v)

        # Study diff outliers
        if config['outliers_from_first_diff']:
            # First diff_type defines outliers for all -> comparable
            outliers_diff = power_diffs['{}_vs_{}'.format(
                test_diffs[0][0], test_diffs[0][1])]
        else:
            # Individual ouliers for each diff_type
            outliers_diff = power_diffs[diff_type]

        if config['outliers_type'] == 'absolute':
            max_diff = config['outliers_absolute']
            select_outliers = np.logical_or(outliers_diff >= max_diff,
                                            outliers_diff <= -1 * max_diff)
        elif config['outliers_type'] == 'percentile':
            left, right = config['outliers_percentile']
            mask_outliers = get_central_data(outliers_diff,
                                             lower=left,
                                             upper=1-right,
                                             return_mask=True)
            select_outliers = np.logical_not(mask_outliers)

        if config['outliers_only']:
            # Evaluate only outliers
            power_diffs[diff_type].mask = np.logical_or(
                power_diffs[diff_type].mask, ~select_outliers)
        else:
            # Mask outliers
            power_diffs[diff_type].mask = np.logical_or(
                power_diffs[diff_type].mask, select_outliers)

        # Define relative power differences
        power_rel_diffs[diff_type] = power_diffs[diff_type]/res[
            'power_'+diff[1]]

        # Parameter differences
        # Apply 1-d power-mask to all 5 parameters
        single_param_mask = power_diffs[diff_type].mask
        x_opt_mask = np.zeros(res['x_opt_'+diff[0]].shape)
        for i in range(5):
            x_opt_mask[:, :, i] = single_param_mask
        x_opt_diff = res['x_opt_'+diff[0]] - res['x_opt_'+diff[1]]
        # Normalise absolute parameter differences with
        # their respective standard deviation for all samples
        norm_distribution = np.ma.array(res['x_opt_' + diff[1]],
                                        mask=x_opt_mask)
        norm_factor = np.std(norm_distribution.reshape((-1, 5)), axis=0)
        # Division by 0 if a parameter has no variance (is not optimized)
        # Thus non-optimized parameter differences are fully masked
        x_opt_diffs[diff_type] = np.ma.array(x_opt_diff/norm_factor,
                                             mask=x_opt_mask)

        # Control paramter outlier selection: #!!! add
        # instead of outliers selection to test - then drop?
        squared_normed_diff = (x_opt_diff**2)
        # Combined parameter distance for each sample
        distance = np.sqrt(np.sum(squared_normed_diff, axis=2))
        x_opt_outliers_sel = distance > 2.95
        # x_opt_outliers_sel = x_opt_diff[:, :, 1] < -0.95

        n_succ_outliers = np.sum(np.logical_and(
            res['power_' + config['main_sample']] > -1, select_outliers))
        n_succ_param_outliers = np.sum(np.logical_and(
            res['power_' + config['main_sample']] > -1, x_opt_outliers_sel))
        n_succ_both = np.sum(np.logical_and(np.logical_and(
            res['power_' + config['main_sample']] > -1, x_opt_outliers_sel),
            select_outliers))
        print('Selecting {} outliers via absolute/central discriminators'.
              format(n_succ_outliers) + ' and {} outliers'.
              format(n_succ_param_outliers) + ' via control parameters.'
              + ' Both selectors match for {} data points.'.
              format(n_succ_both))

    if config['outliers_from_first_diff']:
        # Modify initial data w.r.t outliers
        if config['outliers_only']:
            # Select only outliers
            print('Outlier only evaluation - set other data as "unsuccessful"')
            print('Reducing initial successful data to {} {} outliers'.
                  format(np.sum(select_outliers.reshape((-1))),
                         title_names[config['main_sample']]))
            for data_type in title_names.keys():
                # Set power values to '-1' for all data types:
                # all successful selection will select ONLY outliers
                res['power_'+data_type][~select_outliers] = -1
                res['power_'+data_type][
                    res['power_'+config['main_sample']] <= -1] = -1
                # res['power_'+data_type][power_diffs[
                #    config['main_diff']].mask] = -1
                # !!! this also selects only in cut-in/out by default for all
                # samples
                # only use main_sample <= -1 as selector?
        else:
            # Mask outliers
            print('Selected outliers are matched to failed optimizations:')
            print('Setting {} {} outliers from initial data as unsuccessful'.
                  format(np.sum(select_outliers.reshape((-1))),
                         title_names[config['main_sample']]))
            for data_type in title_names.keys():
                # Set power values to '-1' for all data types:
                # all successful selection will select ONLY data NO outliers
                # !!! only successful runs are compared
                res['power_'+data_type][select_outliers] = -1
                res['power_'+data_type][
                    res['power_'+config['main_sample']] <= -1] = -1
                # res['power_'+data_type][power_diffs[
                #    config['main_diff']].mask] = -1

                # !!! this also selects only in cut-in/out by default for all
                # samples
                # only use main_sample <= -1 as selector?
    else:
        print('No universal outliers definition: "outliers_from_first_diff"'
              'set to False - initial data not modified, no outliers removed')

    return res, power_diffs, power_rel_diffs, x_opt_diffs, titles


def plot_sample_power(data, title='',
                      plot_sels=['succ', 'succ, cut-in/out']):
    title_info = title.replace('/', '').replace('.', '').replace(' ', '_')
    data_sel = {'': True,
                'succ': data > -1,
                'succ, cut-in/out': np.logical_and(data > -1, select_v),
                'cut-in/out': select_v,
                }
    for plot_sel in plot_sels:
        sel = data_sel[plot_sel]
        plot_info = '_power' + plot_sel.replace('/', '_').replace(
            ' ', '_').replace(',', '_')
        plot_diff_pdf(data[sel], 'no', 'P', '[W]',
                      output_file_name=get_file_name(title_info
                                                     + plot_info),
                      title=' '.join([title, plot_sel]))


def eval_step_wise_mask_data(step_masks, step_tags, abs_diffs, rel_diffs,
                             min_samples=5, diff_tag='_diff_power',
                             title='', make_plots=False):
    # TODO use also for clustering validation:
    # optional file name function input?, labels, ...
    n_steps = len(step_masks)
    abs_res = np.ma.array(np.zeros([n_steps, 2]), mask=True)
    rel_res = np.ma.array(np.zeros([n_steps, 2]), mask=True)
    for step_idx in range(n_steps):
        if np.sum(np.logical_not(np.logical_or(
                abs_diffs.mask, step_masks[step_idx]))) < min_samples:
            print(' '.join(step_tags[step_idx].split('_')),
                  '- less than {} matches found, skipping'.format(min_samples))
            continue
        # Mean difference
        abs_data = abs_diffs[~np.logical_or(abs_diffs.mask,
                                            step_masks[step_idx])]
        abs_res[step_idx, :] = (np.mean(abs_data), np.std(abs_data))
        sel_mask = step_tags[step_idx]
        plot_info = diff_tag + sel_mask.replace(' ', '_')

        rel_data = rel_diffs[~np.logical_or(rel_diffs.mask,
                                            step_masks[step_idx])]
        rel_res[step_idx, :] = (np.mean(rel_data), np.std(rel_data))
        if make_plots:
            # Plot histograms of step data: absolute and relative diffs
            plot_diff_pdf(abs_data, 'absolute', 'P', '[W]',
                          output_file_name=get_file_name(title_info
                                                         + plot_info),
                          title=title+sel_mask)
            plot_diff_pdf(rel_data, 'relative', 'P', '[W]',
                          output_file_name=get_file_name(title_info
                                                         + plot_info),
                          title=title+sel_mask)
    return abs_res, rel_res


def eval_diffs_by_velocity_bins(wind_speed, split_velocities, power_diff):
    # Find velocity bin wise masks
    vel_bin_masks, vel_bin_tags = get_velocity_bin_mask(
        wind_speed, split_velocities=split_velocities)
    # Evaluate wind speed bins and plot bin wise pdfs
    vel_abs_res, vel_rel_res = eval_step_wise_mask_data(
        vel_bin_masks, vel_bin_tags, power_diff,
        power_rel_diffs[diff_type], title=title, make_plots=False)
    # List only bins that are filled with enough events
    sel = np.logical_not(vel_abs_res.mask[:, 0])
    split_velocities = np.array(split_velocities)[sel]
    x_ticks = [tag.replace('vel ', '').replace(' to ', '-').replace('0 up',
                                                                    'full')
               for tag in vel_bin_tags]
    plot_config = {'title': title + ' for wind speed ranges',
                   'x_label': 'wind speed ranges [m/s]',
                   'x_ticks': [tick for i, tick in enumerate(x_ticks)
                               if sel[i]],
                   'output_file_name': get_file_name(
                       title_info + 'diffs_vel_bins'),
                   }
    plot_abs_rel_step_wise(split_velocities, vel_abs_res[sel, :],
                           vel_rel_res[sel, :], **plot_config)


if __name__ == '__main__':
    #          Validation config:

    # Initial processing settings
    # Samples and diffs
    main_sample = 'cc_opt'  # 'sample'  #
    test_diffs = [(main_sample, 'cluster'),
                  # ('sample', 'cluster'),
                  # ('cc_opt', 'cluster'),
                  # ('sample', 'cc_opt')
                  ]
    title_names = {
        'sample': 'sample', 'cluster': 'cluster',
        'sample_cc': 'sample w/ cluster contr.',
        'cc_opt': 'opt. of sample w/ cluster contr.',
        }
    config = {
        # Main sample type to be evaluated
        'main_sample': main_sample,
        'main_diff': '{}_vs_cluster'.format(main_sample),
        # Print totals/percentages of raw successful runs & cut-in/out
        'raw_eval_perc_successful_velocity': False,
        # Differences to be evaluated

        # Outliers
        'outliers_type': 'absolute',  # 'absolute' or 'percentile'
        'outliers_absolute': 2500,  # minimal absolute diff of outliers
        'outliers_percentile': (0.09, 0.01),  # (left, right)% excluded percent
        'outliers_only': False,
        # Single reference for outliers:
        # also inital data filtered accordingly
        'outliers_from_first_diff': True,
        }
    # Raw power plots settings
    config_full_power_plots = [
        # (data type, [list of selection options]
        # options: ''(Full), 'succ', 'cut-in/out', 'succ, cut-in/out'
        ('sample', ['succ']),
        ('cluster', ['succ, cut-in/out']),
        ('sample_cc', ['succ']),
        ('cc_opt', ['succ']),
        ]
    # Power diff evaluation type settings
    power_diff_eval = {
        # Plot power diffs as histograms
        'pdf': True,
        # Plot maps of europe with mean and standard deviation of the diff
        'map': False,
        # Plot power diffs categorised by matching cluster id
        'cluster_id_bins': True,
        # Plot power diffs categorised by matching wind sped bins
        'wind_speed_bins': [0, 5, 10, 20, 0],
        # [0, 1.5, 3, 5, 10, 20, 0]
        }
    # Control parameter evaluation type settings
    control_parameter_eval = {
        'distance_pdf': True,
        'distance_map': True,
        'single_parameter_pdf': True,
        'test_bounds': False,
        }
    # AEP evaluation settings
    aep_eval = {
        'do_aep_eval': True,
        # Down_times: operational, maintenance
        'perc_down_time': 30.0,  # in %
        }

    # ---------------------------------------------
    print('------- Initializing...')
    res = read_single_sample_power_output()

    #          Define relevant selections:
    # Select samples in velocity range of cut-in/out wind speed of clustering
    v = res['backscaling']
    select_v = np.logical_and(v >= res['matching_vw_cut_in'],
                              v <= res['matching_vw_cut_out'])

    # Primary evaluation: numbers of successful optimizations / velocity range
    if config['raw_eval_perc_successful_velocity']:
        total = len((res['backscaling']).flat)
        eval_percentages_succ_vel('SAMPLE', total, res['power_'+main_sample],
                                  select_v)
        eval_percentages_succ_vel('CLUSTER', total, res['power_cluster'],
                                  select_v)

    # Process input data to evaluate selection of raw data,
    # power and operational parameter differences for 'test_diffs' combinations
    res, power_diffs, power_rel_diffs, x_opt_diffs, \
        titles = get_eval_diffs_and_data(test_diffs,
                                         res,
                                         config,
                                         title_names)

    print('------- Single parameter evaluation...')
    # Plot single power distributions
    for conf in config_full_power_plots:
        plot_sample_power(res['power_'+conf[0]], title=title_names[conf[0]],
                          plot_sels=conf[1])

    # Get single parameter distributions - multiple selections
    def get_data_list_distr_plots(data_type):
        data_list = [res[data_type].reshape((-1)),
                     res[data_type][res['power_' + main_sample] > -1],
                     res[data_type][np.logical_and(
                         res['power_' + main_sample] > -1, select_v)]]
        data_types = ['full', 'successful', 'successful and in cut-in/out']
        if config['outliers_only']:
            # Outliers evaluation only makes sense for successful data
            data_list = data_list[1:]
            data_types = data_types[1:]
        return data_list, data_types
    # Plot velocity distribution at 100m
    data_list, data_types = get_data_list_distr_plots('backscaling')
    plot_diff_pdf_mult_data(data_list, 'no', r'v$_{100}$', '[m/s]',
                            data_type=data_types,
                            output_file_name=get_file_name('100m_velocity'),
                            title='')
    # Plot cluster id distribution
    data_list, data_types = get_data_list_distr_plots('cluster_id')
    plot_diff_pdf_mult_data(data_list, 'no', r'cluster_id', '[-]',
                            data_type=data_types,
                            output_file_name=get_file_name('cluser_id'),
                            title='')

    print('------- Differences: Power evaluation...')
    # Evaluate differences
    for diff_type in power_diffs:
        title = titles[diff_type]
        title_info = title.replace('/', '').replace('.', '').replace(' ', '_')

        # Plot difference map
        if power_diff_eval['map']:
            # Mean
            plot_info = 'map_mean_diff_abs_power_full'
            plot_diff_map(res['locs'], np.abs(np.mean(
                power_diffs[diff_type], axis=1)),
                title=titles[diff_type], label='mean absolute diff dP [W]',
                output_file_name=get_file_name(title_info + plot_info))
            # Standard deviation
            plot_info = 'map_std_diff_abs_power_full'
            plot_diff_map(res['locs'], np.std(
                power_diffs[diff_type], axis=1), title=titles[diff_type],
                label='std absolute diff dP [W]',
                output_file_name=get_file_name(title_info + plot_info))
        if power_diff_eval['pdf']:
            abs_data = power_diffs[diff_type][~power_diffs[diff_type].mask]
            rel_data = power_rel_diffs[diff_type][~power_diffs[diff_type].mask]

            # Plot power diffs
            file_name = get_file_name(title_info + 'diff_abs_power_full')
            plot_diff_pdf(abs_data, 'absolute', 'P', '[W]',
                          output_file_name=file_name, title=title)

            file_name = get_file_name(title_info + 'diff_rel_power_full')
            plot_diff_pdf(rel_data, 'relative', 'dP/P', '[-]',
                          output_file_name=file_name, title=title)

        if power_diff_eval['cluster_id_bins']:
            # Plot diffs per cluster
            clusters = range(1, n_clusters+1)
            cluster_masks = [power_diffs[diff_type].mask]*n_clusters
            cluster_tags = ['']*n_clusters
            # Find cluster wise masks
            for cluster_id in clusters:
                cluster_masks[cluster_id-1] = np.logical_or(
                    power_diffs[diff_type].mask,
                    res['cluster_id'] != cluster_id)
                cluster_tags[cluster_id-1] = ' in cluster {}'.format(
                    cluster_id)
            cluster_abs_res, cluster_rel_res = eval_step_wise_mask_data(
                cluster_masks, cluster_tags, power_diffs[diff_type],
                power_rel_diffs[diff_type], title=title, make_plots=False)
            # List only clusters that are filled with enough events
            sel = np.logical_not(cluster_abs_res.mask[:, 0])
            clusters = np.array(list(clusters))[sel]

            plot_config = {'title': title + ' for individual clusters',
                           'x_label': 'cluster ids [-]',
                           'x_ticks': list(clusters),
                           'output_file_name': get_file_name(
                               title_info + 'diffs_per_cluster'),
                           }
            plot_abs_rel_step_wise(list(clusters), cluster_abs_res[sel, :],
                                   cluster_rel_res[sel, :], **plot_config)
        if power_diff_eval['wind_speed_bins']:
            # Plot diffs per wind speed bin
            split_velocities = power_diff_eval['wind_speed_bins']
            wind_speed = res['backscaling']
            eval_diffs_by_velocity_bins(wind_speed, split_velocities,
                                        power_diffs[diff_type])

    print('------- Control parameter evaluation...')

    # Overall differences/distances between control parameters x_opt
    for diff_type in x_opt_diffs:
        # Evaluate normed control parameter differences
        x_opt_diff = x_opt_diffs[diff_type]
        squared_normed_diff = (x_opt_diff**2)
        # Combined parameter distance for each sample
        distance = np.sqrt(np.sum(squared_normed_diff, axis=2))
        title = titles[diff_type]
        title_info = title.replace('/', '').replace('.', '').replace(' ', '_')
        plot_info = 'operational_parameter_distance'
        if control_parameter_eval['distance_pdf']:
            plot_diff_pdf(distance.reshape((-1)), 'no',
                          'sqrt(sum (dx_opt²/var x_opt))',
                          '[-]', output_file_name=get_file_name(
                              title_info + plot_info),
                          title=titles[diff_type])
        if control_parameter_eval['distance_map']:
            # Evaluate distance location wise
            plot_info = 'map_' + plot_info
            locs = res['locs']
            mean_distances = [np.mean(
                distance[i_loc, :][~distance.mask[i_loc, :]])
                for i_loc in range(len(locs))]
            # Check all locations have unmasked data
            test_masked = [np.ma.is_masked(loc_val) for
                           loc_val in mean_distances]
            if np.sum(test_masked) > 0:
                mean_distances = [mean_distance for i, mean_distance
                                  in enumerate(mean_distances)
                                  if not test_masked[i]]
                locs = [loc for i, loc in enumerate(locs)
                        if not test_masked[i]]
                dropping_locs = [loc for i, loc in enumerate(locs)
                                 if test_masked[i]]
                print('Dropping locations with only masked data: {}'.format(
                    dropping_locs))
            # Plot distance map
            plot_diff_map(
                locs,
                mean_distances, title=titles[diff_type],
                label='sqrt(sum (dx_opt²/var x_opt))',
                output_file_name=get_file_name(title_info + plot_info))

        # Single parameter differences
        var_name = ['tether force traction',
                    'tether force retraction',
                    'elevation angle traction',
                    'tether length diff',
                    'tether length min']
        if control_parameter_eval['single_parameter_pdf']:
            for i in range(5):
                # FIXME remove hardcoded 5 everywhere -> n control params
                data = x_opt_diff[:, :, i][
                    ~x_opt_diff.mask[:, :, i]].reshape((-1))
                if np.std(data) == 0:
                    # Tether length min not optimized
                    # (optimization: reduce_x only includes first 4 parameters)
                    print('No variation in operational parameter {} ({}).'.
                          format(i, var_name[i]) + ' Skipping.')
                    continue
                plot_info = 'operational_parameter_diff_rel_' \
                    + var_name[i].replace(' ', '_')
                file_name = get_file_name(title_info + plot_info)
                plot_diff_pdf(data, '', 'dx_opt/std(x_opt)', '[-]',
                              output_file_name=file_name,
                              title=': '.join([titles[diff_type],
                                               var_name[i]]))
    diff_type = main_sample+'_vs_cluster'
    single_param_mask = power_diffs[diff_type].mask
    x_opt_mask = np.zeros(res['x_opt_cluster'].shape)
    for i in range(5):
        x_opt_mask[:, :, i] = single_param_mask  # FIXME does this really do what I want? also check up in diff function
    x_opt_sample = np.ma.array(
        res['x_opt_sample'], mask=x_opt_mask)
    x_opt_cluster = np.ma.array(
        res['x_opt_cluster'], mask=x_opt_mask)
    x_opt_cc_opt = np.ma.array(
        res['x_opt_cc_opt'], mask=x_opt_mask)
    x_opt_mean_sample = np.mean(x_opt_sample, axis=(0, 1))
    x_opt_mean_cluster = np.mean(x_opt_cluster, axis=(0, 1))
    x_opt_mean_cc_opt = np.mean(x_opt_cc_opt, axis=(0, 1))
    x_opt_overall_mean = np.mean(np.array([x_opt_mean_sample,
                                           x_opt_mean_cluster,
                                           x_opt_mean_cc_opt]), axis=0)
    # !!! define function mean parameters
    print('Mean x_opt: ')
    print('   Sample: ', x_opt_mean_sample)
    print('   Sample cluster opt: ', x_opt_mean_cc_opt)
    print('   cluster: ', x_opt_mean_cluster)
    print('   overall: ', x_opt_overall_mean)
    # Check if most variation from when not on bounds:
    if control_parameter_eval['test_bounds']:
        bounds = [[3.00000000e+02, 5.00000000e+03],
                  [3.00000000e+02, 5.00000000e+03],
                  [4.36332313e-01, 1.04719755e+00],
                  [1.50000000e+02, 2.50000000e+02],
                  [1.50000000e+02, 2.50000000e+02]]
        # if cluster x_opt on bounds -> mean/ std of abs diffs

        def test_bounds(test_x0, precision=0.05):
            # TODO Test larger range of precision: 5% or so
            return [np.logical_or(np.abs(test_x0[i]-bounds[i][0]) <= precision,
                                  np.abs(test_x0[i]-bounds[i][1]) <= precision)
                    for i in range(len(test_x0))]

        on_bounds = np.array([test_bounds(x_opt_sample[i, :])
                              for i in range(x_opt_sample.shape[0])])
        on_bounds = on_bounds[~power_diffs['sample_vs_cluster']
                              .mask.reshape((-1)), :]
        for i in range(4):
            mask = on_bounds[:, i]
            print('{} - Differences when sample on bounds {} pm {} | \
                  not on bounds {} pm {} '.format(
                  var_name[i], np.mean(x_opt_diffs['sample_vs_cluster'][:, i]
                                       [on_bounds[:, i]]),
                  np.std(x_opt_diffs['sample_vs_cluster'][:, i]
                         [on_bounds[:, i]]),
                  np.mean(x_opt_diffs['sample_vs_cluster'][:, i]
                          [~on_bounds[:, i]]),
                  np.std(x_opt_diffs['sample_vs_cluster'][:, i]
                         [~on_bounds[:, i]])))

    # TODO 2d plot overall diffs x vs p

    # AEP evaluation
    if aep_eval['do_aep_eval']:
        print('------- AEP evaluation...')
        # AEP comparison
        hours_per_year = 24*365
        # Down_times: operational, maintenance
        annual_run_hours = hours_per_year * (1 -
                                             aep_eval['perc_down_time']/100.)

        # AEP estimated via clustering for selected samples
        power_cluster = np.ma.array(res['power_cluster'])
        power_cluster.mask = np.logical_or(res['power_cluster'] <= -1,
                                           ~select_v)
        loc_mean_power_cluster = np.mean(power_cluster, axis=1)
        loc_aep_cluster = loc_mean_power_cluster * annual_run_hours

        # AEP estimated via single sample optimization
        power_sample = np.ma.array(res['power_' + main_sample])
        power_sample.mask = res['power_' + main_sample] <= -1
        loc_mean_power_sample = np.mean(power_sample, axis=1)
        loc_aep_sample = loc_mean_power_sample * annual_run_hours

        title = 'Absolute AEP difference clustering - brute force'
        title_info = title.replace('/', '').replace('.', '').replace(' ', '_')
        # Plot difference map
        plot_info = 'map_aep_abs_diff_' + main_sample + '_cluster'
        plot_diff_map(res['locs'], loc_aep_sample-loc_aep_cluster,
                      title=title, label='dAEP [Wh]',
                      output_file_name=get_file_name(plot_info))

        title = 'Relative AEP difference clustering - brute force'
        title_info = title.replace('/', '').replace('.', '').replace(' ', '_')
        # Plot difference map
        plot_info = 'map_aep_rel_diff_' + main_sample + '_cluster'
        plot_diff_map(res['locs'], ((loc_aep_sample-loc_aep_cluster)
                      / loc_aep_sample), title=title, label='dAEP/AEP [-]',
                      output_file_name=get_file_name(plot_info))
        # TODO overall aep difference? + analyse where difference comes from
        # TODO differences of alls locs in histogram - mean+std

    if plots_interactive:
        print('------- Showing plots...')
        plt.show()
