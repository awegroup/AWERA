import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def get_cluster_avg_power_cycle_height_vs_wind_speed(config):
    harvesting_height = []
    harvesting_height_range = {'min': [], 'max': []}
    ref_wind_speeds = []
    # Read kpis: z component
    # ??? user power curve constructor or just read pickle file?
    # pc = PowerCurveConstructor(None)
    # setattr(pc, 'plots_interactive', config.Plotting.plots_interactive)
    # setattr(pc, 'plot_output_file', config.IO.training_plot_output)
    # pc.import_results(config.power_curve.format(i_profile, 'pickle'))
    for i_profile in range(1, config.Clustering.n_clusters + 1):
        with open(config.IO.power_curve.format(i_profile=i_profile,
                                               suffix='pickle'), 'rb') as f:
            pc_file = pickle.load(f)
            performance_indicators_all = pc_file['performance_indicators']
            succ = [kpis['sim_successful']
                    for kpis in performance_indicators_all]
            ref_wind = pc_file['wind_speeds']
        # Select successful runs only
        performance_indicators = [kpis for i, kpis
                                  in enumerate(performance_indicators_all)
                                  if succ[i]]
        # Calculate mean/range of optimal harvesting height

        mean_opt_harvesting_height = [np.mean([kin.z for kin in kpis['kinematics']])
                                      for kpis in performance_indicators]
        max_harvesting_height = [np.max([kin.z for kin in kpis['kinematics']])
                                 for kpis in performance_indicators]
        min_harvesting_height = [np.min([kin.z for kin in kpis['kinematics']])
                                 for kpis in performance_indicators]
        harvesting_height.append(mean_opt_harvesting_height)
        harvesting_height_range['max'].append(max_harvesting_height)
        harvesting_height_range['min'].append(min_harvesting_height)
        ref_wind_speeds.append(ref_wind)
    return harvesting_height, harvesting_height_range, ref_wind_speeds


def get_data_avg_power_cycle_height_from_cluster(config):
    # Get clustering power production simulation results
    harvesting_height, harvesting_height_range, ref_wind_speeds = \
        get_cluster_avg_power_cycle_height_vs_wind_speed(config)

    # Read labels and backscaling factors
    with open(config.IO.labels, 'rb') as f:
        clustering_output = pickle.load(f)
    data_matching_cluster = clustering_output['labels [-]']
    data_backscaling_from_cluster = clustering_output['backscaling [m/s]']
    n_samples_per_loc = clustering_output['n_samples_per_loc']
    # -> cluster id and matching v_100m
    data_harvesting_height = np.zeros(data_matching_cluster.shape)
    data_max_harvesting_height = np.zeros(data_matching_cluster.shape)
    data_min_harvesting_height = np.zeros(data_matching_cluster.shape)
    # TODO parallelize! if config allows it
    for i_cluster in range(config.Clustering.n_clusters):
        matched_cluster_id = data_matching_cluster == i_cluster
        print('{} samples in cluster {}'.format(sum(matched_cluster_id),
                                                i_cluster))
        data_v = data_backscaling_from_cluster[matched_cluster_id]
        h_of_v = harvesting_height[i_cluster]
        min_h_of_v = harvesting_height_range['min'][i_cluster]
        max_h_of_v = harvesting_height_range['max'][i_cluster]
        v = ref_wind_speeds[i_cluster]
        # TODO interpolation the best here?
        data_harvesting_height[matched_cluster_id] = np.interp(data_v,
                                                               v,
                                                               h_of_v)
        data_min_harvesting_height[matched_cluster_id] = np.interp(data_v,
                                                                   v,
                                                                   min_h_of_v)
        data_max_harvesting_height[matched_cluster_id] = np.interp(data_v,
                                                                   v,
                                                                   max_h_of_v)
    # TODO save  clusters also starting from 1 same as profiles
    # TODO make return backsaling optional?

    # Reshape into location-wise data
    res_height = np.zeros((config.Data.n_locs, n_samples_per_loc))
    res_min_height = np.zeros((config.Data.n_locs, n_samples_per_loc))
    res_max_height = np.zeros((config.Data.n_locs, n_samples_per_loc))
    res_backscaling = np.zeros((config.Data.n_locs, n_samples_per_loc))
    res_cluster = np.zeros((config.Data.n_locs, n_samples_per_loc))
    # TODO  make nicer
    for i in range(config.Data.n_locs):
        res_height[i, :] = \
            data_harvesting_height[i*n_samples_per_loc:(i+1)*n_samples_per_loc]
        res_min_height[i, :] = \
            data_min_harvesting_height[i*n_samples_per_loc:
                                       (i+1)*n_samples_per_loc]
        res_max_height[i, :] = \
            data_max_harvesting_height[i*n_samples_per_loc:
                                       (i+1)*n_samples_per_loc]
        res_backscaling[i, :] = \
            data_backscaling_from_cluster[
                i*n_samples_per_loc:(i+1)*n_samples_per_loc]
        res_cluster[i, :] = \
            data_matching_cluster[i*n_samples_per_loc:(i+1)*n_samples_per_loc]
    #df = pd.DataFrame({'height': res_height, 'backscaling': res_backscaling,
    #                   'label': res_cluster})
    #df.to_csv(config.IO.plot_output.format(
    #    title='data_avg_power_cycle_height_from_cluster').replace('.pdf',
    #                                                                 '.csv'))
    with open(
            config.IO.plot_output.format(
                title='data_avg_power_cycle_height_from_cluster').replace(
                    '.pdf', '.pickle'), 'wb') as f:
        pickle.dump({'height': res_height,
                     'height_range': {'min': res_min_height,
                                      'max': res_max_height},
                     'backscaling': res_backscaling,
                     'label': res_cluster}, f)
    print(res_height)
    print(res_backscaling)
    print(res_cluster)
    return res_height, {'min': res_min_height, 'max': res_max_height},\
        res_backscaling, res_cluster


def get_data_power_from_cluster(config):
    # Get clustering power production simulation results

    # Read labels and backscaling factors
    with open(config.IO.labels, 'rb') as f:
        clustering_output = pickle.load(f)
    data_matching_cluster = clustering_output['labels [-]']
    backscaling = clustering_output['backscaling [m/s]']
    n_samples_per_loc = clustering_output['n_samples_per_loc']
    # -> cluster id and matching v_100m
    data_power = np.zeros(data_matching_cluster.shape)
    for i_cluster in range(config.Clustering.n_clusters):
        df = pd.read_csv(config.IO.power_curve.format(
            suffix='csv', i_profile=i_cluster+1), sep=";")
        v = df['v_100m [m/s]'].values
        # TODO hard coded 100m
        p_of_v = df['P [W]'].values
        matched_cluster_id = data_matching_cluster == i_cluster
        data_v = backscaling[matched_cluster_id]
        data_power[matched_cluster_id] = np.interp(data_v,
                                                   v,
                                                   p_of_v)
        # Read cut-in / out
        limit_estimates = pd.read_csv(config.IO.refined_cut_wind_speeds)
        cut_in = limit_estimates.iloc[i_cluster]['vw_100m_cut_in']
        cut_out = limit_estimates.iloc[i_cluster]['vw_100m_cut_out']
        sel_down_times = np.logical_and(
            np.logical_or(backscaling < cut_in, backscaling > cut_out),
            matched_cluster_id)
        data_power[sel_down_times] = 0

    # TODO make return backsaling optional?

    # Reshape into location-wise data
    res_power = np.zeros((config.Data.n_locs, n_samples_per_loc))
    # TODO  make nicer
    for i in range(config.Data.n_locs):
        res_power[i, :] = \
            data_power[i*n_samples_per_loc:(i+1)*n_samples_per_loc]
    return res_power


def read_cluster_profiles(config, descale_ref=False):
    df = pd.read_csv(config.IO.profiles, sep=";")
    heights = df['height [m]']
    prl = np.zeros((config.Clustering.n_clusters, len(heights)))
    prp = np.zeros((config.Clustering.n_clusters, len(heights)))
    for i in range(config.Clustering.n_clusters):
        u = df['u{} [-]'.format(i+1)]
        v = df['v{} [-]'.format(i+1)]
        sf = df['scale factor{} [-]'.format(i+1)]
        if descale_ref:
            u = u/sf
            v = v/sf
        prl[i, :] = u
        prp[i, :] = v
    return prl, prp, heights


def get_wind_speed_at_height(config, set_height=-1):
    try:
        with open(
                config.IO.plot_output.format(
                    title='data_avg_power_cycle_height_from_cluster').replace(
                        '.pdf', '.pickle'), 'rb') as f:
            height_file = pickle.load(f)
        harvesting_height, backscaling, matching_cluster = \
            height_file['height'], height_file['backscaling'], height_file['label']
    except FileNotFoundError:
        harvesting_height, harv_height_range, backscaling, matching_cluster = \
            get_data_avg_power_cycle_height_from_cluster(config)
    # TODO make optional: rerun?
    # TODO read labels file if set height, make correct shape again
    if set_height > 0:
        harvesting_height = np.zeros(harvesting_height.shape) + set_height

    # Read cluster wind profile shapes
    prl, prp, heights = read_cluster_profiles(config, descale_ref=False)
    v = (prl**2 + prp**2)**0.5
    print('cluster profiles: ', v.shape, heights)

    v_at_harvesting_height = np.zeros(matching_cluster.shape)
    for i_cluster in range(config.Clustering.n_clusters):
        matched_cluster_id = matching_cluster == i_cluster
        cluster_v = v[i_cluster, :]
        cluster_h = heights
        data_h = harvesting_height[matched_cluster_id]

        v_scaled = np.interp(data_h,
                             cluster_h,
                             cluster_v)
        v_at_harvesting_height[matched_cluster_id] = \
            v_scaled * backscaling[matched_cluster_id]
        # TODO or really read sample?
    return v_at_harvesting_height, backscaling


# UTILS


def match_loc_data_map_data(config, data):
    # Match locations with values - rest NaN
    n_lats = len(config.Data.all_lats)
    n_lons = len(config.Data.all_lons)
    data_loc = np.ma.array(np.zeros((n_lats, n_lons)), mask=True)
    # TODO right way around?
    for i, i_loc in enumerate(config.Data.i_locations):
        data_loc[i_loc[0], i_loc[1]] = data[i]
    return data_loc


def barometric_height_formula(h):
    rho_0 = 1.2250  # kg/m³ at +15°C (288.15K)
    g = 9.81  # N/kg
    R_s = 287.058  # J/kgK
    T = 288.15  # K
    rho = rho_0 * np.exp(-g * h / (R_s * T))
    return rho


# PLOTTING UTILS


def plot_map(data,
             fill_range=[0, 20],
             line_levels=[2, 5, 15, 20],
             label='v [m/s]',
             plot_title='',
             n_decimals=0):
    from ..resource_analysis.plot_maps import eval_contour_fill_levels, \
        plot_single_panel
    # TODO automatize with data?
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


def plot_percentiles(perc_5, perc_32, perc_50):
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


def plot_percentile_ratios(perc_5, perc_32, perc_50,
                           plot_range=[1, 2.7]):
    """" Generate power density percentile plot. """
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


# FULL EVAL
def eval_wind_speed_at_harvesting_height(config):

    # TODO split function
    print('Starting eval...')
    v_at_harvesting_height, backscaling = \
        get_wind_speed_at_height(config)

    compare_height = 60  # m
    v_at_compare_height, _ = get_wind_speed_at_height(config,
                                                      set_height=compare_height)
    def eval_v():
        # Average wind speed at harvesting altitude
        mean_v = np.mean(v_at_harvesting_height, axis=1)
        print('Plotting mean velocity at harvesting height...')
        plot_map(match_loc_data_map_data(config, mean_v),
                 fill_range=[5, 12],
                 line_levels=[8, 10],
                 label='v [m/s]',
                 plot_title='Mean wind speed')

        print('Plotting median velocity at harvesting height...')
        # Median wind speed at harvesting altitude
        median_v = np.median(v_at_harvesting_height, axis=1)
        plot_map(match_loc_data_map_data(config, median_v),
                 fill_range=[5, 12],
                 line_levels=[8, 10],
                 label='v [m/s]',
                 plot_title='Median wind speed')
        #wind_speed_perc = np.percentile(v_at_harvesting_height,
        #                            (5, 32, 50),
        #                                axis=1)
        # TODO include? but wind speeds are not really correct for low wind speeds -> cut-in/out effect

        # Average ratio of (wind speed at harvesting altitude) /
        # (wind speed at reference altitude)
        ratio_v = v_at_harvesting_height / backscaling
        mean_ratio_v = np.mean(ratio_v, axis=1)
        print('Plotting mean velocity ratio vs ref at harvesting height...')
        plot_map(match_loc_data_map_data(config, mean_ratio_v),
                 fill_range=[1, 1.5],
                 line_levels=[1.1, 1.3],
                 label='v/v_ref [-]',
                 plot_title='Ratio using {:.0f}m'.format(
                     config.General.ref_height),
                 n_decimals=1)

        # Average ratio of (wind speed at harvesting altitude) /
        # (wind speed at comparison altitude)
        ratio_v = v_at_harvesting_height / v_at_compare_height
        mean_ratio_v = np.mean(ratio_v, axis=1)
        print('Plotting mean velocity ratio vs comparison at harvesting height...')
        plot_map(match_loc_data_map_data(config, mean_ratio_v),
                 fill_range=[1, 1.5],
                 line_levels=[1.1, 1.3],
                 label='v/v_ref [-]',
                 plot_title='Ratio using {:.0f}m'.format(compare_height),
                 n_decimals=1)
    eval_v()


    # 5%, 32%, 50% percentiles of wind power density at harvesting altitude
    print('Plotting power density at harvesting height...')
    from ..resource_analysis.process_data import calc_power
    # TODO need rho, density levels determined by height calculation...

    # Harvesting height
    # TODO read pd file
    with open(
            config.IO.plot_output.format(
                title='data_avg_power_cycle_height_from_cluster').replace(
                    '.pdf', '.pickle'), 'rb') as f:
        height_file = pickle.load(f)
    harv_height, harv_height_range, _, _ = \
        height_file['height'], height_file['height_range'],\
        height_file['backscaling'], height_file['label']
    # harv_height, _, _, _ = \
    #     get_data_avg_power_cycle_height_from_cluster(config)

    power = get_data_power_from_cluster(config)

    def eval_p():
        rho = barometric_height_formula(harv_height)
        # TODO rho -> from height calc era5 data and not baromatric...
        power_density = calc_power(v_at_harvesting_height, rho)  # in W/m**2
        print('Power density: get percentiles...')
        power_density_perc = np.percentile(power_density,
                                           (5, 32, 50),
                                           axis=1)
        plot_percentiles(match_loc_data_map_data(config,
                                                 power_density_perc[0, :]),
                         match_loc_data_map_data(config,
                                                 power_density_perc[1, :]),
                         match_loc_data_map_data(config,
                                                 power_density_perc[2, :]))

        plot_map(match_loc_data_map_data(config,
                                         np.mean(power_density,
                                                 axis=1)*10**(-3)),
                 fill_range=[0, 2],
                 line_levels=[0.5, 1],
                 label=r'Power density [kW/m$^2$]',
                 plot_title='Mean power density',
                 n_decimals=1)
        # Eval power density at fixed height mean and ratio
        v_at_h = v_at_compare_height
        rho = barometric_height_formula(compare_height)
        power_density_at_h = calc_power(v_at_h, rho)  # in W/m**2
        print('Power density: get percentiles...')
        power_density_perc_at_h = np.percentile(power_density_at_h,
                                                (5, 32, 50),
                                                axis=1)
        plot_percentiles(
            match_loc_data_map_data(config, power_density_perc_at_h[0, :]),
            match_loc_data_map_data(config, power_density_perc_at_h[1, :]),
            match_loc_data_map_data(config, power_density_perc_at_h[2, :])
            )

        plot_map(
            match_loc_data_map_data(config, np.mean(power_density_at_h,
                                                    axis=1)*10**(-3)),
            fill_range=[0, 2],
            line_levels=[0.5, 1],
            label=r'Power density [kW/m$^2$]',
            plot_title='Mean power density',
            n_decimals=1
            )
        plot_map(
            match_loc_data_map_data(config,
                                    np.mean(power_density/power_density_at_h,
                                            axis=1)),
            fill_range=[1, 2.6],
            line_levels=[0.5, 1],
            label=r'p/p_ref [-]',
            plot_title='Mean power density ratio',
            n_decimals=1
            )
        plot_percentile_ratios(
            match_loc_data_map_data(
                config,
                power_density_perc[0, :]/power_density_perc_at_h[0, :]),
            match_loc_data_map_data(
                config,
                power_density_perc[1, :]/power_density_perc_at_h[1, :]),
            match_loc_data_map_data(
                config,
                power_density_perc[2, :]/power_density_perc_at_h[2, :])
            )
        rho = barometric_height_formula(
            config.General.ref_height)
        power_density_at_ref = calc_power(backscaling, rho)  # in W/m**2
        print('Power density: get percentiles...')
        power_density_perc_at_ref = np.percentile(power_density_at_ref,
                                                  (5, 32, 50),
                                                  axis=1)
        plot_percentiles(
            match_loc_data_map_data(config, power_density_perc_at_ref[0, :]),
            match_loc_data_map_data(config, power_density_perc_at_ref[1, :]),
            match_loc_data_map_data(config, power_density_perc_at_ref[2, :])
            )

        plot_map(
            match_loc_data_map_data(config, np.mean(power_density_at_ref,
                                                    axis=1)*10**(-3)),
            fill_range=[0, 2.6],
            line_levels=[0.5, 1], # TODO number of deciamls
            label=r'Power density [kW/m$^2$]',
            plot_title='Mean power density',
            n_decimals=1
            )
        plot_map(
            match_loc_data_map_data(config,
                                    np.mean(power_density/power_density_at_ref,
                                            axis=1)),
            fill_range=[1, 2],
            line_levels=[1, 1.25, 1.5, 1.75],
            label=r'p/p_ref [-]',
            plot_title='Mean power density ratio',
            n_decimals=1
            )
        plot_percentile_ratios(
            match_loc_data_map_data(
                config,
                power_density_perc[0, :]/power_density_perc_at_ref[0, :]),
            match_loc_data_map_data(
                config,
                power_density_perc[1, :]/power_density_perc_at_ref[1, :]),
            match_loc_data_map_data(
                config,
                power_density_perc[2, :]/power_density_perc_at_ref[2, :]),
            plot_range=[1, 1.7]
            )
    eval_p()
    # Plot single location timeline
    def eval_single_loc():
        print('Plotting single location timelines...')
        location = (54, -10.25)  # County Mayo
        # TODO put in config?
        # Select location - find location index
        for i, loc in enumerate(config.Data.locations):
            if loc == location:
                i_loc = i
                break

        # Plot timeline - general
        from ..resource_analysis.single_loc_plots import plot_figure_5a, \
            plot_timeline
        # TODO get sample hours
        from datetime import datetime
        date_1 = '01/01/1900 00:00:00.000000'
        date_2 = '01/01/{} 00:00:00.000000'.format(config.Data.start_year)
        date_format_str = '%d/%m/%Y %H:%M:%S.%f'
        start = datetime.strptime(date_1, date_format_str)
        end = datetime.strptime(date_2, date_format_str)
        # Get interval between two timstamps as timedelta object
        diff = end - start
        # Get interval between two timstamps in hours
        start_hours = diff.total_seconds() / 3600
        # n_hours is n_samples_per_loc
        n_hours = v_at_harvesting_height.shape[1]
        hours = np.arange(start_hours, start_hours + n_hours, step=1)
        # Wind speed at harvesting height
        loc_v_at_harvesting_height = v_at_harvesting_height[i_loc, :]
        print('... v at harvesting height selected')
        loc_harv_height = harv_height[i_loc, :]
        loc_harv_height_range = {'min': harv_height_range['min'][i_loc, :],
                                 'max': harv_height_range['max'][i_loc, :]}
        limit_estimates = pd.read_csv(config.IO.refined_cut_wind_speeds)
        cut_in, cut_out = [], []
        for i_cluster in range(config.Clustering.n_clusters):
            cut_in.append(limit_estimates.iloc[i_cluster]['vw_100m_cut_in'])
            cut_out.append(limit_estimates.iloc[i_cluster]['vw_100m_cut_out'])
        avg_cut_in_out = [np.mean(cut_in), np.mean(cut_out)]
        print(avg_cut_in_out)
        print('cut-in/out evaluated')
        # Plot timeline
        plot_figure_5a(hours, loc_v_at_harvesting_height, loc_harv_height,
                       # heights_of_interest, ceiling_id, floor_id,
                       height_range=loc_harv_height_range,
                       ref_velocity=backscaling[i_loc, :],
                       height_bounds=[-1, 389.71],  # 60deg at 450m length #TODO make optional/automate
                       v_bounds=[avg_cut_in_out[0], 20], #TODO v_bounds from average over all clusters - not really showing max v_bounds - maybe just use max in the end?
                       show_n_hours=24*7)

        # Power
        print('Plotting power...')


        # Get power for location
        loc_power = power[i_loc, :]
        # Plot timeline
        plot_timeline(hours, loc_power,
                      ylabel='Power [W]',
                      # heights_of_interest, ceiling_id, floor_id,
                      # data_bounds=[50, 500],
                      show_n_hours=24*7)

    #eval_single_loc()

    def eval_power_map():
        # Power maps 'we talked about'
        # 5% percentile power map
        power_perc = np.percentile(power, 20, axis=1)
        plot_map(match_loc_data_map_data(config, power_perc),
                 fill_range=[0, 300],
                 line_levels=[100],
                 label='P [W]',
                 plot_title='20th Percentile')  # of estimated power production
        power_mean = np.mean(power, axis=1)*10**(-3)
        plot_map(match_loc_data_map_data(config, power_mean),
                 fill_range=[0, 7],
                 line_levels=[3, 5],
                 label='P [kW]',
                 plot_title='Mean Power Production')  # of estimated power production
        power_down = np.sum(power == 0, axis=1)/power.shape[1]*100
        plot_map(match_loc_data_map_data(config, power_down),
                 fill_range=[0, 55],
                 line_levels=[30, 45],
                 label='v not in cut-in/out [%]',
                 plot_title='Down Percentage')  # of estimated power production
    #eval_power_map()
    # TODO capacity factor increase vs fixed height - wind turbine power curve?


    # TODO below
    # --- at night

    # Longest timespan below XX power production map
    # same only night
    # same with sliding window avg?

    # more...
    #plt.show()