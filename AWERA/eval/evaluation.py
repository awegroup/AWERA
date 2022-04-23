import pickle
import pandas as pd
import copy
from ..awera import ChainAWERA
from .eval_utils import sliding_window_avg, count_consecutive_bool

from ..utils.plotting_utils import plot_map, \
    plot_single_map, match_loc_data_map_data, \
    plot_percentile_ratios, plot_percentiles, \
    plot_optimal_height_and_wind_speed_timeline, plot_timeline

from ..utils.wind_resource_utils import calc_power

from ..wind_profile_clustering.read_requested_data import get_wind_data

from .optimal_harvesting_height import get_wind_speed_at_height, \
    barometric_height_formula, get_data_avg_power_cycle_height_from_cluster

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from datetime import datetime

class evalAWERA(ChainAWERA):
    def __init__(self, config):
        """Initialise Clustering and Production classes."""
        super().__init__(config)

    def sliding_window_power(self,
                             time_window=24,  # Hours for hourly data
                             at_night=False,
                             power_lower_bar=None,
                             power_lower_perc=15,
                             read_if_possible=True,
                             locs_slice=None,
                             read_from_slices=None,
                             read_only='t_below_bar'):
        file_name = self.config.IO.plot_output.format(
            title='sliding_window_power').replace('.pdf', '.pickle')
        if time_window != 24:
            file_name = file_name.replace(
                    '.pickle',
                    '_{:.0f}.pickle'.format(time_window))
        if at_night:
            file_name = file_name.replace(
                    '.pickle',
                    'at_night.pickle')
        if locs_slice is not None:
            file_name = file_name.replace(
                    '.pickle',
                    '{}_n_{}.pickle'.format(locs_slice[0], locs_slice[1]))

        res = {}
        if read_if_possible:
            try:
                if read_from_slices is not None:
                    if locs_slice is not None:
                        raise ValueError('Cannot read from all locs slices '
                                         'and a single loc slice at the same'
                                         ' time')

                    def combine_labels(n_i=read_from_slices[0],
                                       n_max=read_from_slices[1]):
                        loc_file_name = file_name.replace(
                            '.pickle',
                            '{{}}_n_{}.pickle'.format(n_max))
                        for i in range(n_i):
                            print('{}/{}'.format(i+1, n_i))
                            with open(loc_file_name.format(i), 'rb') as f:
                                res_i = pickle.load(f)
                            # First label is start for res
                            if i == 0:
                                res = {}
                                res[read_only] = np.ma.empty(
                                    [self.config.Data.n_locs])

                            # Append labels to results, append locations
                            end = (i+1)*read_from_slices[1]
                            if end > self.config.Data.n_locs:
                                end = self.config.Data.n_locs
                            res[read_only][i*read_from_slices[1]:end] = \
                                res_i[read_only]
                        return res
                    res = combine_labels()
                else:
                    with open(file_name, 'rb') as f:
                        res = pickle.load(f)

            except FileNotFoundError:
                res = {}
        if len(res) == 0:
            if power_lower_bar is None:
                # Use percentage of nominal power
                p_nominal = self.clustering_nominal_power()
                power_lower_bar = power_lower_perc/100. * p_nominal

            # Read clustering power, masked values are out of
            # cut-in/out wind speed window of matched power curve
            p_cluster, _, _, n_samples_per_loc = \
                self.match_clustering_power_results(locs_slice=locs_slice)

            # Restructure, per location
            if locs_slice is not None:
                n_locs = len(p_cluster)/n_samples_per_loc
                if n_locs != int(n_locs):
                    raise ValueError('Cannot interpret n_locs, '
                                     'does not match n_samples_per_loc!')
                else:
                    n_locs = int(n_locs)
            else:
                n_locs = self.config.Data.n_locs
            p = p_cluster.reshape((n_locs, n_samples_per_loc))

            if at_night:
                night_time_file = self.config.IO.plot_output.format(
                    title='datetime_night_mask').replace('.pdf', '.pickle')
                try:
                    if not read_if_possible:
                        raise FileNotFoundError('Not supposed to read file.')

                    with open(night_time_file, 'rb') as f:
                        night_mask = pickle.load(f)
                except FileNotFoundError:
                    yearly_night_time = {
                        # inclusive night hours: in Standard european time
                        1: [17, 7], 2: [18, 7], 3: [19, 6], 4: [20, 5],
                        5: [21, 4], 6: [22, 3], 7: [22, 3], 8: [21, 4],
                        9: [20, 5], 10: [19, 6], 11: [18, 6], 12: [17, 7]}
                    night_mask = np.empty((n_samples_per_loc), dtype='bool')
                    single_loc_datetime = get_wind_data(
                        self.config,
                        locs=[self.config.Data.locations[0]])['datetime']
                    dates = pd.to_datetime(single_loc_datetime)
                    months = dates.month
                    hours = dates.hour
                    for i in range(1, 13):
                        month_mask = months == i
                        month_hours = hours[month_mask]
                        night_time = yearly_night_time[i]
                        night_mask[month_mask] = np.logical_or(
                            # UTC = Standard European time - 1
                            month_hours >= night_time[0] - 1,
                            month_hours <= night_time[1] - 1)
                    # Save night mask
                    with open(night_time_file, 'wb') as f:
                        pickle.dump(night_mask, f)
                p = p[:, night_mask]

            # Take a sliding window average over the given time_window
            p_avg = sliding_window_avg(p, time_window)
            print(p_avg)
            p_above_bar = p_avg >= power_lower_bar
            print(p_above_bar)
            # Count consecutive times above bar
            t_above_bar, t_below_bar = count_consecutive_bool(p_above_bar)
            res = {'p_avg': p_avg,
                   'bar': power_lower_bar,
                   'time_window': time_window,
                   't_above_bar': t_above_bar,
                   't_below_bar': t_below_bar}

            with open(file_name, 'wb') as f:
                pickle.dump(res, f)
            print(t_below_bar)
            print('Output Written.')
        else:
            t_below_bar = res['t_below_bar']
            # t_above_bar = res['t_above_bar']
        # Plot map
        if np.min(t_below_bar/24) < 1:
            n_decimals = 1
        else:
            n_decimals = 0
        if at_night:
            day = 10  # hours avg length of night
            label = r'$\Delta$ t [nights]'
        else:
            day = 24  # hours a day
            label = r'$\Delta$ t [d]'
        if locs_slice is None:
            plot_map(self.config, t_below_bar/day,
                     title='Longest timespan below cutoff',
                     label=label,
                     log_scale=False,
                     n_decimals=n_decimals,
                     output_file_name=file_name.replace('.pickle', '.pdf'),
                     line_levels=[2, 5, 15, 20],
                     fill_range=None,
                     overflow=60)
            print('Plotting done.')
        return t_below_bar

    def aep_map(self):
        # TODO include here?
        from ..power_production.aep_map import aep_map
        aep_map(self.config)

    def power_freq(self):
        from ..power_production.plot_power_and_frequency import \
            plot_power_and_frequency
        plot_power_and_frequency(self.config)

    # def plot_operational_timeline(self):
    #     param = []
    #     param_range = {'min': [], 'max': []}
    #     ref_wind_speeds = []
    #     # Read kpis: z component
    #     # ??? user power curve constructor or just read pickle file?
    #     # pc = PowerCurveConstructor(None)
    #     # setattr(pc, 'plots_interactive', config.Plotting.plots_interactive)
    #     # setattr(pc, 'plot_output_file', config.IO.training_plot_output)
    #     # pc.import_results(config.power_curve.format(i_profile, 'pickle'))
    #     for i_profile in range(1, config.Clustering.n_clusters + 1):
    #         with open(config.IO.power_curve.format(i_profile=i_profile,
    #                                                suffix='pickle'), 'rb') as f:
    #             pc_file = pickle.load(f)
    #             performance_indicators_all = pc_file['performance_indicators']
    #             succ = [kpis['sim_successful']
    #                     for kpis in performance_indicators_all]
    #             ref_wind = pc_file['wind_speeds']
    #         # Select successful runs only
    #         performance_indicators = [kpis for i, kpis
    #                                   in enumerate(performance_indicators_all)
    #                                   if succ[i]]
    #         # Calculate mean/range of optimal harvesting height

    #         mean_opt_harvesting_height = [np.mean([kin.z for kin in kpis['kinematics']])
    #                                       for kpis in performance_indicators]
    #         max_harvesting_height = [np.max([kin.z for kin in kpis['kinematics']])
    #                                  for kpis in performance_indicators]
    #         min_harvesting_height = [np.min([kin.z for kin in kpis['kinematics']])
    #                                  for kpis in performance_indicators]
    #         harvesting_height.append(mean_opt_harvesting_height)
    #         harvesting_height_range['max'].append(max_harvesting_height)
    #         harvesting_height_range['min'].append(min_harvesting_height)
    #         ref_wind_speeds.append(ref_wind)

    #     # evaluate for data:
    #     # Read labels and backscaling factors
    #     with open(config.IO.labels, 'rb') as f:
    #         clustering_output = pickle.load(f)
    #     data_matching_cluster = clustering_output['labels [-]']
    #     data_backscaling_from_cluster = clustering_output['backscaling [m/s]']
    #     n_samples_per_loc = clustering_output['n_samples_per_loc']
    #     # -> cluster id and matching v_100m
    #     data_harvesting_height = np.zeros(data_matching_cluster.shape)
    #     data_max_harvesting_height = np.zeros(data_matching_cluster.shape)
    #     data_min_harvesting_height = np.zeros(data_matching_cluster.shape)
    #     mask_cut_wind_speeds = np.empty((data_matching_cluster.shape),
    #                                     dtype='bool')
    #     print('mask sum before:', np.sum(mask_cut_wind_speeds))
    #     limit_estimates = pd.read_csv(config.IO.refined_cut_wind_speeds)
    #     # TODO parallelize! if config allows it
    #     for i_cluster in range(config.Clustering.n_clusters):

    #         matched_cluster_id = data_matching_cluster == i_cluster

    #         # Read cut-in / out
    #         cut_in = limit_estimates.iloc[i_cluster]['vw_100m_cut_in']
    #         cut_out = limit_estimates.iloc[i_cluster]['vw_100m_cut_out']
    #         sel_down_times = np.logical_and(
    #             np.logical_or(data_backscaling_from_cluster < cut_in,
    #                           data_backscaling_from_cluster > cut_out),
    #             matched_cluster_id)
    #         print('Cluster {} has {} down times'.format(
    #             i_cluster+1,
    #             np.sum(sel_down_times)/np.sum(matched_cluster_id)))
    #         mask_cut_wind_speeds[sel_down_times] = True
    #         print('{} samples in cluster {}'.format(sum(matched_cluster_id),
    #                                                 i_cluster))
    #         data_v = data_backscaling_from_cluster[matched_cluster_id]
    #         h_of_v = harvesting_height[i_cluster]
    #         min_h_of_v = harvesting_height_range['min'][i_cluster]
    #         max_h_of_v = harvesting_height_range['max'][i_cluster]
    #         v = ref_wind_speeds[i_cluster]
    #         # TODO interpolation the best here?
    #         data_harvesting_height[matched_cluster_id] = np.interp(data_v,
    #                                                                v,
    #                                                                h_of_v)
    #         data_min_harvesting_height[matched_cluster_id] = np.interp(data_v,
    #                                                                    v,
    #                                                                    min_h_of_v)
    #         data_max_harvesting_height[matched_cluster_id] = np.interp(data_v,
    #                                                                    v,
    #                                                                    max_h_of_v)
    #         # Plot timeline
    #         loc_power = power[i_loc, :]
    #         if np.ma.isMaskedArray(loc_power):
    #             loc_power[loc_power.mask] = 0
    #         # Plot timeline
    #         plot_timeline(hours, loc_power,
    #                       ylabel='Mean Cycle Power [W]',
    #                       # heights_of_interest, ceiling_id, floor_id,
    #                       # data_bounds=[50, 500],
    #                       show_n_hours=24*7,
    #                       plots_interactive=self.config.Plotting.plots_interactive,
    #                       output_file_name=self.config.IO.plot_output.format(
    #                           title='cluster_mean_cycle_power_timeline'))

    def read_locs_slices_results(self, file_name, locs_slice=(23, 1000),
                                 res_part=['mean_v', 'mean_ratio_v', 'compare_height'],
                                 do_not_combine = ['compare_height']):
        loc_tag = '{}_n_{}.pickle'
        if file_name[-len(loc_tag):] != loc_tag:
            file_name = file_name.replace('.pickle', loc_tag).replace('.pdf', loc_tag)

        n_locs = self.config.Data.n_locs
        res = {}
        res_i = np.empty((n_locs))

        if res_part is None:
            res = res_i
        else:
            for r in res_part:
                if r not in do_not_combine:
                    if 'perc' in r:
                        res[r] = np.empty((3, n_locs))
                    else:
                        res[r] = copy.deepcopy(res_i)

        # Read locs slices:
        for i in range(locs_slice[0]):
            print(i)
            end = (i+1)*locs_slice[1]
            if end > n_locs:
                end = n_locs
            start = i*locs_slice[1]

            fn = file_name.format(i, locs_slice[1])
            # Read diff results
            with open(fn, 'rb') as f:
                res_i = pickle.load(f)
            if res_part is not None:
                for r in res_part:
                    print(r)
                    if r not in do_not_combine:
                        print(res[r].shape)
                        print(res[r][start:end].shape)
                        if 'perc' in r:
                            res[r][:, start:end] = res_i[r]
                        else:
                            res[r][start:end] = res_i[r]
                    elif i == 0:
                        res[r] = res_i[r]
            else:
                res[start:end] = res_i
        return res

    # TODO split functions, make scales, inputs etc optional
    def harvesting_height_v_maps(self,
                                 v_at_harvesting_height=None,
                                 # backscaling,
                                 v_at_compare_height=None,
                                 compare_height=None,
                                 tag=''):
        # Average wind speed at harvesting altitude
        try:
            mean_v = self.read_locs_slices_results(self.config.IO.plot_output.format(
                title='wind_speed_at_harvesting_height{}'.format(tag)), res_part=['mean_v'])['mean_v']
        except FileNotFoundError:
            mean_v = np.mean(v_at_harvesting_height, axis=1)
        print('Plotting mean velocity at harvesting height...')
        plot_map(self.config, mean_v,
                  label='v [m/s]', title='Mean wind speed',
                  line_levels=[5, 8, 10, 12],
                  fill_range=[1, 13],
                  output_file_name=self.config.IO.plot_output.format(
                      title='mean_wind_speed_at_harvesting_height{}'.format(tag)))

        # # wind_speed_perc = np.percentile(v_at_harvesting_height,
        # #                            (5, 32, 50),
        # #                                axis=1)
        # # TODO include? but wind speeds are not really correct for
        # # low wind speeds -> cut-in/out effect

        # # Average ratio of (wind speed at harvesting altitude) /
        # # (wind speed at reference altitude)
        # # ratio_v = v_at_harvesting_height / backscaling
        # # mean_ratio_v = np.mean(ratio_v, axis=1)
        # # print('Plotting mean velocity ratio vs ref at harvesting height...')
        # # plot_map(self.config, mean_ratio_v,
        # #          label='v/v_ref [-]', title='Ratio using {:.0f}m'.format(
        # #              self.config.General.ref_height),
        # #          line_levels=[1.1, 1.3],
        # #          fill_range=[1, 2], n_decimals=1,
        # #          output_file_name=self.config.IO.plot_output.format(
        # #              title='mean_wind_speed_ratio_at_harvesting_height_vs_{}_m{}'
        # #              .format(self.config.General.ref_height, tag)))

        # Average ratio of (wind speed at harvesting altitude) /
        # (wind speed at comparison altitude)
        res = {'mean_v': mean_v}
        if True:  # v_at_compare_height is not None:
            try:
                res = self.read_locs_slices_results(self.config.IO.plot_output.format(
                    title='wind_speed_at_harvesting_height{}'
                    .format(tag)), res_part=['mean_ratio_v', 'compare_height'])
                mean_ratio_v = res['mean_ratio_v']
            except FileNotFoundError:
                # Ignore Numpy Masked floating point error bug for now
                np.seterr(all='raise', under='ignore')
                ratio_v = v_at_harvesting_height / v_at_compare_height
                np.seterr(all='raise')
                mean_ratio_v = np.mean(ratio_v, axis=1)
            print('Plotting mean velocity ratio vs comparison at '
                  'harvesting height...')
            plot_map(self.config, mean_ratio_v,
                      label='v/v_ref [-]', title='Ratio using {:.0f}m'.format(
                          compare_height),
                      line_levels=[0.5, 0.7, 1.1, 1.3, 1.7],
                      fill_range=[0, 1.3], n_decimals=1,
                      output_file_name=self.config.IO.plot_output.format(
                          title='mean_wind_speed_at_harvesting_height_vs_{}_m{}'
                          .format(compare_height, tag)))
            res['mean_ratio_v'] = mean_ratio_v
            res['compare_height'] = compare_height
        # with open(
        #         self.config.IO.plot_output.format(
        #             title='wind_speed_at_harvesting_height{}'
        #                   .format(tag)).replace(
        #                   '.pdf', '.pickle'), 'wb') as f:
        #     pickle.dump(res, f)

    def harv_height_power_density_maps(self,
                                       harv_height,
                                       v_at_harvesting_height,
                                       backscaling,
                                       v_at_compare_height=None,
                                       compare_height=None):
        rho = barometric_height_formula(harv_height)

        try:
            power = self.read_locs_slices_results(self.config.IO.plot_output.format(
                title='power_density_percentiles'),
                res_part=['power_density_perc', 'power_density_perc_at_ref', 'mean_ratio'])
            power_density_perc = power['power_density_perc']
            power_density_perc_at_ref = power['power_density_perc_at_ref']
            mean_ratio = power['mean_ratio']
        except FileNotFoundError:
            # TODO rho -> from height calc era5 data and not baromatric...
            power_density = calc_power(v_at_harvesting_height, rho)  # in W/m**2
            print('Power density: get percentiles...')
            power_density_perc = np.percentile(power_density,
                                               (5, 32, 50),
                                               axis=1)
        plot_percentiles(match_loc_data_map_data(self.config,
                                                  power_density_perc[0, :]),
                          match_loc_data_map_data(self.config,
                                                  power_density_perc[1, :]),
                          match_loc_data_map_data(self.config,
                                                  power_density_perc[2, :]),
                          output_file_name=self.config.IO.plot_output.format(
                      title='power_density_percentiles_at_harvesting_height'))

        # # plot_map(self.config, np.mean(power_density, axis=1)*10**(-3),
        # #           fill_range=[0, 2],
        # #           line_levels=[0.5, 1],
        # #           label=r'Power density [kW/m$^2$]',
        # #           title='Mean power density',
        # #           n_decimals=1,
        # #           output_file_name=self.config.IO.plot_output.format(
        # #               title='mean_power_density_at_harvesting_height'))

        # Eval power density at fixed height mean and ratio
        # if v_at_compare_height is not None:
        #     v_at_h = v_at_compare_height
        #     rho = barometric_height_formula(compare_height)
        #     power_density_at_h = calc_power(v_at_h, rho)  # in W/m**2
        #     print('Power density: get percentiles...')
        #     power_density_perc_at_h = np.percentile(power_density_at_h,
        #                                             (5, 32, 50),
        #                                             axis=1)
        #     plot_percentiles(
        #         match_loc_data_map_data(self.config,
        #                                 power_density_perc_at_h[0, :]),
        #         match_loc_data_map_data(self.config,
        #                                 power_density_perc_at_h[1, :]),
        #         match_loc_data_map_data(self.config,
        #                                 power_density_perc_at_h[2, :]),
        #         output_file_name=self.config.IO.plot_output.format(
        #              title='power_density_percentiles_at_{}_m_height'.format(compare_height))
        #         )

        #     plot_map(self.config, np.mean(power_density_at_h, axis=1)*10**(-3),
        #              fill_range=[0, 2],
        #              line_levels=[0.5, 1],
        #              label=r'Power density [kW/m$^2$]',
        #              title='Mean power density',
        #              n_decimals=1,
        #              output_file_name=self.config.IO.plot_output.format(
        #                  title='mean_power_density_at_{}_m'
        #                  .format(compare_height)))

        #     plot_map(self.config, np.mean(power_density/power_density_at_h,
        #                                   axis=1),
        #              fill_range=[1, 3],
        #              line_levels=[0.5, 1, 2],
        #              label=r'p/p_ref [-]',
        #              title='Mean power density ratio to {} m'.format(compare_height),
        #              n_decimals=1,
        #              output_file_name=self.config.IO.plot_output.format(
        #                  title='mean_power_density_ratio_at_harvesting_height_vs_{}_m'
        #                  .format(compare_height)))
        #     plot_percentile_ratios(
        #         match_loc_data_map_data(
        #             self.config,
        #             power_density_perc[0, :]/power_density_perc_at_h[0, :]),
        #         match_loc_data_map_data(
        #             self.config,
        #             power_density_perc[1, :]/power_density_perc_at_h[1, :]),
        #         match_loc_data_map_data(
        #             self.config,
        #             power_density_perc[2, :]/power_density_perc_at_h[2, :]),
        #         output_file_name=self.config.IO.plot_output.format(
        #              title='power_density_percentiles_ratios_at_{}_m_height'.format(compare_height))
        #         )
        # rho = barometric_height_formula(
        #     self.config.General.ref_height)
        # power_density_at_ref = calc_power(backscaling, rho)  # in W/m**2
        # print('Power density: get percentiles...')
        # power_density_perc_at_ref = np.percentile(power_density_at_ref,
        #                                           (5, 32, 50),
        #                                           axis=1)
        plot_percentiles(
            match_loc_data_map_data(self.config,
                                    power_density_perc_at_ref[0, :]),
            match_loc_data_map_data(self.config,
                                    power_density_perc_at_ref[1, :]),
            match_loc_data_map_data(self.config,
                                    power_density_perc_at_ref[2, :]),
            output_file_name=self.config.IO.plot_output.format(
                      title='power_density_percentiles_at_{}_m_height'.format(self.config.General.ref_height))
            )

        # # plot_map(self.config, np.mean(power_density_at_ref, axis=1)*10**(-3),
        # #          fill_range=[0, 3],
        # #          line_levels=[0.5, 1],
        # #          label=r'Power density [kW/m$^2$]',
        # #          title='Mean power density at {} m'.format(self.config.General.ref_height),
        # #          n_decimals=1,
        # #          output_file_name=self.config.IO.plot_output.format(
        # #              title='mean_power_density_at_{}_m'
        # #              .format(self.config.General.ref_height)))
        # np.seterr(all='raise', under='ignore')
        # mean_ratio = np.mean(power_density/power_density_at_ref,
        #                      axis=1)
        # np.seterr(all='raise')

        plot_map(self.config, mean_ratio,
                  fill_range=[0, 3],
                  line_levels=[0.8, 1, 1.25, 1.5, 1.75, 2],
                  label=r'p/p_ref [-]',
                  title='Mean power density ratio to {} m'.format(self.config.General.ref_height),
                  n_decimals=1,
                  output_file_name=self.config.IO.plot_output.format(
                      title='mean_power_density_ratio_at_harvesting_height_vs_{}_m'
                      .format(self.config.General.ref_height)))

        plot_percentile_ratios(
            match_loc_data_map_data(
                self.config,
                power_density_perc[0, :]/power_density_perc_at_ref[0, :]),
            match_loc_data_map_data(
                self.config,
                power_density_perc[1, :]/power_density_perc_at_ref[1, :]),
            match_loc_data_map_data(
                self.config,
                power_density_perc[2, :]/power_density_perc_at_ref[2, :]),
            plot_range=[1, 1.7],
            output_file_name=self.config.IO.plot_output.format(
                      title='power_density_percentiles_ratios_at_{}_m_height'.format(self.config.General.ref_height))
            )

        # res = {
        #     'power_density_perc': power_density_perc,
        #     'power_density_perc_at_ref': power_density_perc_at_ref,
        #     'mean_ratio': mean_ratio,

        #     }
        # with open(
        #         self.config.IO.plot_output.format(
        #             title='power_density_percentiles').replace(
        #                 '.pdf', '.pickle'), 'wb') as f:
        #     pickle.dump(res, f)

    def single_loc_timelines(self,
                             v_at_harvesting_height,
                             harv_height,
                             harv_height_range,
                             backscaling,
                             power,
                             location=(52.0, 5.0)  # Cabauw (Netherlands)
                             ):
        print('Plotting single location timelines...')

        # Select location - find location index
        for i, loc in enumerate(self.config.Data.locations):
            if loc == location:
                i_loc = i
                break

        # Plot timeline - general
        # TODO get sample hours
        date_1 = '01/01/1900 00:00:00.000000'
        date_2 = '01/01/{} 00:00:00.000000'.format(self.config.Data.start_year)
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
        limit_estimates = pd.read_csv(self.config.IO.refined_cut_wind_speeds)
        cut_in, cut_out = [], []
        for i_cluster in range(self.config.Clustering.n_clusters):
            cut_in.append(limit_estimates.iloc[i_cluster]['vw_100m_cut_in'])
            cut_out.append(limit_estimates.iloc[i_cluster]['vw_100m_cut_out'])
        avg_cut_in_out = [np.mean(cut_in), np.mean(cut_out)]
        print(avg_cut_in_out)
        print('cut-in/out evaluated')
        # Plot timeline
        plot_optimal_height_and_wind_speed_timeline(
            hours, loc_v_at_harvesting_height, loc_harv_height,
            # heights_of_interest, ceiling_id, floor_id,
            height_range=loc_harv_height_range,
            ref_velocity=backscaling[i_loc, :],
            height_bounds=[-1, 389.71],
            # 60deg at 450m length #TODO make optional/automate
            v_bounds=[avg_cut_in_out[0], avg_cut_in_out[1]],  # 20],
            # TODO v_bounds from average over all clusters
            # - not really showing max v_bounds - maybe just use max/min...?
            show_n_hours=24*7,
            plots_interactive=self.config.Plotting.plots_interactive,
            output_file_name=self.config.IO.plot_output.format(
                     title='optimal_height_and_wind_speed_timeline'))

        # Power
        print('Plotting single location power...')

        # Get power for location
        loc_power = power[i_loc, :]
        if np.ma.isMaskedArray(loc_power):
            loc_power[loc_power.mask] = 0
        # Plot timeline
        plot_timeline(hours, loc_power,
                      ylabel='Mean Cycle Power [W]',
                      # heights_of_interest, ceiling_id, floor_id,
                      # data_bounds=[50, 500],
                      show_n_hours=24*7,
                      plots_interactive=self.config.Plotting.plots_interactive,
                      output_file_name=self.config.IO.plot_output.format(
                          title='cluster_mean_cycle_power_timeline'))

        with open(
                self.config.IO.plot_output.format(
                    title='cluster_single_loc_timeline').replace(
                        '.pdf', '.pickle'), 'wb') as f:
            pickle.dump({
                'power': loc_power,
                'hours': hours,
                'loc_v_at_harv_height': loc_v_at_harvesting_height,
                'loc_harv_height': loc_harv_height,
                'loc_harv_height_range': loc_harv_height_range,
                'loc_ref_velocity': backscaling[i_loc, :],
                'height_bounds': [-1, 389.71],
                'v_bounds': [avg_cut_in_out[0], avg_cut_in_out[1]],  # 20],
                }, f)

    def power_maps(self,
                   power=None):
        if power is None:
            p_cluster, _, _, n_samples_per_loc = \
                self.match_clustering_power_results()
            power = p_cluster.reshape((self.config.Data.n_locs,
                                       n_samples_per_loc))

        # Power maps
        # 5% percentile power map
        power_perc = np.percentile(power, 20, axis=1)
        plot_map(self.config, power_perc,
                 fill_range=[0, 300],
                 line_levels=[100],
                 label='P [W]',
                 title='20th Percentile',
                 n_decimals=0,
                 output_file_name=self.config.IO.plot_output.format(
                     title='cluster_power_20th_percentile'))
        # of estimated power production
        plot_map(self.config, np.mean(power, axis=1)*10**(-3),
                 fill_range=[0, 7],
                 line_levels=[3, 5],
                 label='P [kW]',
                 title='Mean Cycle Power (mechanical)',
                 n_decimals=0,
                 output_file_name=self.config.IO.plot_output.format(
                     title='cluster_mean_cycle_power'))
        # of estimated power production

        # of estimated power production

    def eval_wind_speed_at_harvesting_height(self, read_heights=True,
                                             processing_only=False):
        print('Starting eval...')
        # TODO implement locs slices..?

        # Harvesting height
        # TODO read pd file
        # if read_heights:
        #     with open(
        #             self.config.IO.plot_output.format(
        #                 title='data_avg_power_cycle_height_from_cluster').replace(
        #                     '.pdf', '.pickle'), 'rb') as f:
        #         height_file = pickle.load(f)
        #     harv_height, harv_height_range, _, _, down_times = \
        #         height_file['height'], height_file['height_range'],\
        #         height_file['backscaling'], height_file['label'], \
        #         height_file['down_times']
        # else:
        #     harv_height, harv_height_range, _, _, down_times = \
        #         get_data_avg_power_cycle_height_from_cluster(self.config)
        # print('Harvesting heights read in/processed.')
        # if processing_only:
        #     return 0

        print('Plotting harvesting height maps...')

        try:
            height = self.read_locs_slices_results(self.config.IO.plot_output.format(
                title='map_harv_height'),
                res_part=['mean', 'min', 'max'])
            min_height = height['min']
            max_height = height['max']
            mean_height = height['mean']
        except FileNotFoundError:
            mean_height = np.empty(harv_height.shape[0])
            min_height = np.empty(harv_height.shape[0])
            max_height = np.empty(harv_height.shape[0])
            for i_loc in range(harv_height.shape[0]):
                mean_height[i_loc] = np.mean(harv_height[i_loc, :][
                    np.logical_not(down_times[i_loc, :])])
                min_height[i_loc] = np.min(harv_height_range['min'][i_loc, :][
                    np.logical_not(down_times[i_loc, :])])
                max_height[i_loc] = np.max(harv_height_range['max'][i_loc, :][
                    np.logical_not(down_times[i_loc, :])])

        # with open(
        #         self.config.IO.plot_output.format(
        #             title='map_harv_height').replace(
        #                 '.pdf', '.pickle'), 'wb') as f:
        #     pickle.dump({'mean': mean_height,
        #                  'min': min_height,
        #                  'max': max_height}, f)
        plot_map(self.config, mean_height,
                  line_levels=[160, 180],
                  label='height [m]',
                  title='AWES harvesting height',
                  n_decimals=0,
                  output_file_name=self.config.IO.plot_output.format(
                      title='map_mean_harv_height'))
        plot_map(self.config, min_height,
                  line_levels=[80, 100, 150, 200, 250, 300],
                  label='height [m]',
                  title='Minimal AWES harvesting height',
                  n_decimals=0,
                  output_file_name=self.config.IO.plot_output.format(
                      title='map_min_harv_height'))
        plot_map(self.config, max_height,
                  line_levels=[150, 200, 250, 300, 350, 400, 450, 500],
                  label='height [m]',
                  title='Maximum AWES harvesting height',
                  n_decimals=0,
                  output_file_name=self.config.IO.plot_output.format(
                      title='map_max_harv_height'))

        try:
            power_down = self.read_locs_slices_results(self.config.IO.plot_output.format(
                title='cluster_down_percentage'),
                res_part=None)
        except FileNotFoundError:
            power_down = np.sum(down_times, axis=1)/down_times.shape[1]*100
        print('Plotting down percentage...')

        plot_map(self.config, power_down,
                  fill_range=[0, 55],
                  line_levels=[30, 45],
                  label='v not in cut-in/out [%]',
                  title='Down Percentage',
                  n_decimals=0,
                  output_file_name=self.config.IO.plot_output.format(
                      title='cluster_down_percentage'))

        # with open(
        #         self.config.IO.plot_output.format(
        #             title='cluster_down_percentage').replace(
        #                 '.pdf', '.pickle'), 'wb') as f:
        #     pickle.dump(power_down, f)

        # v_at_harvesting_height, backscaling = \
        #     get_wind_speed_at_height(self.config)

        # print('Reading cluster power...')
        # p_cluster, _, _, n_samples_per_loc = \
        #     self.match_clustering_power_results()
        # power = p_cluster.reshape((len(self.config.Data.locations), n_samples_per_loc))

        # # Plot single location timeline
        # # Here the wind speeds and heights are not masked/zeroes when out of cut-in/out
        # # The power is always 0/masked out of cut-in/out
        # print('Plotting single location timelines...')
        # # TODO processing for timelines: single location only
        # self.single_loc_timelines(v_at_harvesting_height,
        #                           harv_height,
        #                           harv_height_range,
        #                           backscaling,
        #                           power)
        compare_height = 100  # m
        # v_at_compare_height, _ = get_wind_speed_at_height(
        #     self.config,
        #     set_height=compare_height)

        # # Wind speed considering masking out of cut-in/cut-out
        # v_at_harvesting_height[down_times] = 0
        print('Plotting wind speed maps and ratios')
        self.harvesting_height_v_maps(
            # v_at_harvesting_height,
            # backscaling,
            # v_at_compare_height=v_at_compare_height,
            compare_height=compare_height,
            tag='_0_failing_cut')

        # 5%, 32%, 50% percentiles of wind power density at harvesting altitude
        # TODO need rho, density levels determined by height calculation...
        print('Plotting power density at harvesting height maps...')
        self.harv_height_power_density_maps(0, 0, 0) # harv_height,
                                            #v_at_harvesting_height,
                                            #backscaling)


        # Mask wind speeds of samples out of cut-in/out range
        # v_at_harvesting_height = np.ma.array(v_at_harvesting_height,
        #                                      mask=down_times)
        # v_at_compare_height = np.ma.array(v_at_compare_height,
        #                                   mask=down_times)
        self.harvesting_height_v_maps(
            # v_at_harvesting_height,
            # backscaling,
            v_at_compare_height=1, # v_at_compare_height,
            compare_height=compare_height,
            tag='_mask_failing_cut')

        # TODO for all locs: wirte to file once, use for plotting later
        # Plot power maps
        # in AEP map functionality..
        # print('Plotting power maps...')
        # self.power_maps(power=power)



        # TODO capacity factor increase vs fixed height
        # - wind turbine power curve?
        # see AWERA.power_production.aep_map.compare_cf_AWE_turbine()

        # TODO below
        # --- at night

        # Longest timespan below XX power production map
        # same only night
        # same with sliding window avg?

        # more...
        # plt.show()

    # ------------------------------------------------------------------------

    # AWERA.wind_profile_clustering.plot_location_maps.plot_location_map(config)
    # TODO auto-save plots - done for all? plt.show()

    # TODO AWERA.resource_analysis.plot_maps.plot_all()
