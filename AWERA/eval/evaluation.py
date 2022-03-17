import pickle
import copy
from ..awera import ChainAWERA
from .eval_utils import sliding_window_avg, count_consecutive_bool

from ..utils.plotting_utils import plot_map, \
    plot_single_map, match_loc_data_map_data, \
    plot_percentile_ratios, plot_percentiles, \
    plot_optimal_height_and_wind_speed_timeline, plot_timeline

from ..utils.wind_resource_utils import calc_power

from .optimal_harvesting_height import get_wind_speed_at_height, \
    barometric_height_formula

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
                             power_lower_bar=None,
                             power_lower_perc=15,
                             read_if_possible=True,
                             locs_slice=None,
                             read_from_slices=None,
                             read_only='t_below_bar'):
        file_name = self.config.IO.plot_output.format(
            title='sliding_window_power').replace('.pdf', '.pickle')
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
                # TODO read for all locs slices and combine results
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
        if locs_slice is None:
            plot_map(self.config, t_below_bar/24,
                     title='Longest timspan below cutoff',
                     label=r'$\Delta$ t [d]',
                     log_scale=False,
                     n_decimals=0,
                     output_file_name=file_name.replace('.pickle', '.pdf'),
                     line_levels=[2, 5, 15, 20],
                     fill_range=None,
                     overflow=60)
            print('Ploting done.')
        return t_below_bar

    def aep_map(self):
        # TODO include here?
        from ..power_production.aep_map import aep_map
        aep_map(self.config)

    def power_freq(self):
        from ..power_production.plot_power_and_frequency import \
            plot_power_and_frequency
        plot_power_and_frequency(self.config)

# TODO split functions, make scales, inputs etc optional
    def harvesting_height_v_maps(self,
                                 v_at_harvesting_height,
                                 backscaling,
                                 v_at_compare_height=None,
                                 compare_height=None):
        # Average wind speed at harvesting altitude
        mean_v = np.mean(v_at_harvesting_height, axis=1)
        print('Plotting mean velocity at harvesting height...')
        plot_map(self.config, mean_v,
                 label='v [m/s]', title='Mean wind speed',
                 line_levels=[5, 8, 10, 12],
                 fill_range=[1, 13],
                 output_file_name=self.config.IO.plot_output.format(
                     title='mean_wind_speed_at_harvesting_height'))

        print('Plotting median velocity at harvesting height...')
        # Median wind speed at harvesting altitude
        median_v = np.median(v_at_harvesting_height, axis=1)
        plot_map(self.config, median_v,
                 label='v [m/s]', title='Median wind speed',
                 line_levels=[5, 8, 10, 12],
                 fill_range=[1, 13],
                 output_file_name=self.config.IO.plot_output.format(
                     title='median_wind_speed_at_harvesting_height'))

        # wind_speed_perc = np.percentile(v_at_harvesting_height,
        #                            (5, 32, 50),
        #                                axis=1)
        # TODO include? but wind speeds are not really correct for
        # low wind speeds -> cut-in/out effect

        # Average ratio of (wind speed at harvesting altitude) /
        # (wind speed at reference altitude)
        ratio_v = v_at_harvesting_height / backscaling
        mean_ratio_v = np.mean(ratio_v, axis=1)
        print('Plotting mean velocity ratio vs ref at harvesting height...')
        plot_map(self.config, mean_ratio_v,
                 label='v/v_ref [-]', title='Ratio using {:.0f}m'.format(
                     self.config.General.ref_height),
                 line_levels=[1.1, 1.3],
                 fill_range=[1, 2], n_decimals=1,
                 output_file_name=self.config.IO.plot_output.format(
                     title='mean_wind_speed_ratio_at_harvesting_height_vs_{}_m'
                     .format(self.config.General.ref_height)))

        # Average ratio of (wind speed at harvesting altitude) /
        # (wind speed at comparison altitude)
        if v_at_compare_height is not None:
            ratio_v = v_at_harvesting_height / v_at_compare_height
            mean_ratio_v = np.mean(ratio_v, axis=1)
            print('Plotting mean velocity ratio vs comparison at '
                  'harvesting height...')
            plot_map(self.config, mean_ratio_v,
                     label='v/v_ref [-]', title='Ratio using {:.0f}m'.format(
                         compare_height),
                     line_levels=[1.1, 1.3, 1.7],
                     fill_range=[1, 2], n_decimals=1,
                     output_file_name=self.config.IO.plot_output.format(
                         title='mean_wind_speed_at_harvesting_height_vs_{}_m'
                         .format(compare_height)))

    def harv_height_power_density_maps(self,
                                       harv_height,
                                       v_at_harvesting_height,
                                       backscaling,
                                       v_at_compare_height=None,
                                       compare_height=None):
        rho = barometric_height_formula(harv_height)
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

        plot_map(self.config, np.mean(power_density, axis=1)*10**(-3),
                 fill_range=[0, 2],
                 line_levels=[0.5, 1],
                 label=r'Power density [kW/m$^2$]',
                 title='Mean power density',
                 n_decimals=1,
                 output_file_name=self.config.IO.plot_output.format(
                     title='mean_power_density_at_harvesting_height'))

        # Eval power density at fixed height mean and ratio
        if v_at_compare_height is not None:
            v_at_h = v_at_compare_height
            rho = barometric_height_formula(compare_height)
            power_density_at_h = calc_power(v_at_h, rho)  # in W/m**2
            print('Power density: get percentiles...')
            power_density_perc_at_h = np.percentile(power_density_at_h,
                                                    (5, 32, 50),
                                                    axis=1)
            plot_percentiles(
                match_loc_data_map_data(self.config,
                                        power_density_perc_at_h[0, :]),
                match_loc_data_map_data(self.config,
                                        power_density_perc_at_h[1, :]),
                match_loc_data_map_data(self.config,
                                        power_density_perc_at_h[2, :]),
                output_file_name=self.config.IO.plot_output.format(
                     title='power_density_percentiles_at_{}_m_height'.format(compare_height))
                )

            plot_map(self.config, np.mean(power_density_at_h, axis=1)*10**(-3),
                     fill_range=[0, 2],
                     line_levels=[0.5, 1],
                     label=r'Power density [kW/m$^2$]',
                     title='Mean power density',
                     n_decimals=1,
                     output_file_name=self.config.IO.plot_output.format(
                         title='mean_power_density_at_{}_m'
                         .format(compare_height)))

            plot_map(self.config, np.mean(power_density/power_density_at_h,
                                          axis=1),
                     fill_range=[1, 3],
                     line_levels=[0.5, 1, 2],
                     label=r'p/p_ref [-]',
                     title='Mean power density ratio to {} m'.format(compare_height),
                     n_decimals=1,
                     output_file_name=self.config.IO.plot_output.format(
                         title='mean_power_density_ratio_at_harvesting_height_vs_{}_m'
                         .format(compare_height)))
            plot_percentile_ratios(
                match_loc_data_map_data(
                    self.config,
                    power_density_perc[0, :]/power_density_perc_at_h[0, :]),
                match_loc_data_map_data(
                    self.config,
                    power_density_perc[1, :]/power_density_perc_at_h[1, :]),
                match_loc_data_map_data(
                    self.config,
                    power_density_perc[2, :]/power_density_perc_at_h[2, :]),
                output_file_name=self.config.IO.plot_output.format(
                     title='power_density_percentiles_ratios_at_{}_m_height'.format(compare_height))
                )
        rho = barometric_height_formula(
            self.config.General.ref_height)
        power_density_at_ref = calc_power(backscaling, rho)  # in W/m**2
        print('Power density: get percentiles...')
        power_density_perc_at_ref = np.percentile(power_density_at_ref,
                                                  (5, 32, 50),
                                                  axis=1)
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

        plot_map(self.config, np.mean(power_density_at_ref, axis=1)*10**(-3),
                 fill_range=[0, 3],
                 line_levels=[0.5, 1],
                 label=r'Power density [kW/m$^2$]',
                 title='Mean power density at {} m'.format(self.config.General.ref_height),
                 n_decimals=1,
                 output_file_name=self.config.IO.plot_output.format(
                     title='mean_power_density_at_{}_m'
                     .format(self.config.General.ref_height)))

        plot_map(self.config, np.mean(power_density/power_density_at_ref,
                                      axis=1),
                 fill_range=[1, 3],
                 line_levels=[1, 1.25, 1.5, 1.75, 2],
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
        power_down = np.sum(power == 0, axis=1)/power.shape[1]*100
        plot_map(self.config, power_down,
                 fill_range=[0, 55],
                 line_levels=[30, 45],
                 label='v not in cut-in/out [%]',
                 title='Down Percentage',
                 n_decimals=0,
                 output_file_name=self.config.IO.plot_output.format(
                     title='cluster_down_percentage'))
        # of estimated power production

    def eval_wind_speed_at_harvesting_height(self):
        print('Starting eval...')
        v_at_harvesting_height, backscaling = \
            get_wind_speed_at_height(self.config)

        compare_height = 60  # m
        v_at_compare_height, _ = get_wind_speed_at_height(
            self.config,
            set_height=compare_height)
        # TODO wind speed not considering masking out of cut-in/cut-out

        self.harvesting_height_v_maps(
            v_at_harvesting_height,
            backscaling,
            v_at_compare_height=v_at_compare_height,
            compare_height=compare_height)

        # 5%, 32%, 50% percentiles of wind power density at harvesting altitude
        print('Plotting power density at harvesting height...')
        # TODO need rho, density levels determined by height calculation...

        # Harvesting height
        # TODO read pd file
        with open(
                self.config.IO.plot_output.format(
                    title='data_avg_power_cycle_height_from_cluster').replace(
                        '.pdf', '.pickle'), 'rb') as f:
            height_file = pickle.load(f)
        harv_height, harv_height_range, _, _ = \
            height_file['height'], height_file['height_range'],\
            height_file['backscaling'], height_file['label']
        # harv_height, _, _, _ = \
        #     get_data_avg_power_cycle_height_from_cluster(config)
        print('Plotting power density at harvesting height maps...')
        self.harv_height_power_density_maps(harv_height,
                                            v_at_harvesting_height,
                                            backscaling)
        print('Reading cluster power...')
        p_cluster, _, _, n_samples_per_loc = \
            self.match_clustering_power_results()
        power = p_cluster.reshape((self.config.Data.n_locs, n_samples_per_loc))
        # TODO for all locs: wirte to file once, use for plotting later
        # Plot power maps
        print('Plotting power maps...')
        self.power_maps(power=power)

        # Plot single location timeline
        print('Plotting single location timelines...')
        self.single_loc_timelines(v_at_harvesting_height,
                                  harv_height,
                                  harv_height_range,
                                  backscaling,
                                  power)

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
