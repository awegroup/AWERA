import pickle
from ..awera import ChainAWERA
from .eval_utils import sliding_window_avg, count_consecutive_bool

from ..utils.plotting_utils import plot_map

import matplotlib.pyplot as plt

class evalAWERA(ChainAWERA):
    def __init__(self, config):
        """Initialise Clustering and Production classes."""
        super().__init__(config)

    def sliding_window_power(self,
                             time_window=24,  # Hours for hourly data
                             power_lower_bar=None,
                             power_lower_perc=15,
                             read_if_possible=True):
        file_name = self.config.IO.plot_output.format(
            title='sliding_window_power').replace('.pdf', '.pickle')
        if read_if_possible:
            try:
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
                self.match_clustering_power_results()

            # Restructure, per location
            p = p_cluster.reshape((self.config.Data.n_locs, n_samples_per_loc))

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
        else:
            t_below_bar = res['t_below_bar']
            t_above_bar = res['t_above_bar']
        # Plot map
        plot_map(self.config, t_below_bar,
                 title='Longest timspan below cutoff',
                 label='dt [h]',
                 log_scale=True)
        plt.savefig(file_name.replace('.pickle', '.pdf'))
        # TODO $delta in label
        return t_above_bar, t_below_bar


    #AWERA.eval.optimal_harvesting_height.eval_wind_speed_at_harvesting_height(config)
    #AWERA.power_production.aep_map.compare_cf_AWE_turbine()
    #AWERA.resource_analysis.plot_maps.plot_all()
    #import matplotlib.pyplot as plt
    def aep_map(self):
        # TODO include here?
        from ..power_production.aep_map import aep_map
        aep_map(self.config)

    def power_freq(self):
        from ..power_production.plot_power_and_frequency import \
            plot_power_and_frequency
        plot_power_and_frequency(self.config)

    #AWERA.wind_profile_clustering.plot_location_maps.plot_location_map(config)
    #plt.show()
