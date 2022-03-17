from .wind_profile_clustering.clustering import Clustering
from .power_production.power_production import PowerProduction

from .utils.plotting_utils import plot_map
import numpy as np
import pandas as pd

class ChainAWERA(Clustering, PowerProduction):
    def __init__(self, config):
        """Initialise Clustering and Production classes."""
        super().__init__(config)

    def run(self):
        make_freq = self.config.Clustering.make_freq_distr
        if make_freq:
            setattr(self.config.Clustering, 'make_freq_distr', False)
        self.run_clustering()

        self.run_curves()

        if make_freq:
            setattr(self.config.Clustering, 'make_freq_distr', True)
            self.get_frequency()

    def match_clustering_power_results(self,
                                       i_loc=None,
                                       single_sample_id=None,
                                       return_backscaling=False,
                                       locs_slice=None):
        # Returning masked array of power for
        # masking v outside of wind speed bounds of power curve
        labels, backscaling, n_samples_per_loc, _ = self.read_labels(
            locs_slice=locs_slice)

        if single_sample_id is None and i_loc is None:
            matching_cluster = labels
            profile_ids = np.array(matching_cluster) + 1
            backscaling = np.array(backscaling)
        elif single_sample_id is None:
            matching_cluster = np.array(labels[i_loc*n_samples_per_loc:
                                               (i_loc+1)*n_samples_per_loc])
            backscaling = np.array(backscaling[i_loc*n_samples_per_loc:
                                               (i_loc+1)*n_samples_per_loc])
            profile_ids = np.array(matching_cluster) + 1
        else:
            matching_cluster = np.array([labels[
                single_sample_id + i_loc*n_samples_per_loc]])
            backscaling = np.array([backscaling[
                single_sample_id + i_loc*n_samples_per_loc]])
            profile_ids = np.array([matching_cluster[0]+1])

        used_profiles = list(np.unique(profile_ids))
        p_cluster = np.ma.array(np.empty(profile_ids.shape))
        x_opt_cluster = np.empty((len(profile_ids), 5))

        for profile_id in used_profiles:
            v_data = backscaling[profile_ids == profile_id]
            power_curve = self.read_curve(i_profile=profile_id)
            v_curve = np.array(power_curve['v_100m [m/s]'])

            v_bounds = (np.min(v_curve), np.max(v_curve))
            v_out = np.logical_or(v_data < v_bounds[0],
                                  v_data > v_bounds[1])

            # Match sample wind speed to optimization output
            v_bin_idxs = np.argmin(np.abs(v_curve[np.newaxis, :]
                                          - v_data[:, np.newaxis]), axis=1)
            used_idxs = np.unique(v_bin_idxs)
            x_opts = np.empty((len(v_bin_idxs), 5))
            for v_bin_idx in used_idxs:
                x_opt_i = np.array(
                    [power_curve['F_out [N]'][v_bin_idx],
                     power_curve['F_in [N]'][v_bin_idx],
                     power_curve['theta_out [rad]'][v_bin_idx],
                     power_curve['dl_tether [m]'][v_bin_idx],
                     power_curve['l0_tether [m]'][v_bin_idx]])

                x_opts[v_bin_idxs == v_bin_idx, :] = x_opt_i[np.newaxis, :]
            x_opt_cluster[profile_ids == profile_id, :] = x_opts

            # Interpolate power via wind speeds:
            p_i = np.interp(v_data,
                            v_curve,
                            power_curve['P [W]'])
            p_i = np.ma.array(p_i, mask=v_out)
            p_cluster[profile_ids == profile_id] = p_i

        if return_backscaling:
            return p_cluster, x_opt_cluster, matching_cluster, n_samples_per_loc, backscaling
        else:
            return p_cluster, x_opt_cluster, matching_cluster, n_samples_per_loc

    def clustering_nominal_power(self,
                                 per_power_curve=False):
        p_n = []
        for profile_id in range(1, self.config.Clustering.n_clusters + 1):
            power_curve = self.read_curve(i_profile=profile_id)
            p_n.append(np.max(power_curve['P [W]']))

        if per_power_curve:
            return p_n
        else:
            return max(p_n)

    def plot_location_map(self, plot_ref_locs=True):
        data = np.ones((len(self.config.Data.locations)))
        if plot_ref_locs:
            from .location_selection import reference_locs
            for ref_loc in reference_locs:
                try:
                    data[self.config.Data.locations.index(ref_loc)] = 2
                except ValueError:
                    pass
        # TODO add ref locs 1k/ 5k? -> make combined location set
                # TODO make color_map optional, now manually set in discrete plot coolwarm
        plot_map(self.config,
                 data,
                 title='{} Locations'.format(self.config.Data.n_locs),
                 # log_scale=False,
                 # plot_continuous=False,
                 # line_levels=[2, 5, 15, 20],
                 # fill_range=None,
                 output_file_name=
                 self.config.IO.plot_output_data.format(
                     title=('location_map')),
                 plot_colorbar=False)
        print('Location map plotted.')
#    def evaluate(config):
#        print('To be done :D')
#        # include inheritance from eval?
#        #aep_map(config)
#        # !!! then: eval
#        # - validation
#        # - result plots
#        # TODO make dependent on config what eval to do -> eval class

