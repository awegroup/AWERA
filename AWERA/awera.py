from .wind_profile_clustering.clustering import Clustering
from .power_production.power_production import PowerProduction
from .power_production.aep_map import evaluate_aep
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
    def aep(self):
        return evaluate_aep(self.config)

    def evaluate_power_curve(self,
                             power_curves=[None, None]):
        # Read frequency
        n_clusters = self.config.Clustering.n_clusters
        freq_full, wind_speed_bin_limits = self.read_frequency()

        loc_aep = []
        loc_aep_sq = []
        p_n = []
        for i_loc, loc in enumerate(self.config.Data.locations):
            # Select location data
            freq = freq_full[i_loc, :, :]

            p_bins = np.zeros(freq.shape)

            for i in range(n_clusters):
                i_profile = i + 1
                # Read power curve file
                # TODO make optional trianing / normal
                if None in power_curves:
                    pc = self.read_curve(i_profile=i_profile)
                    v = pc['v_100m [m/s]'].values  # .values[~mask_faulty_point]
                    p = pc['P [W]'].values  # .values[~mask_faulty_point]
                else:
                    if type(power_curves[0]) == list:
                        v, p = power_curves[i]
                    else:
                        v, p = power_curves
                if i_loc == 0:
                    # Once extract nominal (maximal) power of cluster
                    p_n.append(np.max(p))

                    # assert v[0] == wind_speed_bin_limits[i, 0]
                    # TODO differences at 10th decimal threw assertion error
                    err_str = "Wind speed range of power curve {} is different"\
                        " than that of probability distribution: " \
                        "{:.2f} and {:.2f} m/s, respectively."
                    if np.abs(v[0] - wind_speed_bin_limits[i, 0]) > 1e-6:
                        print(err_str.format(i_profile,
                                             wind_speed_bin_limits[i, 0], v[0]))
                    if np.abs(v[-1] - wind_speed_bin_limits[i, -1]) > 1e-6:
                        print(err_str.format(i_profile,
                                             wind_speed_bin_limits[i, -1], v[-1]))
                    # assert np.abs(v[-1] -
                    #      wind_speed_bin_limits[i, -1]) < 1e-6, err_str

                # Determine wind speeds at bin centers and respective power output.
                v_bins = (wind_speed_bin_limits[i, :-1]
                          + wind_speed_bin_limits[i, 1:])/2.
                p_bins[i, :] = np.interp(v_bins, v, p, left=0., right=0.)

            # Weight profile energy production with the frequency of the cluster
            # sum(freq) < 100: non-operation times included
            aep_bins = p_bins * freq/100. * 24*365
            aep_sum = np.sum(aep_bins)*1e-6
            loc_aep.append(aep_sum)

            aep_bins_sq = (p_bins * freq/100. * 24*365) ** 2
            aep_sq_sum = np.sqrt(np.sum(aep_bins_sq))*1e-6
            loc_aep_sq.append(aep_sq_sum)
            if i_loc % 100 == 0:
                print("AEP: {:.2f} MWh".format(aep_sum),
                      'AEP squared in sum {:.2f} MWh, {:.2f}%'.format(
                          aep_sq_sum, aep_sq_sum/aep_sum*100))
                # plot_aep_matrix(config, freq, p_bins, aep_bins,
                #                 plot_info=(config.Data.data_info+str(i_loc)))
                # print(('AEP matrix plotted for location number'
                #       ' {} of {} - at lat {}, lon {}').format(i_loc,
                #                                               config.Data.n_locs,
                #                                               loc[0], loc[1]))
        # Estimate perfectly running & perfect conditions: nominal power
        # get relative cluster frequency:
        rel_cluster_freq = np.sum(freq_full, axis=(0, 2))/self.config.Data.n_locs
        print('Cluster frequency:', rel_cluster_freq)
        print('Cluster frequency sum:', np.sum(rel_cluster_freq))
        # Scale up to 1: run full time with same relative impact of clusters
        rel_cluster_freq = rel_cluster_freq/np.sum(rel_cluster_freq)
        aep_n_cluster = np.sum(np.array(p_n)*rel_cluster_freq)*24*365*1e-6
        aep_n_max = np.max(p_n)*24*365*1e-6
        print('Nominal aep [MWh]:', aep_n_cluster, aep_n_max)
        return {'loc-wise AEP [MWh]': loc_aep,
                'nominal AEP [MWh]': aep_n_max,
                'p_n [kW]': p_n,
                'loc-wise AEP-sq [MWh]': loc_aep_sq,
                }

    def create_cluster_environments(self, input_profiles=None):
        if input_profiles is None:
            input_profiles = pd.read_csv(self.config.IO.profiles, sep=";")
        # 1 height column, 3 columns each profile (u,v,scale factor)
        # TODO remove scale factor?
        n_profiles = int((input_profiles.shape[1]-1)/3)
        # TODO option to read arbitrary profile, n_prifles: len(profiles)
        envs = []
        for i_profile in range(1, n_profiles+1):
            print('Estimating wind speed for profile {}/{}'
                  .format(i_profile, n_profiles))
            heights = input_profiles['height [m]']
            normalised_wind_speeds_u = input_profiles['u{} [-]'.format(i_profile)]
            normalised_wind_speeds_v = input_profiles['v{} [-]'.format(i_profile)]

            # TODO logging? / timing info print('Profile {}'.format(i_profile))
            env = self.create_environment(normalised_wind_speeds_u,
                                          normalised_wind_speeds_v,
                                          heights)
            envs.append(env)
        return envs


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

