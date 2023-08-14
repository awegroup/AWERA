import os
import copy
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from ..wind_profile_clustering.clustering import Clustering
from ..wind_profile_clustering.read_requested_data import get_wind_data
from ..wind_profile_clustering.preprocess_data import reduce_wind_data
from .utils_validation import diffs_original_vs_reco, plot_height_vs_diffs, \
    diff_original_vs_reco
from ..utils.plotting_utils import plot_diff_pdf, \
    plot_abs_rel_step_wise, plot_diff_step_wise, plot_map

from .aep_vs_n_loc import aep_err_vs_n_locs

from ..awera import ChainAWERA

class ValidationProcessingPowerProduction(ChainAWERA):
    def __init__(self, config):
        super().__init__(config)
        # Read configuration from config_validation.yaml file
        self.config.update_from_file(config_file='config_validation.yaml')
        setattr(self, 'loc_tag', '{:.2f}_lat_{:.2f}_lon')

    def single_sample_power(self,
                            processed_data,
                            single_sample_id,
                            i_loc,
                            compare_sample_vs_clustering=None):
        if compare_sample_vs_clustering is None:
            compare_sample_vs_clustering = \
                self.config.Power.compare_sample_vs_clustering
        heights = processed_data['altitude']
        wind_v = processed_data['training_data'][:, len(heights):][0]
        wind_u = processed_data['training_data'][:, :len(heights)][0]
        # TODO why [0]???
        if not np.all(processed_data['normalisation_value'] == 1):
            # Denormalise wind data
            norm_value = processed_data['normalisation_value'][0]
            wind_v = wind_v * norm_value
            wind_u = wind_u * norm_value

        ref_wind_speed = 1  # No backscaling of non-normalised wind profiles
        # Single sample optimization
        optimizer_results = self.single_profile_power(
            heights, wind_u, wind_v,
            x0=[4100., 850., 0.5, 240., 200.0],
            ref_wind_speed=ref_wind_speed,
            return_optimizer=compare_sample_vs_clustering,
            raise_errors=False)
        if compare_sample_vs_clustering:
            x0_opt_sample, x_opt_sample, op_res_sample, \
                cons_sample, kpis_sample, oc = optimizer_results
            if not np.all([x_i == -1 for x_i in x_opt_sample]):
                # Optimization did not fail
                p_sample = kpis_sample['average_power']['cycle']
            else:
                # Negative power result representing failed simulation
                p_sample = -1

            # Read matching clustering results (cluster)
            p_cluster, x_opt_cluster, cluster_label, _, backscaling = \
                self.match_clustering_power_results(
                    i_loc=i_loc,
                    single_sample_id=single_sample_id,
                    return_backscaling=True)
            p_cluster = p_cluster[0]
            if np.ma.is_masked(p_cluster):
                p_cluster = -1
            x_opt_cluster = x_opt_cluster[0, :]
            cluster_label = cluster_label[0]

            # Test clustering output: evaluate point using cluster control (cc)
            oc.x0_real_scale = x_opt_cluster
            oc.random_x0 = False
            try:
                cons_cc, kpis_cc = oc.eval_point()
                p_cc = kpis_cc['average_power']['cycle']
            except self.prod_errors as e:
                print("Clustering output point evaluation finished with a "
                      "simulation error: {}".format(e))
                p_cc = -1
            # Optimise using the clustering control parameters (cc_opt)
            # [and gaussian smeared] as starting values
            # use run_optimization from oc here
            x0_opt_cc_opt, x_opt_cc_opt, op_res_cc_opt,\
                cons_cc_opt, kpis_cc_opt = self.single_profile_power(
                    heights, wind_u, wind_v,
                    x0=x_opt_cluster,
                    oc=oc,
                    ref_wind_speed=ref_wind_speed,
                    return_optimizer=False,
                    raise_errors=False)
            if not np.all([x_i == -1 for x_i in x_opt_cc_opt]):
                # Optimization did not fail
                p_cc_opt = kpis_cc_opt['average_power']['cycle']
            else:
                # Negative power result representing failed simulation
                p_cc_opt = -1
            power = [p_sample, p_cluster, p_cc, p_cc_opt]
            x_opt = [x_opt_sample, x_opt_cluster, x_opt_cc_opt]
            cluster_info = [processed_data['normalisation_value'],
                            cluster_label,
                            backscaling]
            return power, x_opt, cluster_info

        else:
            x0_opt_sample, x_opt_sample, op_res_sample,\
                cons_sample, kpis_sample = optimizer_results
            if not np.all([x_i == -1 for x_i in x_opt_sample]):
                # Optimization did not fail
                p_sample = kpis_sample['average_power']['cycle']
            else:
                # Negative power result representing failed simulation
                p_sample = -1
            return p_sample, x_opt_sample, processed_data['normalisation_value']

    def run_location(self, loc, sel_sample_ids=None):
        if sel_sample_ids is None:
            sel_sample_ids = self.config.Validation_Data.sample_ids

        if self.config.Power.compare_sample_vs_clustering:
            # p_sample, p_cluster, p_cc, p_cc_opt
            power = np.zeros([4, len(sel_sample_ids)])
            # x_opt_sample, x_opt_cluster, x_opt_cc_opt
            x_opt = np.zeros([3, len(sel_sample_ids), 5])
            # normalisation, cluster_id, backscaling
            cluster_info = np.zeros([3, len(sel_sample_ids)])
        else:
            # p_sample
            power = np.zeros([1, len(sel_sample_ids)])
            # x_opt_sample, x_opt_cluster, x_opt_cc_opt
            x_opt = np.zeros([1, len(sel_sample_ids), 5])
            # backscaling
            cluster_info = np.zeros([1, len(sel_sample_ids)])

        # try reading sample power output and labels? -> give to function
        # Read all selected samples for location
        if len(sel_sample_ids) > 0:
            data = get_wind_data(self.config,
                                 locs=[loc],
                                 sel_sample_ids=sel_sample_ids)
        else:
            # No samples specified, run over all samples
            data = get_wind_data(self.config,
                                 locs=[loc])
            sel_sample_ids = list(range(data['n_samples']))
        # write_timing_info('Input read', time.time() - since)
        # Run preprocessing
        processed_data = self.preprocess_data(data,
                                              remove_low_wind_samples=False)
        # Identify i_loc with clustering i_location to read matching
        # clustering output power / control parameters
        i_location = self.config.Data.locations.index(loc)
        # TODO timing
        # Iterate optimization for all samples individually
        for i_sample, sample_id in enumerate(sel_sample_ids):
            print('{}/{} Processing sample {}...'
                  .format(i_sample+1, len(sel_sample_ids), sample_id))
            mask_keep = processed_data['wind_speed'][:, 0] < -999
            mask_keep[i_sample] = True
            processed_data_sample = reduce_wind_data(processed_data,
                                                     mask_keep,
                                                     return_copy=True)
            power[:, i_sample], x_opt[:, i_sample, :], \
                cluster_info[:, i_sample] = \
                self.single_sample_power(processed_data_sample,
                                         sample_id,
                                         i_location)
        print('Processing location (lat {}, lon {}) done.'
              .format(loc[0], loc[1]))

        # Define result dictionary
        if self.config.Power.write_location_wise:
            loc_tag = '{:.2f}_lat_{:.2f}_lon'.format(loc[0], loc[1])
            # Pickle results
            if self.config.Power.compare_sample_vs_clustering:
                res_sample_vs_cluster = {
                    'p_sample': power[0, :],
                    'p_cluster': power[1, :],
                    'p_cc': power[2, :],
                    'p_cc_opt': power[3, :],
                    'x_opt_sample': x_opt[0, :, :],
                    'x_opt_cluster': x_opt[1, :, :],
                    'x_opt_cc_opt': x_opt[2, :, :],
                    'cluster_label': cluster_info[1, :],
                    'backscaling': cluster_info[2, :],
                    'locs': [loc],
                    'sample_ids': sel_sample_ids,
                    }
                with open(self.config.IO.sample_vs_cluster_power
                          .format(loc=loc_tag), 'wb') as f:
                    pickle.dump(res_sample_vs_cluster, f)
            if self.config.Power.save_sample_only_results:
                res_sample = {
                    'p_sample': power[0, :],
                    'x_opt_sample': x_opt[0, :, :],
                    'normalisation value [m/s]': cluster_info[0, :],
                    'locs': [loc],
                    'sample_ids': sel_sample_ids,
                    }
                with open(self.config.IO.sample_power
                          .format(loc=loc_tag), 'wb') as f:
                    pickle.dump(res_sample, f)
        return power, x_opt, cluster_info

    def run_single_location_sample(self, loc, sample_id):
        # Read selected sample for location
        data = get_wind_data(self.config,
                             locs=[loc],
                             sel_sample_ids=[sample_id])

        # Run preprocessing
        processed_data_sample = self.preprocess_data(
            data,
            remove_low_wind_samples=False)
        # Identify i_loc with clustering i_location to read matching
        # clustering output power / control parameters
        i_location = self.config.Data.locations.index(loc)
        # TODO timing + compare to locaton funct?)
        power, x_opt, cluster_info = \
            self.single_sample_power(
                processed_data_sample,
                sample_id,
                i_location)
        return power, x_opt, cluster_info


    def multiple_locations(self,
                           locs=None,
                           sel_sample_ids=None):
        if sel_sample_ids is None:
            sel_sample_ids = self.config.Validation_Data.sample_ids
        if sel_sample_ids is None:
            raise ValueError('No sample ids selected in function or config.')
        if locs is None:
            locs = self.config.Data.locations

        if self.config.Power.compare_sample_vs_clustering:
            # p_sample, p_cluster, p_cc, p_cc_opt
            power = np.zeros([4, len(locs), len(sel_sample_ids)])
            # x_opt_sample, x_opt_cluster, x_opt_cc_opt
            x_opt = np.zeros([3, len(locs), len(sel_sample_ids), 5])
            # cluster_id, backscaling
            cluster_info = np.zeros([3, len(locs), len(sel_sample_ids)])
        else:
            # p_sample
            power = np.zeros([1, len(locs), len(sel_sample_ids)])
            # x_opt_sample, x_opt_cluster, x_opt_cc_opt
            x_opt = np.zeros([1, len(locs), len(sel_sample_ids), 5])
            # backscaling
            cluster_info = np.zeros([1, len(locs), len(sel_sample_ids)])

        if not self.config.Processing.parallel:
            for i_loc, loc in enumerate(locs):
                power[:, i_loc, :], x_opt[:, i_loc, :, :],\
                    cluster_info[:, i_loc, :] = self.run_location(
                        loc,
                        sel_sample_ids=sel_sample_ids)
        else:
            setattr(self.config.Processing, 'parallel', False)
            # Define mapping all locations and all samples, respectively
            # Same location input for all sample ids,
            # one location after the other
            funct = self.run_single_location_sample
            mapping_iterables = [(loc, sample_id) for loc in locs
                                 for sample_id in sel_sample_ids]
            # TODO tqdm include and same as other mutiprocessing
            # TODO include tqdm in production environment
            # TODO timing info: only for single sample production add up?
            # TODO run for the 5k locations? or 1k locations only?

            with Pool(self.config.Processing.n_cores) as p:
                mapping_out = p.starmap(
                    funct,
                    mapping_iterables)
            # Interpret mapping output
            # ??? or write array during maping at different index?
            # -> parallel access
            n_samples = len(sel_sample_ids)
            for i, res_i in enumerate(mapping_out):
                i_loc = i//n_samples
                i_sample = i % n_samples
                power[:, i_loc, i_sample] = res_i[0]
                x_opt[:, i_loc, i_sample, :] = res_i[1]
                cluster_info[:, i_loc, i_sample] = res_i[2]
            setattr(self.config.Processing, 'parallel', True)

        # Define result dictionary
        if not self.config.Power.compare_sample_vs_clustering:
            res_sample = {
                'p_sample': power[0, :, :],
                'x_opt_sample': x_opt[0, :, :, :],
                'normalisation value [m/s]': cluster_info[0, :, :],
                'locs': locs,
                'sample_ids': sel_sample_ids,
                }
            return_res = res_sample
        else:
            res_sample = {
                'p_sample': power[0, :, :],
                'x_opt_sample': x_opt[0, :, :, :],
                'normalisation value [m/s]': cluster_info[0, :, :],
                'locs': locs,
                'sample_ids': sel_sample_ids,
                }

            res_sample_vs_cluster = {
                'p_sample': power[0, :, :],
                'p_cluster': power[1, :, :],
                'p_cc': power[2, :, :],
                'p_cc_opt': power[3, :, :],
                'x_opt_sample': x_opt[0, :, :, :],
                'x_opt_cluster': x_opt[1, :, :, :],
                'x_opt_cc_opt': x_opt[2, :, :, :],
                'cluster_label': cluster_info[1, :, :],
                'backscaling': cluster_info[2, :, :],
                'locs': locs,
                'sample_ids': sel_sample_ids,
                }
            return_res = res_sample_vs_cluster

        if not self.config.Power.write_location_wise:
            # Pickle results
            if self.config.Power.save_sample_only_results:
                with open(self.config.IO.sample_power
                          .format(loc='mult_loc_results_'
                                  + self.config.Data.data_info), 'wb') as f:
                    pickle.dump(res_sample, f)

            if self.config.Power.compare_sample_vs_clustering:
                with open(self.config.IO.sample_vs_cluster_power
                          .format(loc='mult_loc_results'
                                  + self.config.Data.data_info), 'wb') as f:
                    pickle.dump(res_sample_vs_cluster, f)
        elif self.config.Power.write_location_wise and \
                self.config.Processing.parallel:
            def sel_loc(d, i_loc, loc):
                res = {}
                for key, val in d.items():
                    if key in ['sample_ids']:
                        res[key] = val
                    elif key in ['locs']:
                        res[key] = [loc]
                    elif 'x_opt' in key:
                        res[key] = val[i_loc, :, :]
                    else:
                        res[key] = val[i_loc, :]
                return res

            for i_loc, loc in enumerate(locs):
                loc_tag = self.loc_tag.format(loc[0], loc[1])

                if self.config.Power.save_sample_only_results:
                    with open(self.config.IO.sample_power
                              .format(loc=loc_tag), 'wb') as f:
                        pickle.dump(sel_loc(res_sample, i_loc, loc), f)

                if self.config.Power.compare_sample_vs_clustering:
                    with open(self.config.IO.sample_vs_cluster_power
                              .format(loc=loc_tag), 'wb') as f:
                        pickle.dump(sel_loc(res_sample_vs_cluster, i_loc, loc),
                                    f)
            # TODO write location wise output if parallel processing

        return return_res

    def power_curve_spread(self, ref='p_sample',
                           read_sample_only=True,
                           overwrite=False):
        # Read power curves
        p_curves = []
        v_curves = []
        v_bins = []
        for i_c in range(self.config.Clustering.n_clusters):
            df_profile = self.read_curve(i_profile=i_c+1,
                                         return_constructor=False)
            wind_speeds = list(df_profile['v_100m [m/s]'])
            power = list(df_profile['P [W]'])

            p_curves.append(power)
            v_curves.append(wind_speeds)

            v_bins.append(np.linspace(
                wind_speeds[0],
                wind_speeds[-1],
                self.config.Clustering.n_wind_speed_bins+1))

        # TODO make range optional
        p_bins = np.linspace(-2000, 11000,
                             self.config.Clustering.n_wind_speed_bins + 1)
        p_mids = 0.5*(p_bins[1:] + p_bins[:-1])

        try:
            if overwrite:
                raise FileNotFoundError
            with open(self.config.IO.cluster_validation_plotting.format(
                    title=('power_curve_spread_hist_vs_{}'.format(ref))).replace(
                    '.pdf', '.pickle'), 'rb') as f:
                p_sample_histo = pickle.load(f)
            print('Power diffs read.')
        except FileNotFoundError:

            if read_sample_only:
                labels_full, backscaling, n_samples_per_loc, _ = self.read_labels()
                labels_all_locs = labels_full.reshape(
                    self.config.Data.n_locs, n_samples_per_loc)
                labels_all_locs = labels_all_locs[
                    :, self.config.Validation_Data.sample_ids]
                backscaling_all_locs = backscaling.reshape(
                    self.config.Data.n_locs, n_samples_per_loc)
                backscaling_all_locs = backscaling_all_locs[
                    :, self.config.Validation_Data.sample_ids]
                print(labels_all_locs.shape)

            # Read single sample results
            # Single sample results as power histogram per wind speed bin
            p_sample_histo = np.zeros((self.config.Clustering.n_clusters,
                                       self.config.Clustering.n_wind_speed_bins,
                                       self.config.Clustering.n_wind_speed_bins
                                       ))
            # TODO single sample production with 5k loc tag
            failed_locs = []
            for i_loc, loc in enumerate(self.config.Data.locations):
                if i_loc % 100 == 0:
                    print('Running ', i_loc, loc, '...')
                loc_tag = self.loc_tag.format(loc[0], loc[1])

                if read_sample_only:
                    try:
                        with open(self.config.IO.sample_power
                                  .format(loc=loc_tag), 'rb') as f:
                            res = pickle.load(f)

                    except FileNotFoundError:
                        print('Location not found: ', i_loc, loc)
                        failed_locs.append((i_loc, loc))
                        continue
                    label_loc = labels_all_locs[i_loc]
                    v_loc = backscaling_all_locs[i_loc]
                else:
                    try:
                        with open(self.config.IO.sample_vs_cluster_power
                                  .format(loc=loc_tag), 'rb') as f:
                            res = pickle.load(f)

                    except FileNotFoundError:
                        print('Location not found: ', i_loc, loc)
                        failed_locs.append((i_loc, loc))
                        continue

                    label_loc = res['cluster_label']
                    v_loc = res['backscaling']

                p_loc = res[ref]

                for i_c in range(self.config.Clustering.n_clusters):
                    i_start = 0
                    for i_v, (v0, v1) in enumerate(zip(v_bins[i_c][:-1],
                                                       v_bins[i_c][1:])):
                        v_sel = np.logical_and(
                            np.logical_and(
                                np.logical_and(
                                    (v_loc >= v0),
                                    (v_loc < v1)),
                                (label_loc == i_c)),
                            p_loc != -1)
                        p_i_cluster = p_loc[v_sel]
                        if len(p_i_cluster) == 0:
                            if i_v == i_start:
                                i_start = i_v + 1
                            continue
                        elif np.any(p_i_cluster < p_bins[0]):
                            print(i_loc, loc, 'Power smaller than bin range'
                                  'for wind speed bin ', i_v, (v0, v1))
                            print(p_i_cluster)
                        elif np.any(p_i_cluster > p_bins[-1]):
                            print(i_loc, loc, 'Power larger than bin range'
                                  'for wind speed bin ', i_v, (v0, v1))
                        # Fill power histogram
                        hist, _ = np.histogram(p_i_cluster, bins=p_bins)
                        p_sample_histo[i_c, i_v, :] += hist

            print('Saving power curve spread histograms....')
            pickle.dump(p_sample_histo, open(
                self.config.IO.cluster_validation_plotting.format(
                    title=('power_curve_spread_hist_vs_{}'.format(ref))).replace(
                    '.pdf', '.pickle'), 'wb'))
            if len(failed_locs) > 0:
                print('Failed: ', len(failed_locs), [f[0] for f in failed_locs])

        # Plot power curves
        for i_c in range(self.config.Clustering.n_clusters):
            print('Plotting Cluster ', i_c+1)
            self.plot_power_curves(
                pcs=[(np.array(v_curves[i_c]), np.array(p_curves[i_c]))],
                labels=str(i_c+1),
                lim=[np.min((v_bins[i_c][0], v_curves[i_c][0])),
                     np.max((v_bins[i_c][-1], v_curves[i_c][-1]))],
                save_plot=False)
            # Add single production info to plot
            v_box_data = []
            v_box_error = []
            v_mids = 0.5*(v_bins[i_c][1:] + v_bins[i_c][:-1])
            for i_v in range(self.config.Clustering.n_wind_speed_bins):
                hist_data = p_sample_histo[i_c, i_v, :]
                # Don't consider outliers: 99% center data only:
                # TODO make optional

                if np.sum(hist_data) == 0:
                    print('No samples in velocity bin',
                          ' {} at {}, sum of histogram {}'.format(
                              i_v, v_mids[i_v], np.sum(hist_data)))

                    v_box_data.append(0)
                    v_box_error.append(0)
                    continue
                n_perc = np.sum(hist_data)*0.005

                n_perc_right = copy.deepcopy(n_perc)
                outliers_left = []
                outliers_right = []
                non_zero_hist = np.where(np.array(hist_data) != 0)[0]
                data_min = p_bins[non_zero_hist[0]]
                data_max = p_bins[non_zero_hist[-1]+1]

                # TODO n_power_bins
                for i in range(self.config.Clustering.n_wind_speed_bins):
                    if n_perc != 0:
                        if hist_data[i] > n_perc:
                            hist_data[i] = hist_data[i]-n_perc
                            outliers_left.append((p_mids[i], n_perc))
                            n_perc = 0
                        else:
                            n_perc = n_perc - hist_data[i]
                            outliers_left.append((p_mids[i], hist_data[i]))
                            hist_data[i] = 0
                    if n_perc_right != 0:
                        if hist_data[-i] > n_perc_right:
                            hist_data[-i] = hist_data[-i]-n_perc_right
                            outliers_right.append((p_mids[-i], n_perc_right))
                            n_perc_right = 0
                        else:
                            n_perc_right = n_perc_right - hist_data[-i]
                            outliers_right.append((p_mids[-i], hist_data[-i]))
                            hist_data[-i] = 0
                    if n_perc_right == 0 and n_perc == 0:
                        break
                # Evaluate power histogram
                mean = np.average(p_mids, weights=hist_data)
                v_box_data.append(mean)
                v_box_error.append(np.sqrt(np.average((p_mids-mean)**2,
                                                      weights=hist_data)))
                non_zero_hist = np.where(np.array(hist_data) != 0)[0]
                box_min = p_bins[non_zero_hist[0]]
                box_max = p_bins[non_zero_hist[-1]+1]
                plt.plot((v_mids[i_v], v_mids[i_v]), (data_min/1000,
                                                      data_max/1000),
                         color='darkcyan', alpha=.25)
                # plt.scatter((v_mids[i_v], v_mids[i_v]), (data_min/1000,
                #                                          data_max/1000),
                #             marker='_', color='darkcyan', alpha=.25)

                plt.bar(v_mids[i_v], box_max/1000-box_min/1000,
                        bottom=box_min/1000,
                        width=0.8*(v_bins[i_c][1]-v_bins[i_c][0]),
                        fill=True, facecolor='darkcyan', alpha=.25
                        )

            # Plot histogram mean and errorbars
            plt.errorbar(v_mids, np.array(v_box_data)/1000,
                         yerr=np.array(v_box_error)/1000,
                         alpha=0.75, zorder=1, color='darkcyan',
                         ecolor='darkcyan', label='Sample Mech. Power',
                         fmt='_')
            plt.legend()
            # Save plot
            title = 'power_curve_spread_vs_{}_profile_{}'.format(ref, i_c+1)
            plt.savefig(self.config.IO.plot_output.format(title=title))
            print('Plot output saved. Power curve spread done.')

    def plot_power_diff_maps(self,
                             read_sample_only=True,
                             ref='p_sample',
                             overwrite=False):
        # Read single sample data
        power_diffs = {}
        # Testing:
        # setattr(self.config.Data, 'n_locs', 10)
        # setattr(self.config.Data, 'locations', self.config.Data.locations[:10])
        # TODO why is config not updated in plot maps?

        # Try reading diffs:
        try:
            if overwrite:
                raise FileNotFoundError
            with open(self.config.IO.cluster_validation_plotting.format(
                    title=('diff_sample_cluster_power_')).replace(
                    '.pdf', '.pickle'), 'rb') as f:
                power_diffs = pickle.load(f)
            print('Power diffs read.')
        except FileNotFoundError:
            print('Processing single sample results to get power diffs...')
            single_diff = np.ma.empty((self.config.Data.n_locs,
                                       len(self.config.Validation_Data.sample_ids)))
            for i_loc, loc in enumerate(self.config.Data.locations):
                if i_loc % 100 == 0:
                    print('Running ', i_loc, loc, '...')
                loc_tag = self.loc_tag.format(loc[0], loc[1])

                if read_sample_only:
                    try:
                        with open(self.config.IO.sample_power
                                  .format(loc=loc_tag), 'rb') as f:
                            res = pickle.load(f)

                    except FileNotFoundError:
                        print('Location not found: ', i_loc, loc)
                        continue

                    p_cluster, _, _, n_samples_per_loc, backscaling = \
                        self.match_clustering_power_results(i_loc=i_loc,
                                                            single_sample_id=None,
                                                            return_backscaling=True,
                                                            locs_slice=None)
                    # Select sample ids
                    p_cluster = p_cluster[self.config.Validation_Data.sample_ids]
                    # v_loc = backscaling[self.config.Validation_Data.sample_ids]

                else:
                    with open(self.config.IO.sample_vs_cluster_power
                              .format(loc=loc_tag), 'rb') as f:
                        res = pickle.load(f)

                    # label_loc = res['cluster_label']
                    # v_loc = res['backscaling']
                    p_cluster = res['p_cluster']

                p_loc = res[ref]

                # Mask -1 results
                p_loc = np.ma.array(p_loc, mask=p_loc == -1)

                # Get abs/rel differences
                power_diffs_loc = diff_original_vs_reco(p_loc, p_cluster)
                for diff_type in power_diffs_loc:
                    if i_loc == 0:
                        power_diffs[diff_type] = copy.copy(single_diff)
                    power_diffs[diff_type][i_loc, :] = power_diffs_loc[diff_type]

            print('Saving power diffs....')
            pickle.dump(power_diffs, open(
                self.config.IO.cluster_validation_plotting.format(
                    title=('diff_sample_cluster_power_')).replace(
                    '.pdf', '.pickle'), 'wb'))

        # Plot difference maps
        overflow_dict = {'absolute': [[-200, 1500], 3000],
                         'relative': [[-0.1, 0.5], 1]}
        for diff_type in power_diffs:
            # Plot difference map
            unit = {'absolute': '[W]', 'relative': '[-]'}
            res_type = ['bias', 'std']
            res_label = ['bias {} error {}'.format(diff_type,
                                                   unit[diff_type]),
                         r'$\sigma$ {} error {}'.format(diff_type,
                                                        unit[diff_type])]
            res_title = ['Sample vs Cluster QSM'.format(diff_type),
                         'Sample vs Cluster QSM'
                         .format(diff_type)]
            overflow = overflow_dict[diff_type]
            for i in range(2):
                # Known issue with masked arrays: FloatingPointErrors
                np.seterr(all='raise', under='ignore')
                rt = res_type[i]
                if i == 0:
                    data = np.mean(power_diffs[diff_type], axis=1)
                elif i == 1:
                    data = np.std(power_diffs[diff_type], axis=1)
                np.seterr(all='raise')
                # format (n_locs, (mean, std))
                plot_map(self.config,
                         data,
                         title=res_title[i],
                         label=res_label[i],
                         # log_scale=False,
                         # plot_continuous=False,
                         # line_levels=[2, 5, 15, 20],
                         # fill_range=None,
                         output_file_name=
                         self.config.IO.cluster_validation_plotting.format(
                             title=('diff_map_power_{}_{}_diff'.format(
                                 rt, diff_type))),
                         overflow=overflow[i])
        print('Power diff maps plotted.')

class ValidationProcessingClustering(Clustering):
    def __init__(self, config):

        super().__init__(config)

        setattr(self, 'config', config)
        self.config.update_from_file(config_file='config_validation.yaml')

    def eval_velocities(self,
                        mean_diffs, full_diffs,
                        heights, wind_speeds,
                        plot_output_file_name='vel_diff_pdf_{vel_tag}.pdf',
                        save_full_diffs=None,
                        use_backscaling=False):
        # run for a single set of mean diffs, full diffs
        split_velocities = self.config.Clustering.split_velocities
        vel_res_dict = {
                'relative': ma.array(
                    np.zeros(
                        (len(self.config.Clustering.eval_heights),
                         len(split_velocities), 2)),
                    mask=True),
                'absolute': ma.array(
                    np.zeros(
                        (len(self.config.Clustering.eval_heights),
                         len(split_velocities), 2)),
                    mask=True),
                'vel_bins': split_velocities,
                'eval_heights': self.config.Clustering.eval_heights
                }
        vel_res = {}
        full_diffs_hist = {}
        backscaling_res = {}
        for wind_orientation in self.config.Clustering.wind_type_eval:
            print('Eval velocities {}'.format(wind_orientation))
            vel_res[wind_orientation] = copy.deepcopy(vel_res_dict)
            full_diffs_hist[wind_orientation] = {}
            backscaling_res[wind_orientation] = {}
            # Define heights to be plotted - no heights given, evaluate all
            if self.config.Clustering.eval_heights == []:
                eval_heights = heights
            else:
                eval_heights = self.config.Clustering.eval_heights

            for diff_type, val in mean_diffs[wind_orientation].items():
                full_diffs_hist[wind_orientation][diff_type] = {}
                backscaling_res[wind_orientation][diff_type] = {}
                diff_vals = full_diffs[wind_orientation][diff_type]
                mean, std = val
                # Plot PDFs: Distribution of differences for each height.
                full_diffs_hist[wind_orientation][diff_type]['full'] = {}
                for height_idx, height in enumerate(heights):
                    if height not in eval_heights:
                        continue
                    full_diffs_hist[wind_orientation][diff_type][height] = {}
                    # Find mask
                    if use_backscaling:
                        wind_speed = wind_speeds
                    else:
                        wind_speed = wind_speeds[:, height_idx]
                    for vel_idx, vel in enumerate(split_velocities):
                        if vel_idx in [len(split_velocities)-2,
                                       len(split_velocities)-1]:
                            vel_mask = wind_speed >= vel
                            vel_tag = '_vel_{}_up'.format(vel)
                            if vel == 0:
                                vel = 'full'  # naming in later dicts
                        else:
                            vel_mask = (wind_speed >= vel) * \
                                (wind_speed < split_velocities[vel_idx+1])
                            vel_tag = '_vel_{}_to_{}'.format(
                                vel, split_velocities[vel_idx+1])
                        if np.sum(vel_mask) < 5:
                            print(' '.join(vel_tag.split('_')),
                                  'm/s: less than 5 matches found ',
                                  'SKIPPING')
                            continue

                        if use_backscaling:
                            if vel not in backscaling_res[wind_orientation][
                                    diff_type]:
                                diff_value = diff_vals[
                                    vel_mask, :].reshape((-1))
                                backscaling_res[wind_orientation][
                                    diff_type][vel] = (np.mean(diff_value),
                                                       np.std(diff_value))

                        if save_full_diffs is not None:
                            diff_value = diff_vals[:, height_idx][vel_mask]
                            full_diffs_hist[wind_orientation][
                                diff_type][height][vel] = \
                                np.histogram(diff_value, bins=100)
                            if vel not in full_diffs_hist[wind_orientation][
                                    diff_type]['full']:
                                diff_value = diff_vals[
                                    vel_mask, :].reshape((-1))
                                full_diffs_hist[wind_orientation][
                                    diff_type]['full'][vel] = \
                                    np.histogram(diff_value, bins=100)
                        diff_value = diff_vals[:, height_idx][vel_mask]
                        vel_res[
                            wind_orientation][
                                diff_type][
                                    eval_heights.index(height),
                                    vel_idx, :] = (np.mean(diff_value),
                                                   np.std(diff_value))
                        # TODO make pdf plotting optional
                        if diff_type == 'absolute':
                            parameter = 'v'
                            unit = 'm/s'
                        elif diff_type == 'relative':
                            parameter = 'dv/v'
                            unit = '[-]'
                        o_file = plot_output_file_name.format(
                            vel_tag=vel_tag)
                        plot_diff_pdf(
                            diff_value, diff_type, parameter, unit,
                            output_file_name=o_file,
                            title=' '.join(vel_tag.split('_'))
                            + '{} difference {} wind data at {} m'.format(
                                diff_type, wind_orientation, height),
                            plots_interactive=self.config.Plotting
                            .plots_interactive)
        if save_full_diffs is not None:
            # Write results to file
            pickle.dump(full_diffs_hist, open(
                save_full_diffs.replace(
                    '.pickle', '_vel.pickle'), 'wb'))
        if use_backscaling:
            return {'height_dep': vel_res, 'full': backscaling_res}
        else:
            return vel_res

    def evaluate_diffs(self, original_data, reco_data, altitudes,
                       wind_speed, plot_output_file_name='vel_pdf_{title}.pdf',
                       n_samples_per_loc=10,
                       save_full_diffs=None,
                       backscaling=None,
                       loc_only=False):
        # Evaluate differences between reconstructed and original data
        # TODO this would use much less ram if the processing was done
        # one after another for wind orientations and diff types
        # TODO maybe even use memmap for the full diffs?
        mean_diffs, full_diffs = diffs_original_vs_reco(
            original_data,
            reco_data,
            len(altitudes),
            wind_type_eval=self.config.Clustering.wind_type_eval)

        if not loc_only:
            # Velociy bin wise eval
            plot_output_file_name = plot_output_file_name\
                .format(title='vel_diff_pdf_{vel_tag}')
            vel_res = self.eval_velocities(
                mean_diffs, full_diffs,
                altitudes,
                wind_speed,
                plot_output_file_name=plot_output_file_name,
                save_full_diffs=save_full_diffs)
            if backscaling is not None:
                if save_full_diffs is not None:
                    save_full_diffs_b = save_full_diffs.replace('.pdf',
                                                                '_backscaling.pdf')
                else:
                    save_full_diffs_b = save_full_diffs

                vel_res_b = self.eval_velocities(
                    mean_diffs, full_diffs,
                    altitudes,
                    backscaling,
                    plot_output_file_name=plot_output_file_name
                    .replace('.pdf', '_backscaling.pdf'),
                    save_full_diffs=save_full_diffs_b,
                    use_backscaling=True)

        loc_diffs = {}
        for w_o in full_diffs:
            loc_diffs[w_o] = {}
            for diff_type, diff_val in full_diffs[w_o].items():
                loc_diffs[w_o][diff_type] = {}

                diff_value = diff_val.reshape(
                    (len(self.config.Data.locations),
                     n_samples_per_loc, -1))

                loc_diffs[w_o][
                    diff_type]['full'] = (np.mean(diff_value,
                                                  axis=(1, 2)),
                                          np.std(diff_value,
                                                 axis=(1, 2)))
                if backscaling is not None:
                    v_loc = backscaling.reshape(
                        len(self.config.Data.locations),
                        n_samples_per_loc)
                    # Location wise 3m/s and up
                    mask = v_loc > 3
                    res = np.empty((len(self.config.Data.locations), 2))
                    for i in range(len(self.config.Data.locations)):
                        res[i, :] = (
                            np.mean(diff_value[i, mask[i, :], :]
                                    .reshape((-1))),
                            np.std(diff_value[i, mask[i, :], :]
                                   .reshape((-1))))
                    loc_diffs[w_o][
                        diff_type]['v_greater_3'] = res
                    # Location wise 3m/s and up
                    mask = v_loc > 5
                    res = np.empty((len(self.config.Data.locations), 2))
                    for i in range(len(self.config.Data.locations)):
                        # Locations with no samples above 5 m/s backscaling
                        if np.sum(mask[i, :]) == 0:
                            res[i, :] = [-9999, -9999]
                        else:
                            res[i, :] = (
                                np.mean(diff_value[i, mask[i, :], :]
                                        .reshape((-1))),
                                np.std(diff_value[i, mask[i, :], :]
                                       .reshape((-1))))
                    loc_diffs[w_o][
                        diff_type]['v_greater_5'] = res
        if not loc_only:
            total_mean_diffs = {}
            for w_o in full_diffs:
                total_mean_diffs[w_o] = {}
                for diff_type, diff_val in full_diffs[w_o].items():
                    total_mean_diffs[w_o][diff_type] = {}
                    total_mean_diffs[w_o][diff_type]['full'] = \
                        (np.mean(diff_val), np.std(diff_val))
                    if backscaling is not None:
                        # More than 3 m/s wind speed ref
                        mask = backscaling > 3
                        total_mean_diffs[w_o][diff_type]['v_greater_3'] = \
                            (np.mean(diff_val[mask, :]), np.std(diff_val[mask, :]))

                        # More than 1 m/s wind speed ref
                        mask = backscaling > 1
                        total_mean_diffs[w_o][diff_type]['v_greater_1'] = \
                            (np.mean(diff_val[mask, :]), np.std(diff_val[mask, :]))

        if not loc_only:
            diff_res = {'height_diffs': mean_diffs,
                        'vel_diffs': vel_res,
                        'loc_diffs': loc_diffs,
                        'mean_diffs': total_mean_diffs,
                        }
            if backscaling is not None:
                diff_res['vel_backscaling_diffs'] = vel_res_b
        else:
            diff_res = {'loc_diffs': loc_diffs,
                        }

        if save_full_diffs is not None:
            # Write results to file
            full_diffs_hist = {}
            for w_o in full_diffs:
                full_diffs_hist[w_o] = {}
                for diff_type, diff_val in full_diffs[w_o].items():
                    full_diffs_hist[w_o][diff_type] = {}
                    # TODO make n_bins optional? set custom bin range?
                    for i, height in enumerate(altitudes):
                        full_diffs_hist[w_o][diff_type][height] = \
                            np.histogram(diff_val[:, i], bins=100)

                    diff_value = diff_val.reshape((-1))
                    full_diffs_hist[w_o][
                        diff_type]['full'] = \
                        np.histogram(diff_value, bins=100)

                    # TODO remove, keep ... ?
                    # v_loc = wind_speed.reshape(
                    #     (len(self.config.Data.locations), -1))
                    # n_samples_per_loc = v_loc.shape[1]
                    # diff_value = diff_val.reshape(
                    #     (len(self.config.Data.locations),
                    #      n_samples_per_loc, -1))
                    # # binned = np.empty((diff_value.shape[0], 2, 101))
                    # for i in range(diff_value.shape[0]):
                    #     histo = np.histogram(diff_value[i, :, :]
                    #                          .reshape(-1),
                    #                          bins=100)
                    #     binned[i, 0, :-1] = histo[0]
                    #     binned[i, 1, :] = histo[1]
                    # full_diffs_hist[w_o][
                    #     diff_type]['loc_wise'] = binned
                    # full_diffs_hist[w_o][
                    #     diff_type]['loc_wise'] = (np.mean(diff_value,
                    #                                       axis=(1, 2)),
                    #                               np.std(diff_value,
                    #                                      axis=(1, 2)))

                    # Location wise 3m/s and up
                    # mask = v_loc <= 3
                    # # binned = np.empty((len(self.config.Data.locations),
                    # #                    2, 101))
                    # res = np.empty((len(self.config.Data.locations), 2))
                    # for i in range(len(self.config.Data.locations)):
                    #     # histo = np.histogram(
                    #     #     diff_value[i, mask[i, :], :].reshape((-1)),
                    #     #     bins=100)
                    #     # binned[i, 0, :-1] = histo[0]
                    #     # binned[i, 1, :] = histo[1]
                    #     res[i, :] = (
                    #         np.mean(diff_value[i, mask[i, :], :]
                    #                 .reshape((-1))),
                    #         np.std(diff_value[i, mask[i, :], :]
                    #                .reshape((-1))))
                    # # full_diffs_hist[w_o][
                    # #     diff_type]['loc_wise_greater_3'] = binned
                    # full_diffs_hist[w_o][
                    #     diff_type]['loc_wise_greater_3'] = res
            pickle.dump(full_diffs_hist, open(
                save_full_diffs, 'wb'))
        del full_diffs

        return diff_res

    def process_clustering_validation(self,
                                      training_wind_data=None,
                                      testing_wind_data=None,
                                      return_data=False,
                                      save_full_diffs=False,
                                      loc_only=False,
                                      locs_slice=None):
        # Evaluate performance of one combination of n_pcs and n_clusters
        if self.config.Clustering.Validation_type.training == 'cut':
            training_remove_low_wind = True
        else:
            training_remove_low_wind = False
        if (self.config.Data.data_info ==
                self.config.Clustering.training.data_info):
            test_is_train = True
        else:
            test_is_train = False
        try:
            profiles = self.read_profiles()
            pipeline = self.read_pipeline()
            _, _, _, cluster_mapping = self.read_labels(data_type='training')
            read_training = False
        except FileNotFoundError:
            print('Read data...')
            # Set config data to clustering:
            # Read training data
            if training_wind_data is None:
                print('Train profiles and read training data...')
                profiles, pipeline, cluster_mapping, training_wind_data = \
                    self.train_profiles(
                        data=None,
                        training_remove_low_wind_samples=training_remove_low_wind,
                        return_pipeline=True,
                        return_data=True)

                read_training = True
            else:
                print('Train profiles...')
                profiles, pipeline, cluster_mapping = self.train_profiles(
                    data=training_wind_data,
                    training_remove_low_wind_samples=training_remove_low_wind,
                    return_pipeline=True,
                    return_data=False)

        # Read testing data
        if self.config.Clustering.Validation_type.testing == 'cut':
            testing_remove_low_wind = True
        else:
            testing_remove_low_wind = False

        # TODO check if need to read testing data, or same as training data
        # and training data was read

        locations = self.config.Data.locations
        n_locs = len(locations)
        if locs_slice is not None:
            end = (locs_slice[0]+1)*locs_slice[1]
            if end > n_locs:
                end = n_locs
            locations = locations[locs_slice[0]*locs_slice[1]:end]
            setattr(self.config.Data, 'locations', locations)

        if testing_wind_data is None:
            if test_is_train and read_training:
                testing_wind_data = training_wind_data
            else:
                if locs_slice is not None:
                    testing_wind_data = get_wind_data(
                        self.config,
                        locs=locations)
                    setattr(self.config.Data, 'locations', locations)
                # TODO dont return full wind data if test is not train
                else:
                    print('Read testing data...')
                    testing_wind_data = get_wind_data(self.config)

        # Predict labels
        try:
            labels, backscaling, n_samples_per_loc, _ = self.read_labels(
                locs_slice=locs_slice)
        except FileNotFoundError:
            try:
                labels, backscaling, n_samples_per_loc, _ = self.read_labels(
                    locs_slice=locs_slice)
                if locs_slice is not None:
                    labels = labels[locs_slice[0] * locs_slice[1]
                                    * n_samples_per_loc:end*n_samples_per_loc]
                    backscaling = backscaling[
                        locs_slice[0] * locs_slice[1]
                        * n_samples_per_loc:end*n_samples_per_loc]
            except FileNotFoundError:
                print('Predict testing data labels...')
                if locs_slice is not None:
                    write_output = False
                else:
                    write_output = True
                labels, backscaling, n_samples_per_loc = self.predict_labels(
                    data=testing_wind_data,
                    pipeline=pipeline,
                    cluster_mapping=cluster_mapping,
                    remove_low_wind_samples=testing_remove_low_wind,
                    write_output=write_output)

        # Reconstruct data
        if 'training_data' not in testing_wind_data:
            # Data not yet preprocessed
            testing_wind_data = self.preprocess_data(
                        testing_wind_data,
                        remove_low_wind_samples=testing_remove_low_wind)
        n_altitudes = len(testing_wind_data['altitude'])
        print('Test data: ', testing_wind_data['training_data'].shape)
        reco_data = np.zeros(testing_wind_data['training_data'].shape)
        # Fill reco data array with assigned cluster wind profile
        for i_cluster in range(self.config.Clustering.n_clusters):
            mask_cluster = labels == i_cluster
            reco_data[mask_cluster, :n_altitudes] = \
                np.array(profiles['u{} [-]'.format(i_cluster+1)])[np.newaxis,
                                                                  :]
            reco_data[mask_cluster, n_altitudes:] = \
                np.array(profiles['v{} [-]'.format(i_cluster+1)])[np.newaxis,
                                                                  :]
        # Scale cluster wind profiles to real wind profile
        reco_data = reco_data * backscaling[:, np.newaxis]

        # Scale testing data back to real wind data
        # using 'training_data' format
        original_data = testing_wind_data['training_data']\
            * testing_wind_data['normalisation_value'][:, np.newaxis]
        # TODO save original data not denormalise every time

        if save_full_diffs:
            save_diffs = self.config.IO.cluster_validation_processing.replace(
                    '.pickle', '_full.pickle')
            if locs_slice is not None:
                save_diffs = save_diffs.replace(
                    '.pickle',
                    '{}_n_{}.pickle'.format(locs_slice[0], locs_slice[1]))
        else:
            save_diffs = None
        diff_res = self.evaluate_diffs(
            original_data, reco_data,
            testing_wind_data['altitude'],
            testing_wind_data['wind_speed'],
            plot_output_file_name=self.config.IO
            .cluster_validation_plotting_pdfs,
            save_full_diffs=save_diffs,
            backscaling=backscaling,
            n_samples_per_loc=n_samples_per_loc,
            loc_only=loc_only)
        # Write results to file
        output_file_name = self.config.IO.cluster_validation_processing
        if locs_slice is not None:
            output_file_name = output_file_name.replace(
                '.pickle',
                '{}_n_{}.pickle'.format(locs_slice[0], locs_slice[1]))
        pickle.dump(diff_res, open(
            output_file_name, 'wb'))

        # Return results, optional: return test data
        if return_data:
            return diff_res, training_wind_data, testing_wind_data
        else:
            return diff_res

    def process_pca_validation(self, testing_wind_data, save_full_diffs=False):
        # PCA only:
        # Read pca
        # TODO make pca if no readable
        with open(self.config.IO.pca_pipeline, 'rb') as f:
            pca = pickle.load(f)
        data_pc = pca.transform(testing_wind_data['training_data'])
        data_back_pc = pca.inverse_transform(data_pc)
        reco_data = data_back_pc\
            * testing_wind_data['normalisation_value'][:, np.newaxis]
        # Scale testing data back to real wind data
        # using 'training_data' format
        original_data = testing_wind_data['training_data']\
            * testing_wind_data['normalisation_value'][:, np.newaxis]
        # TODO test original data -> wind speed vs wind speed

        if save_full_diffs:
            save_diffs = self.config.IO.pca_validation_processing.replace(
                    '.pickle', '_full.pickle')
        else:
            save_diffs = None
        diff_res = self.evaluate_diffs(
            original_data, reco_data,
            testing_wind_data['altitude'],
            testing_wind_data['wind_speed'],  # TODO this is not normalisation vallue / backsclaing
            plot_output_file_name=self.config.IO.pca_validation_plotting_pdfs,
            save_full_diffs=save_diffs,
            n_samples_per_loc=testing_wind_data['n_samples_per_loc'])
        # Write results to file
        pickle.dump(diff_res, open(
            self.config.IO.pca_validation_processing, 'wb'))

        return diff_res

    def process_all(self, min_n_pcs=4, save_full_diffs=False,
                    loc_cluster_only=False,
                    locs_slice=None,
                    return_diffs=False):
        if not loc_cluster_only and locs_slice is not None:
            raise ValueError('Cannot process pca results via locs slice yet.')
            # TODO
        cluster_diffs = []
        pca_diffs = []
        all_n_pcs = []
        for n_pcs in range(min_n_pcs,
                           self.config.Clustering.eval_n_pc_up_to + 1):
            print('N PCS: {}'.format(n_pcs))
            for i, n_clusters in \
                    enumerate(self.config.Clustering.eval_n_clusters):
                print('N CLUSTERS: {}'.format(n_clusters))
                self.config.update({
                    'Clustering': {
                        'n_clusters': n_clusters,
                        'n_pcs': n_pcs
                        },
                    })
                if n_pcs == min_n_pcs and i == 0:
                    diff_res, training_wind_data, testing_wind_data = \
                        self.process_clustering_validation(
                            return_data=True,
                            save_full_diffs=save_full_diffs,
                            loc_only=loc_cluster_only,
                            locs_slice=locs_slice)
                else:
                    diff_res = self.process_clustering_validation(
                        training_wind_data=training_wind_data,
                        testing_wind_data=testing_wind_data,
                        save_full_diffs=save_full_diffs,
                        loc_only=loc_cluster_only,
                        locs_slice=locs_slice)
                if return_diffs:
                    cluster_diffs.append(diff_res)
            if not loc_cluster_only:
                pca_diffs.append(
                    self.process_pca_validation(
                        testing_wind_data=testing_wind_data,
                        save_full_diffs=save_full_diffs)
                    )
            if return_diffs:
                all_n_pcs.append(n_pcs)
        res = {'cluster_diffs': cluster_diffs,
               'pca_diffs': pca_diffs,
               'all_n_pcs': all_n_pcs,
               'all_n_clusters': self.config.Clustering.eval_n_clusters,
               }
        if return_diffs:
            return res



class ValidationPlottingClustering(Clustering):
    def __init__(self, config):
        super().__init__(config)

        # Read configuration from config_validation.yaml file
        self.config.update_from_file(config_file='config_validation.yaml')

    def plot_single_height_dependence(self,
                                      wind_orientation, diff_type,
                                      pca_data,
                                      cluster_data=None):
        # Takes data in the for (mean, std) giving the mean/std
        # for all heights, respectively
        #                   One set of n_clusters, n_pcs

        # Height dependence
        heights = self.config.Data.height_range
        if cluster_data is not None:
            plot_height_vs_diffs(self.config, heights, wind_orientation,
                                 diff_type, pca_data[0], pca_data[1],
                                 cluster_mean=cluster_data[0],
                                 cluster_std=cluster_data[1])
        else:
            plot_height_vs_diffs(self.config, heights, wind_orientation,
                                 diff_type, pca_data[0], pca_data[1])

    def plot_single_height_velocity_dependence(self,
                                               wind_orientation,
                                               diff_type,
                                               height,
                                               vel_res,
                                               split_velocities=None,
                                               eval_components=None,
                                               eval_type='PCs',
                                               plotting_cluster=True,
                                               tag=''):
        # run:       for height_idx, height in enumerate(eval_heights):
        # vel res is list of [(v_bins, (mean, std)), ...] for each component
        # eval type 'clusters' 'PCs'
        # Velocity Dependence

        # TODO use plot_diff_step_wise function
        if split_velocities is None:
            split_velocities = self.config.Clustering.split_velocities
        x = np.array(range(len(split_velocities)))

        if eval_components is None:
            eval_components = self.config.Clustering.eval_n_pcs
            eval_type = 'PCs'
        if not isinstance(eval_components, list):
            eval_components = [eval_components]

        fig, ax = plt.subplots(figsize=(5.5, 3.7))
        plt.xlabel('velocity ranges in m/s')
        if diff_type == 'absolute':
            plt.ylabel('{} diff for v {} [m/s]'.format(diff_type,
                                                       wind_orientation))
        else:
            plt.ylabel('{} diff for v {} [-]'.format(diff_type,
                                                     wind_orientation))
        if not plotting_cluster:
            plt.ylim((-0.5, 0.5))
        else:
            plt.ylim((-2.5, 2.5))

        plot_dict = {}
        for idx, n in enumerate(eval_components):
            y = vel_res[idx][:, 0]
            dy = vel_res[idx][:, 1]
            if len(eval_components) > 1:
                shift = -0.25 + 0.5/(len(eval_components)-1) * idx
            else:
                shift = 0
            shifted_x = x+shift
            if np.ma.isMaskedArray(y):
                y = y[~y.mask]
                dy = dy[~y.mask]
                shifted_x = shifted_x[~y.mask]
            plot_dict[n] = plt.errorbar(shifted_x, y, yerr=dy, fmt='_')
        ax.set_xticks(x)

        def get_format(v, i):
            if i > 0 and v == 0:
                tag = 'full'
            elif i < len(split_velocities)-1 \
                    and split_velocities[i+1] == 0:
                tag = '{} up'.format(v)
            else:
                tag = '{}-{}'.format(v, split_velocities[i+1])
            return tag
        ax.set_xticklabels([get_format(v, i) for i, v
                            in enumerate(split_velocities)])
        legend_list = [plot_item for key, plot_item in plot_dict.items()]
        legend_names = ['{} {}'.format(key, eval_type) for key in plot_dict]
        plt.legend(legend_list, legend_names, loc='upper right')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        eval_comp_tag = '_'.join([str(e) for e in [eval_components[0], eval_components[-1]]])
        if height != -1:
            height_tag = 'at {} m'.format(height)
        else:
            height_tag = 'all heights'
        w_o = {'perpendicular': 'prp', 'parallel': 'prl', 'abs':'abs'}
        if not self.config.Plotting.plots_interactive:
            if eval_type == 'clusters':
                plt.title('{} diff of v {} {} using {} PCs'.format(
                    diff_type, w_o[wind_orientation], height_tag,
                    self.config.Clustering.n_pcs))
                # Compare different kinds numbers of clusters
                plt.savefig(self.config.IO.cluster_validation_plotting.format(
                    title=('diff_vs_vel_'
                           + '{}_wind_{}_diff_{}_{}_clusters_{}'.format(
                               w_o[wind_orientation], diff_type,
                               height_tag.replace(' ', '_'),
                               eval_comp_tag, tag))))
            elif eval_type == 'PCs' and plotting_cluster:
                plt.title('{} diff of v {} {} using {} clusters'.format(
                    diff_type, w_o[wind_orientation], height_tag,
                    self.config.Clustering.n_clusters))
                plt.savefig(self.config.IO.cluster_validation_plotting.format(
                    title=('diff_vs_vel_'
                           + '{}_wind_{}_diff_{}_{}_pcs_{}'.format(
                               w_o[wind_orientation], diff_type,
                               height_tag.replace(' ', '_'),
                               eval_comp_tag, tag))))
            else:
                plt.title('{} diff of v {} {} using only PCA'.format(
                    diff_type, w_o[wind_orientation], height_tag,
                    self.config.Clustering.n_clusters))
                plt.savefig(self.config.IO.pca_validation_plotting.format(
                    title=('diff_vs_vel_'
                           + '{}_wind_{}_diff_{}_{}_pcs_{}'.format(
                               w_o[wind_orientation], diff_type,
                               height_tag.replace(' ', '_'),
                               eval_comp_tag, tag))))



#         #                   All n_clusters, all n_pcs:
#    # Plot dependence of mean differences for each height depending on the number of pcs
#     for height_idx, height in enumerate(wind_data['altitude']):
#         if height not in eval_heights:
#             continue

#         for wind_orientation in n_pc_dependence:
#             for diff_type in n_pc_dependence[wind_orientation]:
#                 x = np.array(range(eval_n_pc_up_to-2)) + 3
#                 y_pc = n_pc_dependence[wind_orientation][diff_type][2:, 0, height_idx]
#                 dy_pc = n_pc_dependence[wind_orientation][diff_type][2:, 1, height_idx]

#                 plt.xlabel('# pcs')
#                 if diff_type == 'absolute':
#                     plt.ylabel('{} diff for v {} in m/s'.format(diff_type, wind_orientation))
#                     plt.ylim((-1.5, 1.5))
#                 else:
#                     plt.ylabel('{} diff for v {}'.format(diff_type, wind_orientation))
#                     # plt.ylim((-0.6,0.6))
#                     plt.ylim((-1.5, 1.5))
#                 plt.title('{} diff of v {} at {} m'.format(diff_type, wind_orientation, height))

#                 if len(eval_clusters) == 0:
#                     # Plot detailed number of pcs dependence for only pc analysis
#                     plt.errorbar(x, y_pc, yerr=dy_pc, fmt='+', color='tab:blue')
#                     if wind_orientation in sample_mean_dict:
#                         plt.text(plt.xlim()[1]*0.55, plt.ylim()[1]*0.8, 'mean: {:.2E} +- {:.2E}'.format(
#                             sample_mean_dict[wind_orientation][diff_type][0][height_idx],
#                             sample_mean_dict[wind_orientation][diff_type][1][height_idx]))
#                     plt.text(plt.xlim()[1]*0.55, plt.ylim()[1]*0.7, '#pc=1: {:.2E} +- {:.2E}'.format(
#                         n_pc_dependence[wind_orientation][diff_type][0, 0, height_idx],
#                         n_pc_dependence[wind_orientation][diff_type][0, 1, height_idx]))
#                     plt.text(plt.xlim()[1]*0.55, plt.ylim()[1]*0.6, '#pc=2: {:.2E} +- {:.2E}'.format(
#                         n_pc_dependence[wind_orientation][diff_type][1, 0, height_idx],
#                         n_pc_dependence[wind_orientation][diff_type][1, 1, height_idx]))
#                     plt.savefig(result_dir_validation + 'pc_only/{}_wind_{}_diff_vs_number_of_pcs_{}_m'.format(
#                                 wind_orientation, diff_type, height) + data_info + '.pdf')
#                     # Clear plots after saving, otherwise plotted on top of each other
#                     plt.cla()
#                     plt.clf()
#                 else:
#                     # Plot number of pcs dependence comparing all analyses with number of clusters given in eval_clusters
#                     plot_dict = {}
#                     for n_cluster_idx, n_clusters in enumerate(n_cluster_dependence):
#                         y = n_cluster_dependence[n_clusters][wind_orientation][diff_type][2:, 0, height_idx]
#                         dy = n_cluster_dependence[n_clusters][wind_orientation][diff_type][2:, 1, height_idx]
#                         shift = -0.25 + 0.5/(len(n_cluster_dependence)) * n_cluster_idx
#                         plot_dict[n_clusters] = plt.errorbar(x+shift, y, yerr=dy, fmt='+')

#                     ax = plt.axes()
#                     ax.set_xticks(x)

#                     legend_list = [plot_item for key, plot_item in plot_dict.items()]
#                     legend_names = ['{} clusters'.format(key) for key, plot_item in plot_dict.items()]
#                     pc = plt.errorbar(x+0.25, y_pc, yerr=dy_pc, fmt='+', color='b')
#                     legend_list.insert(0, pc)
#                     legend_names.insert(0, 'pc only')
#                     plt.legend(legend_list, legend_names)
#                     plt.savefig(result_dir_validation + '{}_wind_cluster_{}_diff_vs_number_of_pcs_{}_m'.format(
#                                 wind_orientation, diff_type, height) + data_info + '.pdf')
#                     # Clear plots after saving, otherwise plotted on top of each other
#                     plt.cla()
#                     plt.clf()

    def read_single_validation_result(self, eval_type='cluster',
                                      read_full=False,
                                      file_name=None):
        # eval type: 'cluster' or 'pca'
        # read evaluate diffs resullts
        # TODO add option to read full diffs
        if file_name is None:
            if eval_type == 'cluster':
                file_name = self.config.IO.cluster_validation_processing
            elif eval_type == 'pca':
                file_name = self.config.IO.pca_validation_processing
            else:
                raise ValueError('Eval type unknown. '
                                 'Choose fron "cluster" and "pca"')
            if read_full:
                file_name.replace('.pickle', '_full.pickle')
        # Read diff results
        with open(file_name, 'rb') as f:
            all_diffs = pickle.load(f)
        # 'height diffs': mean height _diffs,
        # 'vel_diffs': vel_res, backscaling
        # loc diffs
        # mean diffs

        return all_diffs

    def read_locs_slice_validation_result(self, locs_slice, eval_type='cluster',
                                          read_full=False,
                                          file_name=None,
                                          sel='loc_diffs'):
        # eval type: 'cluster' or 'pca'
        # read evaluate diffs resullts
        # TODO add option to read full diffs
        if file_name is None:
            if eval_type == 'cluster':
                file_name = self.config.IO.cluster_validation_processing
            elif eval_type == 'pca':
                file_name = self.config.IO.pca_validation_processing
            else:
                raise ValueError('Eval type unknown. '
                                 'Choose fron "cluster" and "pca"')
            if read_full:
                file_name.replace('.pickle', '_full.pickle')

        res = {}
        locations = self.config.Data.locations
        n_locs = len(locations)

        # Read locs slices:
        for i in range(locs_slice[0]):
            end = (i+1)*locs_slice[1]
            if end > n_locs:
                end = n_locs
            start = i*locs_slice[1]

            fn = file_name.replace(
                '.pickle',
                '{}_n_{}.pickle'.format(i, locs_slice[1]))
            # Read diff results
            with open(fn, 'rb') as f:
                slice_diffs = pickle.load(f)[sel]
            for w_o in slice_diffs:
                if i == 0:
                    res[w_o] = {}
                for diff_type in slice_diffs[w_o]:
                    if i == 0:
                        res[w_o][diff_type] = {}
                    for sel_v in slice_diffs[w_o][diff_type]:
                        if i == 0:
                            res[w_o][diff_type][sel_v] = np.empty((n_locs, 2))
                        if sel_v == 'full':
                            # TODO fix this: all same shape!
                            res[w_o][diff_type][sel_v][start:end, 0] = \
                                slice_diffs[w_o][diff_type][sel_v][0]
                            res[w_o][diff_type][sel_v][start:end, 1] = \
                                slice_diffs[w_o][diff_type][sel_v][1]
                        else:
                            res[w_o][diff_type][sel_v][start:end, :] = \
                                slice_diffs[w_o][diff_type][sel_v]
        # 'height diffs': mean height _diffs,
        # 'vel_diffs': vel_res, backscaling
        # loc diffs
        # mean diffs

        return res

    def plot_all_single_loc(self, min_n_pcs=4,
                            plot_height_dependence=True,
                            plot_single_velocity_dependence=True,
                            plot_velocity_dependence=True,
                            n_pcs_final=5,
                            plot_backscaling=False,
                            sel='full'):
        # TODO plot backscaling
        # ['vel_backscaling_diffs']{'height_dep': vel_res, 'full': backscaling_res}
        if plot_backscaling:
            cluster_vel_diffs_full = []
            tag = 'back'
        else:
            tag = 'wind'
        pca_diffs = []
        pca_height_diffs = []
        cluster_vel_diffs = []
        pca_vel_diffs = []
        all_n_pcs = []
        for i, n_clusters in \
                enumerate(self.config.Clustering.eval_n_clusters):
            print('N CLUSTERS: {}'.format(n_clusters))
            self.config.update({
                'Clustering': {
                    'n_clusters': n_clusters,
                    },
                })
            cluster_diffs_pc = []
            cluster_vel_diffs_pc = []
            if plot_backscaling:
                cluster_vel_diffs_full_pc = []
            for i_pcs, n_pcs in \
                    enumerate(range(
                        min_n_pcs,
                        self.config.Clustering.eval_n_pc_up_to + 1)):
                print('N PCS: {}'.format(n_pcs))
                self.config.update({
                        'Clustering': {
                            'n_pcs': n_pcs
                            },
                        })
                if i == 0:
                    single_pca_diffs = self.read_single_validation_result(
                        eval_type='pca')
                    pca_height_diffs.append(single_pca_diffs['height_diffs'])
                    pca_diffs.append(single_pca_diffs['mean_diffs'])
                    pca_vel_diffs.append(single_pca_diffs['vel_diffs'])
                    all_n_pcs.append(n_pcs)

                single_cluster_diffs = self.read_single_validation_result()
                cluster_diffs_pc.append(single_cluster_diffs['mean_diffs'])
                if plot_backscaling:
                    cluster_vel_diffs_pc.append(
                        single_cluster_diffs['vel_backscaling_diffs']
                        ['height_dep'])
                    cluster_vel_diffs_full_pc.append(
                        single_cluster_diffs['vel_backscaling_diffs']
                        [sel])

                else:
                    cluster_vel_diffs_pc.append(
                        single_cluster_diffs['vel_diffs'])

                for w_o in single_cluster_diffs['height_diffs']:
                    for diff_type in single_cluster_diffs['height_diffs'][w_o]:
                        # Plot height dependence of clustering and pc errors
                        if plot_height_dependence:
                            self.plot_single_height_dependence(
                                w_o, diff_type,
                                pca_data=pca_height_diffs[i_pcs][w_o][diff_type],
                                cluster_data=single_cluster_diffs['height_diffs'][
                                    w_o][diff_type])
            if plot_backscaling:
                cluster_vel_diffs_full.append(
                    cluster_vel_diffs_full_pc[all_n_pcs.index(n_pcs_final)])
            cluster_vel_diffs.append(
                cluster_vel_diffs_pc[all_n_pcs.index(n_pcs_final)])
            for w_o in single_cluster_diffs['height_diffs']:
                for diff_type in single_cluster_diffs['height_diffs'][w_o]:
                    if plot_single_velocity_dependence:
                        for i, height in enumerate(
                                self.config.Clustering.eval_heights):
                            self.plot_single_height_velocity_dependence(
                                w_o, diff_type, height,
                                [d[w_o][diff_type][i, :, :]
                                 for d in cluster_vel_diffs_pc],
                                split_velocities=None,
                                eval_components=all_n_pcs,
                                eval_type='PCs',
                                plotting_cluster=True,
                                tag=tag)
                        # TODO add only pca results plotting?
                    if plot_velocity_dependence:
                        # Backscliang vs n_pcs plot
                        if plot_backscaling:
                            diffs = [d[w_o][diff_type]
                                     for d in cluster_vel_diffs_full_pc]
                            d_res = []
                            for d in diffs:
                                v_res = np.empty(
                                    (len(cluster_vel_diffs[0][w_o]['vel_bins']), 2))
                                for i, vel in enumerate(d):
                                    v_res[i, :] = d[vel]
                                d_res.append(v_res)

                            self.plot_single_height_velocity_dependence(
                                w_o, diff_type, -1,
                                d_res,
                                split_velocities=None,
                                eval_components=all_n_pcs,
                                eval_type='PCs',
                                plotting_cluster=True,
                                tag=tag)
                if len(all_n_pcs) > 1:
                    # Plot delta v_wind vs original depending on n_pcs
                    plot_config = {
                        'title':  'Mean error using {} clusters'.format(
                            self.config.Clustering.n_clusters),
                        'x_label': '# PCs',
                        'x_ticks': [str(l) for l in all_n_pcs],
                        'y_lim': [(-1.7, 1.7), (-1.2, 1.2)],
                        'output_file_name':
                            self.config.IO.cluster_validation_plotting.format(
                                title='diff_vs_n_pcs_{}_wind'.format(w_o)),
                        'plots_interactive':
                            self.config.Plotting.plots_interactive,
                        }
                    abs_diff = np.zeros((len(all_n_pcs), 2))
                    abs_diff[:, 0] = [d[w_o]['absolute'][sel][0]
                                      for d in cluster_diffs_pc]
                    abs_diff[:, 1] = [d[w_o]['absolute'][sel][1]
                                      for d in cluster_diffs_pc]

                    rel_diff = np.zeros((len(all_n_pcs), 2))
                    rel_diff[:, 0] = [d[w_o]['relative'][sel][0]
                                      for d in cluster_diffs_pc]
                    rel_diff[:, 1] = [d[w_o]['relative'][sel][1]
                                      for d in cluster_diffs_pc]

                    plot_abs_rel_step_wise(all_n_pcs, abs_diff, rel_diff,
                                           **plot_config)

        print('FINAL N PCS: {}'.format(n_pcs_final))
        self.config.update({
                'Clustering': {
                    'n_pcs': n_pcs_final
                    },
                })
        for w_o in single_cluster_diffs['height_diffs']:
            for diff_type in single_cluster_diffs['height_diffs'][w_o]:
                if plot_single_velocity_dependence:
                    # Plot velocity dependence all clusters
                    for i, height in enumerate(
                            self.config.Clustering.eval_heights):
                        vel_res = [diff[w_o][diff_type][i, :, :]
                                   for diff in cluster_vel_diffs]
                        self.plot_single_height_velocity_dependence(
                            w_o, diff_type, height,
                            vel_res,
                            split_velocities=None,
                            eval_components=self.config.Clustering
                            .eval_n_clusters,
                            eval_type='clusters',
                            plotting_cluster=True,
                            tag=tag)
                if plot_velocity_dependence:
                    # Plot full height range vel diffs for backscaling
                    if plot_backscaling:
                        diffs = [d[w_o][diff_type]
                                 for d in cluster_vel_diffs_full]
                        d_res = []
                        for d in diffs:
                            v_res = np.empty(
                                (len(cluster_vel_diffs[0][w_o]['vel_bins']), 2))

                            for i, vel in enumerate(d):
                                v_res[i, :] = d[vel]
                            d_res.append(v_res)
                            print(d_res)
                        self.plot_single_height_velocity_dependence(
                            w_o, diff_type, -1,
                            d_res,
                            split_velocities=None,
                            eval_components=self.config.Clustering.eval_n_clusters,
                            eval_type='clusters',
                            plotting_cluster=True,
                            tag=tag)

    def plot_cluster_loc_diffs(self,
                               training_locs=[(5, 'europe_ref'),
                                              (500, 'europe'),
                                              (1000, 'europe'),
                                              (5000, 'europe')],
                               data_locs=[(5, 'europe_ref')]*4,
                               sel='full',
                               n_pcs=None):
        # sel = ['full'] ['v_greater_3']['v_greater_1']
        if n_pcs is not None:
            self.config.update({'Clustering': {'n_pcs': n_pcs}})
        if data_locs is None:
            data_locs = training_locs

        cluster_diffs = [np.empty((len(training_locs), 2))
                         for n in
                         self.config.Clustering.eval_n_clusters]
        for i_loc, loc_type in enumerate(training_locs):
            # Update location settings
            print('LOC TYPE TRAINING: ', loc_type,
                  'LOC TYPE DATA: ', data_locs[i_loc])
            self.config.update({
                'Data': {
                    'n_locs': data_locs[i_loc][0],
                    'location_type': data_locs[i_loc][1],
                    },
                'Clustering': {
                    'training': {'n_locs': loc_type[0],
                                 'location_type': loc_type[1],
                                 }
                    },
                })

            for i, n_clusters in \
                    enumerate(self.config.Clustering.eval_n_clusters):
                print('N CLUSTERS: {}'.format(n_clusters))
                self.config.update({
                    'Clustering': {
                        'n_clusters': n_clusters,
                        },
                    })
                single_cluster_mean_diffs = \
                    self.read_single_validation_result()['mean_diffs']

                if i_loc == 0 and i == 0:
                    res_diffs = {}
                for w_o in single_cluster_mean_diffs:
                    if w_o not in res_diffs:
                        res_diffs[w_o] = {}
                    for diff_type in single_cluster_mean_diffs[w_o]:
                        if diff_type not in res_diffs[w_o]:
                            res_diffs[w_o][diff_type] = copy.deepcopy(cluster_diffs)
                        res_diffs[w_o][diff_type][i][i_loc, :] = \
                            single_cluster_mean_diffs[w_o][diff_type][sel]
            print(res_diffs)
        # Plot n_cluster and n_locs dependence
        sel_tag = {'v_greater_1': 'vg1', 'full':'v_full', 'v_greater_3':'vg3',
                   'v_greater_5': 'vg5'}
        for w_o in res_diffs:
            for diff_type in res_diffs[w_o]:
                diff = res_diffs[w_o][diff_type]

                plot_config = {
                    'title':  'Mean {} error '.format(diff_type),
                    'x_label': '# Training Locations',
                    'x_ticks': [str(l[0]) for l in training_locs],
                    'output_file_name':
                        self.config.IO.cluster_validation_plotting.format(
                            title=('diff_vs_n_locs_n_clusters'
                                   + '_{}_wind_{}_diff_{}'.format(w_o, diff_type, sel_tag[sel]))
                            ),
                    'plots_interactive':
                        self.config.Plotting.plots_interactive,
                    }

                plot_diff_step_wise(
                    [l[0] for l in training_locs],
                    diff,
                    eval_components=self.config.Clustering.eval_n_clusters,
                    diff_type=diff_type,
                    eval_type='clusters',
                    **plot_config)


    def plot_cluster_diff_maps(self,
                               eval_type='cluster',
                               sel='full',
                               locs_slice=None):
        # sel: greater_3, greater_1, full
        # eval_type: 'cluster', 'pca'
        if eval_type == 'cluster':
            output_file_name = self.config.IO.cluster_validation_plotting
        elif eval_type == 'pca':
            output_file_name = self.config.IO.pca_validation_plotting

        # Read location diffs from full diffs file
        if locs_slice is not None:
            loc_diffs = self.read_locs_slice_validation_result(
                locs_slice, eval_type='cluster',
                read_full=False,
                file_name=None,
                sel='loc_diffs')
        else:
            loc_diffs = self.read_single_validation_result(
                eval_type=eval_type, read_full=False)['loc_diffs']
        # Plot diff maps abs, rel, mean, std
        for w_o in loc_diffs:
            # TODO make overflow dict into funct... config... ?
            overflow_dict = {'absolute': [None, None],  # [mean, std]
                             'relative': [0.2, 0.8]}
            for diff_type in loc_diffs[w_o]:
                # TODO remove:
                if w_o != 'abs':
                    continue
                print('plotting ', w_o, diff_type)
                loc_diff = loc_diffs[w_o][diff_type][sel]
                # Extract mean and standard deviation
                unit = {'absolute': '[m/s]', 'relative': '[-]'}
                res_type = ['bias', 'std']
                res_label = ['bias {}'.format(unit[diff_type]),
                             r'$\sigma$ {}'.format(unit[diff_type])]
                res_title = ['Clustering: {} error'.format(diff_type),
                             'Clustering: {} error'.format(diff_type)]
                overflow = overflow_dict[diff_type]
                for i in range(2):
                    rt = res_type[i]
                    if sel == 'full' and locs_slice is None:
                        # TODO fix this: all same shape!
                        data = loc_diff[i]
                    else:
                        data = loc_diff[:, i]
                    # format (n_locs, (mean, std))
                    if i == 1 and diff_type == 'relative':
                        # Don't plot line levels, too spotted
                        line_levels = []
                    else:
                        line_levels = None
                    if overflow[i] is not None:
                        # TODO improve test overflow
                        # TODO test lenif overflow[i] == 1:
                        if np.max(data) < overflow[i]:
                            overflow[i] = None
                    plot_map(self.config,
                             data,
                             title='{}'.format(res_title[i]),
                             label=res_label[i],
                             # log_scale=False,
                             # plot_continuous=False,
                             line_levels=line_levels,
                             # fill_range=None,
                             output_file_name=output_file_name.format(
                                 title=('diff_map_{}_{}_{}_wind_{}_diff_{}_wind_speeds'.format(
                                     eval_type, rt, w_o, diff_type, sel))),
                             overflow=overflow[i],
                             n_decimals=2)
                    # TODO plot map --> discrete? or plot_single_map: contour/filled


class ValidationChain(ChainAWERA):
    def __init__(self, config):
        super().__init__(config)
    def aep_vs_n_locs(self,
                      prediction_settings=None,
                      data_settings=None,
                      i_ref=0,
                      set_labels=None,
                      run_missing_prediction=True):
        print('Comparing clustering AEP results for settings: ',
              prediction_settings)
        # TODO get i_ref from comparing prediction settings and data_settings
        if data_settings is not None:
            self.config.update({'Data': data_settings})
        if prediction_settings is not None:
            if not isinstance(prediction_settings, list):
                prediction_settings = [prediction_settings]
        elif prediction_settings is None:
            prediction_settings = [self.config.Clustering.dictify()]

        if run_missing_prediction:
            for settings in prediction_settings:
                self.config.update({'Clustering': settings})
                labels_file = self.config.IO.labels
                freq_file = self.config.IO.freq_distr
                if not os.path.isfile(labels_file):
                    print('Predicting labels: ', settings)
                    self.predict_labels()
                if not os.path.isfile(freq_file):
                    print('Evaluating frequency: ', settings)
                    self.get_frequency()

        # TODO include this functionality here
        aep_err_vs_n_locs(self.config,
                          prediction_settings=prediction_settings,
                          i_ref=i_ref,
                          set_labels=set_labels,
                          )


