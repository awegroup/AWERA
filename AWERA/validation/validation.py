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
from ..validation.utils_validation import diffs_original_vs_reco
from ..utils.plotting_utils import plot_diff_pdf

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

        w = (wind_u**2 + wind_v**2)**.5
        # Interpolate normalised wind speed at reference height 100m
        backscaling = np.interp(self.config.General.ref_height, heights, w)

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
            p_cluster, x_opt_cluster, cluster_label, _ = \
                self.match_clustering_power_results(
                    i_loc=i_loc,
                    single_sample_id=single_sample_id)
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
                    x0=[4100., 850., 0.5, 240., 200.0],
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
            cluster_info = [cluster_label, backscaling]
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
            return p_sample, x_opt_sample, backscaling

    def run_location(self, loc, sel_sample_ids=None):
        if sel_sample_ids is None:
            sel_sample_ids = self.config.Validation_Data.sample_ids

        if self.config.Power.compare_sample_vs_clustering:
            # p_sample, p_cluster, p_cc, p_cc_opt
            power = np.zeros([4, len(sel_sample_ids)])
            # x_opt_sample, x_opt_cluster, x_opt_cc_opt
            x_opt = np.zeros([3, len(sel_sample_ids), 5])
            # cluster_id, backscaling
            cluster_info = np.zeros([2, len(sel_sample_ids)])
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
        if not self.config.Power.write_location_wise:
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
                    'cluster_label': cluster_info[0, :],
                    'backscaling': cluster_info[1, :],
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
                    'backscaling': cluster_info[1, :],
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
            cluster_info = np.zeros([2, len(locs), len(sel_sample_ids)])
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
                'backscaling': cluster_info[0, :, :],
                'locs': locs,
                'sample_ids': sel_sample_ids,
                }
            return_res = res_sample
        else:
            res_sample = {
                'p_sample': power[0, :, :],
                'x_opt_sample': x_opt[0, :, :, :],
                'backscaling': cluster_info[1, :, :],
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
                'cluster_label': cluster_info[0, :, :],
                'backscaling': cluster_info[1, :, :],
                'locs': locs,
                'sample_ids': sel_sample_ids,
                }
            return_res = res_sample_vs_cluster

        if not self.config.Power.write_location_wise:
            # Pickle results
            if self.config.Power.save_sample_only_results:
                with open(self.config.IO.sample_power
                          .format(loc='mult_loc_results'), 'wb') as f:
                    pickle.dump(res_sample, f)

            if self.config.Power.compare_sample_vs_clustering:
                with open(self.config.IO.sample_vs_cluster_power
                          .format(loc='mult_loc_results'), 'wb') as f:
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

    def power_curve_spread(self, ref='p_cc_opt'):
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

        # Evaluate for each cluster and velocity bin for mean, std, min, max
        sample_res = np.zeros([self.config.Clustering.n_clusters,
                               self.config.Clustering.n_wind_speed_bins,
                               4])

        # Read single sample results
        for loc in self.config.Data.locations:
            if loc == self.config.Data.locations[0]:
                continue
            loc_tag = self.loc_tag.format(loc[0], loc[1])

            with open(self.config.IO.sample_vs_cluster_power
                      .format(loc=loc_tag), 'rb') as f:
                res = pickle.load(f)

            # TODO read 'wrong' sample output and read labels individually
            #ref = 'p_cluster'
            p_loc = res[ref]
            label_loc = res['cluster_label']

            print('-------------', loc, ref)
            print(p_loc[np.logical_and(p_loc != -1, label_loc == 0)])
            print('---')

            v_loc = res['backscaling']
            print(v_loc[np.logical_and(p_loc != -1, label_loc == 0)])
            print('---')


            for i_c in [0]:  # range(self.config.Clustering.n_clusters):
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
                    print((v0, v1), p_i_cluster)
                    print('---------------------------------------------')
                    # Mean, standard deviation, min, max
                    p_avg = np.mean(p_i_cluster)
                    p_std = np.std(p_i_cluster)
                    p_min = np.min(p_i_cluster)
                    p_max = np.max(p_i_cluster)

                    if i_v == i_start:
                        p_avg = np.mean((sample_res[i_c, i_v, 0], p_avg))
                        # TODO how to combine std correctly?
                        # is there a good way or would I need to
                        # save up the data?: but 2 locs basically indep...?
                        p_std = np.sqrt((sample_res[i_c, i_v, 1]**2
                                         + p_std**2)/2)
                        p_min = np.min((sample_res[i_c, i_v, 2], p_min))
                        p_max = np.max((sample_res[i_c, i_v, 3], p_max))
                    sample_res[i_c, i_v, :] = (p_avg, p_std, p_min, p_max)

        # Plot power curves
        for i_c in [0]:  # range(self.config.Clusterinng.n_clusters):
            self.plot_power_curves(pcs=[(np.array(v_curves[i_c]),
                                         np.array(p_curves[i_c]))],
                                   labels=str(i_c+1),
                                   lim=[v_bins[i_c][0], v_bins[i_c][-1]])
            # Add single production info to plot
            plt.plot(v_bins[i_c][:-1], sample_res[i_c, :, 0]/1000,  '.',
                     label='s-mean')
            # Plot transparent std band around mean
            # plt.fill_between(
            #     v_bins[i_c][:-1],
            #     (sample_res[i_c, :, 0]-sample_res[i_c, :, 1])/1000,
            #     (sample_res[i_c, :, 0]+sample_res[i_c, :, 1])/1000,
            #     alpha=0.5, label='s-std')
            plt.plot(v_bins[i_c][:-1], sample_res[i_c, :, 2]/1000, '.',
                     label='s-min')
            plt.plot(v_bins[i_c][:-1], sample_res[i_c, :, 3]/1000, '.',
                     label='s-max')
            plt.legend()
            # Save plot
            title = 'power_curve_spread_vs_{}_profile_{}'.format(ref, i_c+1)
            plt.savefig(self.config.IO.plot_output.format(title=title))


class ValidationProcessingClustering(Clustering):
    def __init__(self, config):
        super().__init__(config)
        # Read configuration from config_validation.yaml file
        self.config.update_from_file(config_file='config_validation.yaml')

    def eval_velocities(self,
                        mean_diffs, full_diffs,
                        heights, wind_speeds,
                        plot_output_file_name='vel_diff_pdf_{vel_tag}.pdf'):
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
                }
        vel_res = {}
        for eval_wind_type in self.config.Clustering.wind_type_eval:
            vel_res[eval_wind_type] = copy.deepcopy(vel_res_dict)
            # Define heights to be plotted - no heights given, evaluate all
            if self.config.Clustering.eval_heights == []:
                eval_heights = heights
            else:
                eval_heights = self.config.Clustering.eval_heights
            for wind_orientation in mean_diffs:
                if wind_orientation not in \
                        self.config.Clustering.wind_type_eval:
                    continue
                for diff_type, val in mean_diffs[wind_orientation].items():
                    diff_vals = full_diffs[wind_orientation][diff_type]
                    mean, std = val
                    # Plot PDFs: Distribution of differences for each height.
                    for height_idx, height in enumerate(heights):
                        if height not in eval_heights:
                            continue
                        # Find mask
                        wind_speed = wind_speeds[:, height_idx]
                        for vel_idx, vel in enumerate(split_velocities):
                            if vel_idx in [len(split_velocities)-2,
                                           len(split_velocities)-1]:
                                vel_mask = wind_speed >= vel
                                vel_tag = '_vel_{}_up'.format(vel)
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

        return vel_res

    def evaluate_diffs(self, original_data, reco_data, altitudes,
                       wind_speed, plot_output_file_name='vel_pdf_{info}.pdf'):
        # Evaluate differences between reconstructed and original data
        mean_diffs, full_diffs = diffs_original_vs_reco(
            original_data,
            reco_data,
            len(altitudes),
            wind_type_eval=self.config.Clustering.wind_type_eval)

        # Velociy bin wise eval
        plot_output_file_name = plot_output_file_name\
            .format(info='vel_diff_pdf_{vel_tag}')
        vel_res = self.eval_velocities(
            mean_diffs, full_diffs,
            altitudes,
            wind_speed,
            plot_output_file_name=plot_output_file_name)
        diff_res = {'diffs': mean_diffs,
                    'vel_diffs': vel_res,
                    }
        return diff_res

    def process_clustering_validation(self,
                                      training_wind_data=None,
                                      testing_wind_data=None,
                                      return_data=False):
        # Evaluate performance of one combination of n_pcs and n_clusters
        if self.config.Clustering.Validation_type.training == 'cut':
            training_remove_low_wind = True
        else:
            training_remove_low_wind = False
        try:
            profiles = self.read_profiles()
            pipeline = self.read_pipeline()
            _, _, _, cluster_mapping = self.read_labels(data_type='training')
            read_training = False
        except FileNotFoundError:
            # Set config data to clustering:
            # Read training data
            config = copy.deepcopy(self.config)
            self.config.update(
                {'Data': self.config.Clustering.training.__dict__})
            if training_wind_data is None:
                training_wind_data = get_wind_data(self.config)
                if 'training_data' not in training_wind_data:
                    # Data not yet preprocessed
                    training_wind_data = self.preprocess_data(
                        training_wind_data,
                        remove_low_wind_samples=training_remove_low_wind)
                read_training = True
            profiles, pipeline, cluster_mapping = self.train_profiles(
                data=training_wind_data,
                training_remove_low_wind_samples=training_remove_low_wind,
                return_pipeline=True)
            # Reset config
            setattr(self, 'config', config)

        # Read testing data
        if self.config.Clustering.Validation_type.testing == 'cut':
            testing_remove_low_wind = True
        else:
            testing_remove_low_wind = False
        if (self.config.Data.data_info ==
                self.config.Clustering.training.data_info) and \
                testing_remove_low_wind == training_remove_low_wind:
            test_is_train = True
        else:
            test_is_train = False
        # TODO check if need to read testing data, or same as training data
        # and training data was read
        if testing_wind_data is None:
            if test_is_train and read_training:
                testing_wind_data = training_wind_data
            else:
                testing_wind_data = get_wind_data(self.config)

        # Predict labels
        try:
            labels, backscaling, n_samples_per_loc, _ = self.read_labels()
        except FileNotFoundError:
            labels, backscaling, n_samples_per_loc = self.predict_labels(
                data=testing_wind_data,
                pipeline=pipeline,
                cluster_mapping=cluster_mapping,
                remove_low_wind_samples=testing_remove_low_wind)

        # Reconstruct data
        if 'training_data' not in testing_wind_data:
            # Data not yet preprocessed
            testing_wind_data = self.preprocess_data(
                        testing_wind_data,
                        remove_low_wind_samples=testing_remove_low_wind)
        n_altitudes = len(testing_wind_data['altitude'])
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

        diff_res = self.evaluate_diffs(
            original_data, reco_data,
            testing_wind_data['altitude'],
            testing_wind_data['wind_speed'],
            plot_output_file_name=self.config.IO
            .cluster_validation_plotting_pdfs)
        # Write results to file
        pickle.dump(diff_res, open(
            self.config.IO.cluster_validation_processing, 'wb'))

        # Return results, optional: return test data
        if return_data:
            return diff_res, training_wind_data, testing_wind_data
        else:
            return diff_res

    def process_pca_validation(self, testing_wind_data):
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

        diff_res = self.evaluate_diffs(
            original_data, reco_data,
            testing_wind_data['altitude'],
            testing_wind_data['wind_speed'],
            plot_output_file_name=self.config.IO.pca_validation_plotting_pdfs)
        # Write results to file
        pickle.dump(diff_res, open(
            self.config.IO.pca_validation_processing, 'wb'))

        return diff_res

    def process_all(self, min_n_pcs=4):
        cluster_diffs = []
        pca_diffs = []
        all_n_pcs = []
        for n_pcs in range(min_n_pcs,
                           self.config.Clustering.eval_n_pc_up_to + 1):
            for n_clusters in self.config.Clustering.eval_n_clusters:
                self.config.update({
                    'Clustering': {
                        'n_clusters': n_clusters,
                        'n_pcs': n_pcs
                        },
                    })
                if n_pcs == min_n_pcs:
                    diff_res, training_wind_data, testing_wind_data = \
                        self.process_clustering_validation(
                            return_data=True)
                else:
                    diff_res = self.process_clustering_validation(
                        training_wind_data=training_wind_data,
                        testing_wind_data=testing_wind_data)
                cluster_diffs.append(diff_res)
            pca_diffs.append(
                self.process_pca_validation(
                    testing_wind_data=testing_wind_data)
                )
            all_n_pcs.append(n_pcs)
        return {'cluster_diffs': cluster_diffs,
                'pca_diffs': pca_diffs,
                'all_n_pcs': all_n_pcs,
                'all_n_clusters': self.config.Clustering.eval_n_clusters,
                }



# class ValidationPlottingClustering(Clustering):
#     def __init__(self, config):
#         super().__init__(config)
#         # Read configuration from config_validation.yaml file
#         self.config.update_from_file(config_file='config_validation.yaml')

#         #                   One set of n_clusters, n_pcs:
#         # Height dependence
#         if n_clusters > 0:
#             plot_height_vs_diffs(heights, wind_orientation, diff_type, n, data_info,
#                                  pc_mean, pc_std,
#                                  cluster_mean=cluster_mean, cluster_std=cluster_std,
#                                  n_clusters=n_clusters)
#         else:
#             plot_height_vs_diffs(heights, wind_orientation, diff_type, n, data_info,
#                                  pc_mean, pc_std)
#         # Velocity Dependence
#         # ---- Plot velocity dependence vs n_pc
#         for wind_orientation in vel_res:
#             for fit_type in vel_res[wind_orientation]:
#                 if fit_type == 'cluster' and n_clusters == 0: continue
#                 elif fit_type == 'pc' and n_clusters > 0: continue
#                 for diff_type in vel_res[wind_orientation][fit_type]:
#                     for height_idx, height in enumerate(eval_heights):
#                         x = np.array(range(len(split_velocities)))

#                         plt.xlabel('velocity ranges in m/s')
#                         if diff_type == 'absolute':
#                             plt.ylabel('{} diff for v {} in m/s'.format(diff_type, wind_orientation))
#                         else:
#                             plt.ylabel('{} diff for v {}'.format(diff_type, wind_orientation))
#                         if n_clusters == 0:
#                             plt.ylim((-0.5,0.5))
#                         else:
#                             plt.ylim((-1.2,1.2))
#                         plot_dict = {}
#                         for pc_idx, n_pcs in enumerate(eval_pcs):
#                             y = vel_res[wind_orientation][fit_type][diff_type][height_idx, pc_idx, :, 0]
#                             dy = vel_res[wind_orientation][fit_type][diff_type][height_idx, pc_idx, :, 1]
#                             if len(eval_pcs) > 1:
#                                 shift = -0.25 + 0.5/(len(eval_pcs)-1) * pc_idx
#                             else:
#                                 shift = 0
#                             shifted_x = x+shift
#                             use_only = y.mask == False
#                             y = y[use_only]
#                             dy = dy[use_only]
#                             shifted_x = shifted_x[use_only]
#                             plot_dict[n_pcs] = plt.errorbar(shifted_x, y, yerr=dy, fmt='+')
#                         ax = plt.axes()  # This triggers a deprecation warning, works either way, ignore
#                         ax.set_xticks(x)
#                         ax.set_xticklabels(['0-1.5','1.5-3', '3-5', '5-10', '10-20', '20 up', 'full'])
#                         legend_list = [plot_item for key, plot_item in plot_dict.items()]
#                         legend_names = ['{} pcs'.format(key) for key in plot_dict]

#                         plt.legend(legend_list, legend_names)
#                         if n_clusters > 0:
#                             plt.title('{} diff of v {} at {} m using {} {}'.format(diff_type, wind_orientation, height, n_clusters, fit_type))
#                             plt.savefig(result_dir_validation + '{}_cluster/{}_wind_{}_{}_{}_diff_vs_velocity_ranges_{}_m'.format(
#                                         n_clusters, wind_orientation, n_clusters, fit_type, diff_type, height) + data_info + '.pdf')
#                         else:
#                             plt.title('{} diff of v {} at {} m using {}'.format(diff_type, wind_orientation, height, fit_type))
#                             plt.savefig(result_dir_validation + 'pc_only/{}_wind_{}_{}_diff_vs_velocity_ranges_{}_m'.format(
#                                         wind_orientation, fit_type, diff_type, height) + data_info + '.pdf')
#                         # Clear plots after saving, otherwise plotted on top of each other
#                         plt.cla()
#                         plt.clf()


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

class ValidationChain(ChainAWERA):
    def __init__(self, config):
        super().__init__(config)
    def aep_vs_n_locs(self,
                      prediction_settings=None,
                      data_settings=None,
                      i_ref=0,
                      set_labels=None,
                      run_missing_prediction=True):
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

        aep_err_vs_n_locs(self.config,
                          prediction_settings=prediction_settings,
                          i_ref=i_ref,
                          set_labels=set_labels,
                          )