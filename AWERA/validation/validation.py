import os
import copy
import pickle
import numpy as np
import numpy.ma as ma
from multiprocessing import Pool
from ..wind_profile_clustering.clustering import Clustering
from ..wind_profile_clustering.read_requested_data import get_wind_data
from ..wind_profile_clustering.preprocess_data import reduce_wind_data
from ..wind_profile_clustering.utils_validation import plot_diff_pdf, \
    diffs_original_vs_reco

from ..awera import ChainAWERA

class ValidationProcessingPowerProduction(ChainAWERA):
    def __init__(self, config):
        super().__init__(config)
        # Read configuration from config_validation.yaml file
        self.config.update_from_file(config_file='config_validation.yaml')

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
                    return_optimizer=False)
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
            if not np.all(x_opt_sample == -1):
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
                    'cluster_label': cluster_info[0, :],
                    'backscaling': cluster_info[1, :],
                    'locs': [loc],
                    'sample_ids': sel_sample_ids,
                    }
                with open(self.config.IO.sample_vs_cluster_power
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
        res_sample = {
            'p_sample': power[0, :, :],
            'x_opt_sample': x_opt[0, :, :, :],
            'cluster_label': cluster_info[0, :, :],
            'backscaling': cluster_info[1, :, :],
            'locs': locs,
            'sample_ids': sel_sample_ids,
            }
        return_res = res_sample
        if self.config.Power.compare_sample_vs_clustering:
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
                with open(self.config.IO.sample_vs_cluster_power
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
                loc_tag = '{:.2f}_lat_{:.2f}_lon'.format(loc[0], loc[1])

                if self.config.Power.save_sample_only_results:
                    with open(self.config.IO.sample_vs_cluster_power
                              .format(loc=loc_tag), 'wb') as f:
                        pickle.dump(sel_loc(res_sample, i_loc, loc), f)

                if self.config.Power.compare_sample_vs_clustering:
                    with open(self.config.IO.sample_vs_cluster_power
                              .format(loc=loc_tag), 'wb') as f:
                        pickle.dump(sel_loc(res_sample_vs_cluster, i_loc, loc),
                                    f)
            # TODO write location wise output if parallel processing

        return return_res


class ValidationProcessingClustering(Clustering):
    def __init__(self, config):
        super().__init__(config)
        # Read configuration from config_validation.yaml file
        self.config.update_from_file(config_file='config_validation.yaml')

    def clustering_validation():
        #TODO include clustering validation processing
        # rn custering, write results to files in driectory validation
        # range of n pcs, n_clusters, ....?

        # what about different training and data? -> prediction validation
        # 5000 locs training, DIFFERENT 5000locs predicted
        #     -> difference in wind profiles, wind speed bins... always on data
        pass


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



