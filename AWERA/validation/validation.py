import os
import numpy as np
from ..wind_profile_clustering.clustering import Clustering
from ..wind_profile_clustering.read_requested_data import get_wind_data

class ValidationEval():
    def __init__(self, config):
        setattr(self, 'config', config)
    # TODO add validation config? / update config: validation config addition?


class ValidationProcessingClustering(Clustering):
    def __init__(self, config):
        super().__init__(config)
        config_validation = {}  # TODO read from config_validation.yaml
        self.config.update(config_validation)

    def clustering_validation():
        #TODO include clustering validation processing
        # rn custering, write results to files in driectory validation
        # range of n pcs, n_clusters, ....?

        # what about different training and data? -> prediction validation
        # 5000 locs training, DIFFERENT 5000locs predicted
        #     -> difference in wind profiles, wind speed bins... always on data
        pass

    def clustering_validation_old(self):
        # Evaluate performance of pcs and nclusters
        if not self.config.Plotting.plots_interactive:
            # Check for result directories before analysis
            self.check_result_dirs()

        # Read wind data
        wind_data = get_wind_data(self.config)
        from .preprocess_data import preprocess_data
        wind_data_full = preprocess_data(
            wind_data,
            remove_low_wind_samples=False,
            normalize=self.config.Clustering.do_normalize_data)
        wind_data_cut = preprocess_data(
            wind_data,
            remove_low_wind_samples=True,
            normalize=self.config.Clustering.do_normalize_data)

        # Non-normalized data :
        # imitate data structure with norm factor set to 1
        if not self.config.Clustering.do_normalize_data:
            wind_data_full['normalisation_value'] = np.zeros(
                (wind_data_full['wind_speed'].shape[0])) + 1
            wind_data_cut['normalisation_value'] = np.zeros(
                (wind_data_cut['wind_speed'].shape[0])) + 1
            wind_data_full['training_data'] = np.concatenate(
                (wind_data_full['wind_speed_parallel'],
                 wind_data_full['wind_speed_perpendicular']),
                1)
            wind_data_cut['training_data'] = np.concatenate(
                (wind_data_cut['wind_speed_parallel'],
                 wind_data_cut['wind_speed_perpendicular']),
                1)

        # Choose training and test(original) data combination
        # depending on validation type
        if self.config.Clustering.validation_type == 'full_training_full_test':
            training_data = wind_data_full
            test_data = wind_data_full
        elif self.config.Clustering.validation_type ==\
                'cut_training_full_test':
            training_data = wind_data_cut
            test_data = wind_data_full
        elif self.config.Clustering.validation_type == 'cut_training_cut_test':
            training_data = wind_data_cut
            test_data = wind_data_cut
        # TODO implement reading Data / Clustering data for training
        # and testing independently
        # TODO training data only read if profiles need to be trained,
        # labels need to be predicted etc
        self.processing_validation(test_data)


    def processing_validation(self, test_data):
        # TODO copy function to here, break apart
        # into plotting and processing ->
        # (multiple times run clustering/prediction and comparison?)
        # and utils
        evaluate_pc_analysis(training_data, test_data, data_info, eval_pcs=eval_pcs, eval_heights=eval_heights,
                             eval_clusters=eval_clusters, eval_n_pc_up_to=eval_n_pc_up_to, sel_velocity = [3,20])


    def check_result_dirs(self):
        result_dir_validation = self.config.IO.result_dir \
            + self.config.IO.result_dir_validation
        cluster_result_dirs = [result_dir_validation
                               + '_'.join([str(eval_c), 'cluster'])
                               for eval_c in
                               self.config.Clustering.eval_n_clusters]
        for eval_c in self.config.Clustering.eval_n_clusters:
            cluster_result_dirs.append(
                result_dir_validation + '_'.join([str(eval_c), 'cluster'])
                + '/diff_vel_pdf')
        cluster_result_dirs.append(result_dir_validation + 'pc_only')
        cluster_result_dirs.append(result_dir_validation + 'pc_only'
                                   + '/diff_vel_pdf')
        missing_dirs = False
        for cluster_result_dir in cluster_result_dirs:
            if not os.path.isdir(cluster_result_dir):
                print('MISSING result dir:', cluster_result_dir)
                missing_dirs = True
                if self.config.General.make_result_subdirs:
                    os.mkdir(cluster_result_dir)
        if missing_dirs and not self.config.General.make_result_subdirs:
            raise OSError('Missing result dirsectories for plots, '
                          'add dirs and rerun, generate result dirs'
                          ' by setting make_result_subdirs to True')

class ValidationProcessingProduction():
    def __init__(self, config):
        setattr(self, 'config', config)
    # TODO add validation config? / update config with validation config addition?

    def production_validation():
        # run single sample production on test samples, 5k test locations
        # write to file indicating location
        # -> all samples of location in same file: single_sample_production dir.

        # TODO include production validation processing
        # same region, different n_culsters/n_pcs in custering
        # - read different profiles, production, compare to single sample results
        #     write results to files in driectory validation
        # range of n pcs, n_clusters, ....?
        pass
