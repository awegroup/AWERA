import pandas as pd
import numpy as np
import copy
import pickle
import sys

from .read_requested_data import get_wind_data
from .preprocess_data import preprocess_data
from .wind_profile_clustering import cluster_normalized_wind_profiles_pca, \
    export_wind_profile_shapes, \
    predict_cluster, single_location_prediction
from .cluster_frequency import \
    location_wise_frequency_distribution
# TODO import all functions needed
#TODO if this works: remove export profiles and probability?

#TODO set pipeline etc as attribute or just return every time? -> more api-like to return

class Clustering:
    #TODO inherit from config... or as is set config object as config item?

    def __init__(self, config, read_input=True):
        # Set configuration from Config class object
        setattr(self, 'config', config)
        if read_input:
            if not self.config.Clustering.make_profiles:
                self.read_profiles()
            if not self.config.Clustering.predict_labels:
                self.read_labels()

# --------------------------- Full Clustering Procedure

    def train_profiles(self, return_pipeline=False):
        # Set Data to read to training data
        config = copy.deepcopy(self.config)
        self.config.update(
            {'Data': self.config.Clustering.training.__dict__})
        data = get_wind_data(self.config)
        processed_data = preprocess_data(self.config, data)

        res = cluster_normalized_wind_profiles_pca(
            processed_data['training_data'],
            self.config.Clustering.n_clusters,
            n_pcs=self.config.Clustering.n_pcs)
        prl, prp = res['clusters_feature']['parallel'], \
            res['clusters_feature']['perpendicular']

        # Free up some memory
        del processed_data
        profiles = export_wind_profile_shapes(
            data['altitude'],
            prl, prp,
            self.config.IO.profiles,
            ref_height=self.config.General.ref_height)
        # setattr(self, 'profiles', profiles)

        # Write mapping pipeline to file
        pipeline = res['data_processing_pipeline']
        pickle.dump(pipeline, open(self.config.IO.cluster_pipeline, 'wb'))
        # setattr(self, 'pipeline', pipeline)
        # setattr(self, 'cluster_mapping', res['cluster_mapping'])
        training_data_full = preprocess_data(self.config,
                                             data,
                                             remove_low_wind_samples=False)
        # TODO make wirting output optional?
        self.predict_labels(data=training_data_full,
                            pipeline=pipeline,
                            cluster_mapping=res['cluster_mapping'])
        setattr(self, 'config', config)
        if return_pipeline:
            return profiles, pipeline, res['cluster_mapping']
        else:
            return profiles

    def predict_labels(self,
                       data=None,
                       pipeline=None,
                       cluster_mapping=None):
        # TODO this can also be done step by step for the data
        # or in parallel - fill labels incrementally
        if pipeline is None:
            pipeline = self.read_pipeline()
        # Sort cluster labels same as mapping from training
        # (largest to smallest cluster):
        if cluster_mapping is None:
            _, _, _, cluster_mapping = self.read_labels(data_type='training')

        if data is not None:
            locations = data['locations']
            n_samples_per_loc = data['n_samples_per_loc']
            res_labels, frequency_clusters = predict_cluster(
                data['training_data'],
                self.config.Clustering.n_clusters,
                pipeline.predict,
                cluster_mapping)

            res_scale = np.array([
                np.interp(
                    self.config.General.ref_height,
                    data['altitude'],
                    data['wind_speed'][i_sample, :])
                for i_sample in range(
                        data['wind_speed'].shape[0])])
        else:
            locations = self.config.Data.locations
            n_samples_per_loc = get_wind_data(
                self.config,
                locs=[locations[0]])['n_samples_per_loc']

            res_labels = np.zeros(len(locations)*n_samples_per_loc)
            res_scale = np.zeros(len(locations)*n_samples_per_loc)

            if self.config.Processing.parallel:
                # Unset parallel processing: reading input in single process
                # cannot start new processes for reading input in parallel
                setattr(self.config.Processing, 'parallel', False)

                # TODO no import here
                # TODO check if parallel ->
                from multiprocessing import get_context
                from tqdm import tqdm
                import functools
                funct = functools.partial(single_location_prediction,
                                          self.config,
                                          pipeline,
                                          cluster_mapping)
                if self.config.Processing.progress_out == 'stdout':
                    file = sys.stdout
                else:
                    file = sys.stderr
                # Start multiprocessing Pool
                # use spawn instead of fork:
                # pipeline can be used by child processes
                # otherwise same key/lock on pipeline object
                # - leading to infinite loop
                with get_context("spawn").Pool(
                        self.config.Processing.n_cores) as p:
                    results = list(tqdm(p.imap(funct, locations),
                                        total=len(locations), file=file))
                    # TODO is this more RAM intensive?
                    for i, val in enumerate(results):
                        j = i*n_samples_per_loc
                        res_labels[j:(j+n_samples_per_loc)] = val[0]
                        res_scale[j:(j+n_samples_per_loc)] = val[1]
                setattr(self.config.Processing, 'parallel', True)
            else:
                import functools
                funct = functools.partial(single_location_prediction,
                                          self.config,
                                          pipeline,
                                          cluster_mapping)
                # TODO add progress bar
                for i, loc in enumerate(locations):
                    j = i*n_samples_per_loc
                    res_labels[j:(j+n_samples_per_loc)], \
                        res_scale[j:(j+n_samples_per_loc)] = funct(loc)

        # Write cluster labels to file
        cluster_info_dict = {
            'n clusters': self.config.Clustering.n_clusters,
            'n samples': len(locations)*n_samples_per_loc,
            'n pcs': self.config.Clustering.n_pcs,
            'labels [-]': res_labels,
            'cluster_mapping': cluster_mapping,
            'backscaling [m/s]': res_scale,
            'training_data_info': self.config.Clustering.training.data_info,
            'locations': locations,
            'n_samples_per_loc': n_samples_per_loc,
            }
        pickle.dump(cluster_info_dict,
                    open(self.config.IO.labels, 'wb'))

        return (cluster_info_dict['labels [-]'],
                cluster_info_dict['backscaling [m/s]'],
                cluster_info_dict['n_samples_per_loc'])

    def get_frequency(self,
                      labels=None,
                      backscaling=None,
                      n_samples_per_loc=None):
        # TODO make this also parallel/serial, not all input at same time

        if labels is None:
            labels, backscaling, n_samples_per_loc, _ = \
                self.read_labels(data_type='data')

        freq_distr = location_wise_frequency_distribution(
            self.config,
            self.config.Data.locations,
            labels,
            len(labels),
            n_samples_per_loc,
            backscaling)
        # Frequency and corresponding wind speed bin limits
        return freq_distr['frequency'], freq_distr['wind_speed_bin_limits']


    def run_clustering(self):
        if self.config.Clustering.make_profiles:
            if self.config.Clustering.predict_labels:
                profiles, pipeline, cluster_mapping = self.train_profiles(
                    return_pipeline=True)
                print('Make profiles done.')

                self.predict_labels(
                    pipeline=pipeline,
                    cluster_mapping=cluster_mapping)
                # TODO check if clustering data and data are same: predicting done in make profiles?
            else:
                self.train_profiles()
        elif self.config.Clustering.predict_labels:
            labels, backscaling, n_samples_per_loc = self.predict_labels()
            print('Predict labels done.')
            if self.config.Clustering.make_freq_distr:
                self.get_frequency(
                    labels=labels,
                    backscaling=backscaling,
                    n_samples_per_loc=n_samples_per_loc)

        elif self.config.Clustering.make_freq_distr:
            #TODO need wind speed bins from production here - change this?
            self.get_frequency()
            print('Frequency distribution done.')

    # Read available output
    def read_profiles(self):
        profiles = pd.read_csv(
            self.config.IO.profiles, sep=";")
        return profiles

    def read_pipeline(self):
        with open(self.config.IO.cluster_pipeline, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline

    def read_labels(self, data_type='Data'):
        if data_type in ['Data', 'data']:
            file_name = self.config.IO.labels
        elif data_type in ['Training', 'training']:
            file_name = self.config.IO.training_labels
        with open(file_name, 'rb') as f:
            labels_file = pickle.load(f)
        return (
            labels_file['labels [-]'],
            labels_file['backscaling [m/s]'],
            labels_file['n_samples_per_loc'],
            labels_file['cluster_mapping'])

    def read_frequency(self):
        with open(self.config.IO.freq_distr, 'rb') as f:
            freq_distr = pickle.load(f)
        return freq_distr['frequency'], freq_distr['wind_speed_bin_limits']

