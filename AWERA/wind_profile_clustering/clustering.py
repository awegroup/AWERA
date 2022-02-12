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

    def __init__(self, config):
        # Set configuration from Config class object
        super().__init__(config)
        setattr(self, 'config', config)

# --------------------------- Full Clustering Procedure

    def preprocess_data(self,
                        data,
                        config=None,
                        remove_low_wind_samples=True,
                        normalize=None):
        if config is None:
            config = self.config
        if normalize is None:
            try:
                normalize = self.config.Clustering.do_normalize_data
            except AttributeError:
                normalize = True

        # Preprocess data
        return preprocess_data(
            config,
            data,
            remove_low_wind_samples=remove_low_wind_samples,
            normalize=normalize)

    def train_profiles(self,
                       data=None,
                       training_remove_low_wind_samples=False,
                       return_pipeline=False):
        # Set Data to read to training data
        config = copy.deepcopy(self.config)
        self.config.update(
            {'Data': self.config.Clustering.training.__dict__})
        if data is None:
            data = get_wind_data(self.config)

        processed_data = self.preprocess_data(
            data,
            remove_low_wind_samples=training_remove_low_wind_samples)

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
        if self.config.Clustering.save_pca_pipeline:
            pca_pipeline = res['pca']
            pickle.dump(pca_pipeline, open(self.config.IO.pca_pipeline, 'wb'))
        # setattr(self, 'pipeline', pipeline)
        # setattr(self, 'cluster_mapping', res['cluster_mapping'])
        training_data_full = self.preprocess_data(
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
                       cluster_mapping=None,
                       remove_low_wind_samples=False,
                       locs_slice=None):
        # TODO this can also be done step by step for the data
        # or in parallel - fill labels incrementally
        if pipeline is None:
            pipeline = self.read_pipeline()
        # Sort cluster labels same as mapping from training
        # (largest to smallest cluster):
        if cluster_mapping is None:
            _, _, _, cluster_mapping = self.read_labels(data_type='training')

        try:
            normalize = self.config.Clustering.do_normalize_data
        except AttributeError:
            normalize = True

        if data is not None:
            locations = data['locations']
            n_locs = len(locations)
            if locs_slice is not None:
                end = (locs_slice[0]+1)*locs_slice[1]
                if end > n_locs:
                    end = n_locs
                locations = locations[locs_slice[0]*locs_slice[1]:end]
            n_samples_per_loc = data['n_samples_per_loc']
            if 'training_data' not in data:
                # Preprocess data
                data = self.preprocess_data(
                    data,
                    remove_low_wind_samples=remove_low_wind_samples,
                    normalize=normalize)
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
            n_locs = len(locations)
            if locs_slice is not None:
                end = (locs_slice[0]+1)*locs_slice[1]
                if end > n_locs:
                    end = n_locs
                locations = locations[locs_slice[0]*locs_slice[1]:end]
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
                funct = functools.partial(
                    single_location_prediction,
                    self.config,
                    pipeline,
                    cluster_mapping,
                    remove_low_wind_samples=remove_low_wind_samples,
                    normalize=normalize)
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
                funct = functools.partial(
                    single_location_prediction,
                    self.config,
                    pipeline,
                    cluster_mapping,
                    remove_low_wind_samples=remove_low_wind_samples,
                    normalize=normalize)
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
        # TODO include locs_slicing in config
        if locs_slice is None:
            file_name = self.config.IO.labels
        else:
            file_name = self.config.IO.labels.replace(
                '.pickle',
                '{}_n_{}.pickle'.format(locs_slice[0], locs_slice[1]))
        pickle.dump(cluster_info_dict,
                    open(file_name, 'wb'))

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


    def run_clustering(self, return_pipeline=False):
        if self.config.Clustering.make_profiles:
            if self.config.Clustering.predict_labels:
                profiles, pipeline, cluster_mapping = self.train_profiles(
                    return_pipeline=True)
                print('Make profiles done.')

                labels, backscaling, n_samples_per_loc = self.predict_labels(
                    pipeline=pipeline,
                    cluster_mapping=cluster_mapping)
                # TODO check if clustering data and data are same: predicting done in make profiles?
                if return_pipeline:
                    return profiles, pipeline, cluster_mapping, \
                        labels, backscaling, n_samples_per_loc
                else:
                    return profiles, \
                        labels, backscaling, n_samples_per_loc

            else:
                profiles = self.train_profiles(return_pipeline=return_pipeline)
                return profiles

        elif self.config.Clustering.predict_labels:
            labels, backscaling, n_samples_per_loc = self.predict_labels()
            print('Predict labels done.')
            if self.config.Clustering.make_freq_distr:
                freq, wind_speed_bin_limits = self.get_frequency(
                    labels=labels,
                    backscaling=backscaling,
                    n_samples_per_loc=n_samples_per_loc)
                return labels, backscaling, n_samples_per_loc, \
                    freq, wind_speed_bin_limits
            else:
                return labels, backscaling, n_samples_per_loc

        elif self.config.Clustering.make_freq_distr:
            #TODO need wind speed bins from production here - change this?
            freq, wind_speed_bin_limits = self.get_frequency()
            print('Frequency distribution done.')
            return freq, wind_speed_bin_limits

    def plot_cluster_shapes(self):
        from .wind_profile_clustering import plot_wind_profile_shapes
        # TODO this can only plot 8 cluster shapes for now
        df = pd.read_csv(self.config.IO.profiles, sep=";")
        heights = df['height [m]']
        print(heights)
        prl = np.zeros((self.config.Clustering.n_clusters, len(heights)))
        prp = np.zeros((self.config.Clustering.n_clusters, len(heights)))
        for i in range(self.config.Clustering.n_clusters):
            u = df['u{} [-]'.format(i+1)]
            v = df['v{} [-]'.format(i+1)]
            sf = df['scale factor{} [-]'.format(i+1)]
            prl[i, :] = u/sf
            prp[i, :] = v/sf

        plot_wind_profile_shapes(self.config,
                                 heights,
                                 prl, prp,
                                 (prl ** 2 + prp ** 2) ** .5,
                                 plot_info=
                                 self.config.Clustering.training.data_info)

    def visualize_clustering_flow(self,
                                  loc=None,
                                  sample_id=None):
        # Read sample wind data
        if loc is None:
            loc = self.config.Data.locations[0]
        if sample_id is None:
            sample_id = 0
        data = get_wind_data(self.config,
                             sel_sample_ids=[sample_id],
                             locs=[loc])
        # Plot original sample wind profile

        # Plot sample wind profile after preprocessing
        # care: remove low wind samples?

        # Plot sample wind profile, PC profiles and reconstructed profile

        # Plot matched cluster profile (with cluster tag),
        # backscaled to sample profile

    # Read available output
    def read_profiles(self):
        profiles = pd.read_csv(
            self.config.IO.profiles, sep=";")
        return profiles

    def read_pipeline(self):
        with open(self.config.IO.cluster_pipeline, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline

    def read_labels(self, data_type='Data',
                    file_name=None,
                    return_file=False):
        if file_name is None:
            if data_type in ['Data', 'data']:
                file_name = self.config.IO.labels
            elif data_type in ['Training', 'training']:
                file_name = self.config.IO.training_labels

        try:
            with open(file_name, 'rb') as f:
                labels_file = pickle.load(f)
        except FileNotFoundError as e:
            print('Error: Trying to read labels but file not found'
                  ' - run predict_labels first.')
            raise e
        if return_file:
            return labels_file
        else:
            return (
                labels_file['labels [-]'],
                labels_file['backscaling [m/s]'],
                labels_file['n_samples_per_loc'],
                labels_file['cluster_mapping'])

    def combine_labels(self, n_i=23, n_max=1000):
        file_name = self.config.IO.labels.replace(
            '.pickle',
            '{{}}_n_{}.pickle'.format(n_max))
        for i in range(n_i):
            print('{}/{}'.format(i+1, n_i))
            res_i = self.read_labels(file_name=file_name.format(i),
                                     return_file=True)
            # First label is start for res
            if i == 0:
                res = copy.deepcopy(res_i)
            else:
                # Append labels to results, append locations
                for key in ['labels [-]', 'backscaling [m/s]']:
                    res[key] = np.append(res[key], res_i[key])
        file_name = self.config.IO.labels
        pickle.dump(res,
                    open(file_name, 'wb'), protocol=4)
        return res

    def read_frequency(self):
        with open(self.config.IO.freq_distr, 'rb') as f:
            freq_distr = pickle.load(f)
        return freq_distr['frequency'], freq_distr['wind_speed_bin_limits']
