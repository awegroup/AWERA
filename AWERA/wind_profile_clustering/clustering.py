from .read_requested_data import get_wind_data
from .preprocess_data import preprocess_data
from .wind_profile_clustering import cluster_normalized_wind_profiles_pca, \
    export_wind_profile_shapes, \
    predict_cluster, single_location_prediction
# TODO import all functions needed
#TODO move functions to utils / wind profile clustering

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

    def train_profiles(self):
        # Set Data to read to training data
        config_training_data = copy.deepcopy(self.config)
        config_training_data.update(
            {'Data': self.config.Clustering.training.__dict__})
        data = get_wind_data(config_training_data)
        processed_data = preprocess_data(config_training_data, data)

        res = cluster_normalized_wind_profiles_pca(
            processed_data['training_data'],
            self.config.Clustering.n_clusters,
            n_pcs=config.Clustering.n_pcs)
        prl, prp = res['clusters_feature']['parallel'], \
            res['clusters_feature']['perpendicular']

        # Free up some memory
        del processed_data
        cluster_wind_profiles = export_wind_profile_shapes(
            data['altitude'],
            prl, prp,
            self.config.IO.cluster_profiles,
            ref_height=self.config.Clustering.preprocessing.ref_vector_height)
        setattr(self, 'profiles', cluster_wind_profiles)

        # Write mapping pipeline to file
        pipeline = res['data_processing_pipeline']
        pickle.dump(pipeline, open(config.IO.cluster_pipeline, 'wb'))
        setattr(self, 'pipeline', clustering_pipeline)
        setattr(self, 'cluster_mapping', res['cluster_mapping'])
        training_data_full = preprocess_data(config_training_data,
                                             data,
                                             remove_low_wind_samples=False)
        # TODO make wirting output optional, make return optional, return profiles!
        predict_cluster_labels(self, data=training_data_full)
        return training_data_full

    def predict_labels(self, data=None):
        # TODO this can also be done step by step for the data
        # or in parallel - fill labels incrementally
        if not hasattr(self, 'pipeline'):
            self.read_pipeline()
        # Sort cluster labels same as mapping from training
        # (largest to smallest cluster):
        if not hasattr(self, 'cluster_mapping'):
            self.read_labels(data_type='training')

        locations = self.config.Data.locations
        n_samples_per_loc = get_wind_data(
            config,
            locs=[locations[0]])['n_samples_per_loc']

        res_labels = np.zeros(len(locations)*n_samples_per_loc)
        res_scale = np.zeros(len(locations)*n_samples_per_loc)

        if data is not None:
            res_labels, frequency_clusters = predict_cluster(
                data['training_data'],
                self.config.Clustering.n_clusters,
                self.pipeline.predict,
                self.cluster_mapping)

            res_scale = np.array([
                np.interp(
                    self.config.Clustering.preprocessing.ref_vector_height,
                    processed_data_full['altitude'],
                    processed_data_full['wind_speed'][i_sample, :])
                for i_sample in range(
                        processed_data_full['wind_speed'].shape[0])])
        elif self.config.Processing.parallel:
            # Unset parallel processing: reading input in single process
            # cannot start new processes for reading input in parallel
            setattr(config.Processing, 'parallel', False)

            # TODO no import here
            # TODO check if parallel ->
            from multiprocessing import get_context
            from tqdm import tqdm
            import functools
            funct = functools.partial(single_location_prediction,
                                      config,
                                      pipeline,
                                      cluster_mapping)
            if config.Processing.progress_out == 'stdout':
                file = sys.stdout
            else:
                file = sys.stderr
            # Start multiprocessing Pool
            # use spawn instead of fork: pipeline can be used by child processes
            # otherwise same key/lock on pipeline object - leading to infinite loop
            with get_context("spawn").Pool(config.Processing.n_cores) as p:
                results = list(tqdm(p.imap(funct, locations),
                                    total=len(locations), file=file))
                # TODO is this more RAM intensive?
                for i, val in enumerate(results):
                    res_labels[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = \
                        val[0]
                    res_scale[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = val[1]
            setattr(config.Processing, 'parallel', True)
        else:
            import functools
            funct = functools.partial(single_location_prediction,
                                      config,
                                      pipeline,
                                      cluster_mapping)
            # TODO add progress bar
            for i, loc in enumerate(locations):
                res_labels[i*n_samples_per_loc:(i+1)*n_samples_per_loc], \
                    res_scale[i*n_samples_per_loc:(i+1)*n_samples_per_loc] = \
                    funct(loc)

        # Write cluster labels to file
        cluster_info_dict = {
            'n clusters': config.Clustering.n_clusters,
            'n samples': len(locations)*n_samples_per_loc,
            'n pcs': config.Clustering.n_pcs,
            'labels [-]': res_labels,
            'cluster_mapping': cluster_mapping,
            'backscaling [m/s]': res_scale,
            'training_data_info': training_labels_file['training_data_info'],
            'locations': locations,
            'n_samples_per_loc': n_samples_per_loc,
            }
        setattr(self, 'labels', cluster_info_dict['labels [-]'])
        setattr(self, 'backscaling', cluster_info_dict['backscaling [m/s]'])
        setattr(self, 'n_samples', cluster_info_dict['n samples'])
        setattr(self, 'n_samples_per_loc',
                cluster_info_dict['n_samples_per_loc'])

        pickle.dump(cluster_info_dict, open(config.IO.cluster_labels, 'wb'))

    def export_profiles_and_probability(self):
        if self.config.Clustering.make_profiles:
            self.train_profiles()
            print('Make profiles done.')

        if self.config.Clustering.predict_labels:
            # TODO check if clustering data and data are same: predicting done in make profiles?
            self.predict_labels()
            print('Predict labels done.')

        if self.config.Clustering.make_freq_distr:
            self.export_frequency_distr()
            print('Frequency distribution done.')


    # Read available output
    def read_profiles(self):

    def read_pipeline(self):
        with open(self.config.IO.cluster_pipeline, 'rb') as f:
            pipeline = pickle.load(f)
        setattr(self, 'pipeline', pipeline)

    def read_labels(self, data_type='Data'):
        if data_type in ['Data', 'data']:
            file_name = self.config.IO.cluster_labels
        elif data_type in ['Training', 'training']:
            file_name = self.config.IO.training_cluster_labels
        with open(file_name, 'rb') as f:
            labels_file = pickle.load(f)
        if data_type in ['Data', 'data']:
            setattr(self, 'labels', labels_file['labels [-]'])
            setattr(self, 'backscaling', labels_file['backscaling [m/s]'])
            setattr(self, 'n_samples', labels_file['n samples'])
            setattr(self, 'n_samples_per_loc',
                    labels_file['n_samples_per_loc'])
        setattr(self, 'cluster_mapping', labels_file['cluster_mapping'])

    #TODO updating function: read pipeline/profiles again if clustering config is changed, read labels again if Data is changed...

