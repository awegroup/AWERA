import pandas as pd
import numpy as np
import copy
import pickle
import sys

import matplotlib.pyplot as plt

from .read_requested_data import get_wind_data
from .preprocess_data import preprocess_data
from .wind_profile_clustering import cluster_normalized_wind_profiles_pca, \
    export_wind_profile_shapes, \
    predict_cluster, single_location_prediction, \
    plot_original_vs_cluster_wind_profile_shapes, \
    projection_plot_of_clusters, visualise_patterns
from .cluster_frequency import \
    location_wise_frequency_distribution
from .plot_location_maps import plot_cluster_freq_maps
from .plot_cluster_frequency import plot_frequency
from .principal_component_analysis import plot_frequency_projection, \
    plot_mean_and_pc_profiles
from sklearn.decomposition import PCA
from itertools import accumulate
# TODO import all functions needed
#TODO if this works: remove export profiles and probability?

#TODO set pipeline etc as attribute or just return every time? -> more api-like to return

class Clustering:
    #TODO inherit from config... or as is set config object as config item?

    def __init__(self, config):
        # Set configuration from Config class object
        try:
            super().__init__(config)
        except TypeError:
            print('Clustering initialised. '
                  'No further super() initialisation with config possible.')
        setattr(self, 'config', config)

# --------------------------- Full Clustering Procedure

    def get_wind_data(self):
        return get_wind_data(self.config)

    def preprocess_data(self,
                        data,
                        config=None,
                        remove_low_wind_samples=True,
                        return_copy=True,
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
            return_copy=return_copy,
            normalize=normalize)

    def train_profiles(self,
                       data=None,
                       training_remove_low_wind_samples=True,
                       return_pipeline=False,
                       return_data=False):
        # Set Data to read to training data
        config = copy.deepcopy(self.config)
        self.config.update(
            {'Data': self.config.Clustering.training.__dict__})
        if data is None:
            data = get_wind_data(self.config)
        print('Initial data shape: ', data['wind_speed_north'].shape)
        processed_data = self.preprocess_data(
            data,
            remove_low_wind_samples=training_remove_low_wind_samples)
        print('Training data shape: ', processed_data['training_data'].shape)
        res = cluster_normalized_wind_profiles_pca(
            processed_data['training_data'],
            self.config.Clustering.n_clusters,
            n_pcs=self.config.Clustering.n_pcs)
        prl, prp = res['clusters_feature']['parallel'], \
            res['clusters_feature']['perpendicular']

        # Free up some memory
        del processed_data
        profiles, scale_factors = export_wind_profile_shapes(
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
            remove_low_wind_samples=False,
            return_copy=False)
        print('Testing data shape: ',
              training_data_full['training_data'].shape)
        print('Data shape: ',
              data['training_data'].shape)
        # TODO make wirting output optional?
        self.predict_labels(data=training_data_full,
                            pipeline=pipeline,
                            cluster_mapping=res['cluster_mapping'],
                            scale_factors=scale_factors)
        setattr(self, 'config', config)
        if return_pipeline and return_data:
            return profiles, pipeline, res['cluster_mapping'], data
        elif return_pipeline:
            return profiles, pipeline, res['cluster_mapping']
        elif return_data:
            return profiles, data
        else:
            return profiles

    def predict_labels(self,
                       data=None,
                       pipeline=None,
                       cluster_mapping=None,
                       remove_low_wind_samples=False,
                       locs_slice=None,
                       scale_factors=[],
                       write_output=True):
        # TODO this can also be done step by step for the data
        # or in parallel - fill labels incrementally
        if pipeline is None:
            pipeline = self.read_pipeline()
        # Sort cluster labels same as mapping from training
        # (largest to smallest cluster):
        if cluster_mapping is None:
            _, _, _, cluster_mapping = self.read_labels(
                data_type='training')
        if scale_factors == []:
            profiles = self.read_profiles()
            scale_factors = []
            for i in range(self.config.Clustering.n_clusters):
                scale_factors.append(
                    profiles['scale factor{} [-]'.format(i+1)][0])
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
            res_scale = data['normalisation_value']
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

        backscaling = self.get_backscaling(res_labels,
                                           res_scale,
                                           scale_factors=scale_factors)
        # Write cluster labels to file
        cluster_info_dict = {
            'n clusters': self.config.Clustering.n_clusters,
            'n samples': len(locations)*n_samples_per_loc,
            'n pcs': self.config.Clustering.n_pcs,
            'labels [-]': res_labels,
            'cluster_mapping': cluster_mapping,
            'backscaling [m/s]': backscaling,
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
        if write_output:
            pickle.dump(cluster_info_dict,
                        open(file_name, 'wb'))

        return (cluster_info_dict['labels [-]'],
                cluster_info_dict['backscaling [m/s]'],
                cluster_info_dict['n_samples_per_loc'])

    def get_backscaling(self, labels, norm, scale_factors=[]):
        if scale_factors == []:
            profiles = self.read_profiles()
            scale_factors = []
            for i in range(self.config.Clustering.n_clusters):
                scale_factors.append(
                    profiles['scale factor{} [-]'.format(i+1)])
            scale_factors = [sf[0] for sf in scale_factors]
        backscaling = np.empty_like(norm)
        for i, sf in enumerate(scale_factors):
            backscaling[labels == i] = norm[labels == i] / sf
        return backscaling

    def get_frequency(self,
                      labels=None,
                      backscaling=None,
                      n_samples_per_loc=None):
        # TODO make this also parallel/serial, not all input at same time
        if labels is None:
            print('Get Frequency. Read labels...')
            labels, backscaling, n_samples_per_loc, _ = \
                self.read_labels(data_type='data')

        print('Get Frequency. Evaluate labels...')
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

    def get_wind_speed_at_height(self, wind_speeds, h, heights=None):
        if heights is None:
            heights = self.config.Data.height_range
        v = np.interp(h, heights,
                      wind_speeds, left=np.nan, right=np.nan)
        return v

    def scale_profile(self, wind_speeds, v_ref, h_ref, heights=None):
        v_ref_0 = self.get_wind_speed_at_height(wind_speeds, h_ref,
                                                heights=heights)
        return np.array(wind_speeds)*v_ref/v_ref_0

    def plot_cluster_shapes(self, scale_back_sf=False,
                            x_lim_profiles=[-2.2, 3.2],
                            y_lim_profiles=[-1.7, 1.7]):
        from .wind_profile_clustering import plot_wind_profile_shapes
        # TODO this can only plot 8 cluster shapes for now
        for i in range(self.config.Clustering.n_clusters):
            prl_i, prp_i, heights, sf = \
                self.read_profiles(i_profile=i+1)
            if i == 0:
                print(heights)
                prl = np.zeros((self.config.Clustering.n_clusters,
                                len(heights)))
                prp = np.zeros((self.config.Clustering.n_clusters,
                                len(heights)))
            if scale_back_sf:
                prl_i = np.array(prl_i)/sf
                prp_i = np.array(prp_i)/sf
            prl[i, :] = prl_i
            prp[i, :] = prp_i

        plot_wind_profile_shapes(self.config,
                                 heights,
                                 prl, prp,
                                 (prl ** 2 + prp ** 2) ** .5,
                                 x_lim_profiles=x_lim_profiles,
                                 y_lim_profiles=y_lim_profiles)

    def plot_cluster_frequency(self):
        plot_frequency(self.config)

    def analyse_pc(self, data=None, pipeline=None,
                   remove_low_wind_samples=True,
                   return_data=False):
        if data is None:
            data = self.preprocess_data(
                get_wind_data(self.config),
                remove_low_wind_samples=remove_low_wind_samples,
                return_copy=False)
        else:
            data = self.preprocess_data(
                data,
                remove_low_wind_samples=remove_low_wind_samples,
                return_copy=True)

        altitudes = data['altitude']
        normalized_data = data['training_data']
        # Perform principal component analyis.
        n_features = self.config.Clustering.n_pcs
        if pipeline is None:
            try:
                pipeline = self.read_pipeline(read_pca=True)
                data_pc = pipeline.transform(normalized_data)
            except FileNotFoundError:
                print('No PCA pipeline predefined. Run PCA on data..')
                pca = PCA(n_components=n_features)
                pipeline = pca
                # print("{} features reduced to {} components.".format(
                #     n_features, n_components))
                data_pc = pipeline.fit_transform(normalized_data)

        print("{:.1f}% of variance retained ".format(
            np.sum(pipeline.explained_variance_ratio_[:2])*100)
            + "using first two principal components.")
        cum_var_exp = list(accumulate(pipeline.explained_variance_ratio_*100))
        print("Cumulative variance retained: " + ", ".join(["{:.2f}".format(var)
                                                            for var in cum_var_exp]
                                                           ))
        var = pipeline.explained_variance_

        # Plot results.
        plot_frequency_projection(self.config, data_pc)
        markers_pc1, markers_pc2 = [-var[0]**.5, var[0]**.5, 0, 0], \
            [0, 0, -var[1]**.5, var[1]**.5]
        # TODO what is this?
        plt.plot(markers_pc1, markers_pc2, 's', mfc="white",
                 alpha=1, ms=12, mec='k')
        for i, (pc1, pc2) in enumerate(zip(markers_pc1, markers_pc2)):
            plt.plot(pc1, pc2, marker='${}$'.format(i+1), alpha=1, ms=7, mec='k')
        if not self.config.Plotting.plots_interactive:
            plt.savefig(self.config.IO.plot_output
                        .format(title='pc_frequency_projection_markers')
                        .replace('.pdf', '.png'))

        def get_pc_profile(i_pc=-1, multiplier=1., plot_pc=False):
            # Determine profile data by transforming data
            # in PC to original coordinate system.
            if i_pc == -1:
                mean_profile = pipeline.inverse_transform(np.zeros(n_features))
                profile = mean_profile
            else:
                profile_cmp = np.zeros(n_features)
                profile_cmp[i_pc] = multiplier
                profile = pipeline.inverse_transform(profile_cmp)
                if plot_pc:
                    mean_profile = pipeline.inverse_transform(np.zeros(n_features))
                    profile -= mean_profile
            prl = profile[:len(altitudes)]
            prp = profile[len(altitudes):]
            return prl, prp

        plot_mean_and_pc_profiles(self.config,
                                  altitudes,
                                  var,
                                  get_pc_profile)
        if return_data:
            return data

    def cluster_pc_projection(self, data=None, pipeline=None,
                              remove_low_wind_samples=True, return_data=True):
        if data is None:
            data = get_wind_data(self.config)
        if 'training_data' not in data:
            wind_data = self.preprocess_data(
                data,
                remove_low_wind_samples=remove_low_wind_samples,
                return_copy=True)
            data = self.preprocess_data(
                data,
                remove_low_wind_samples=False,
                return_copy=False)

        altitudes = wind_data['altitude']
        normalized_data = wind_data['training_data']
        # Perform principal component analyis.
        n_features = self.config.Clustering.n_pcs
        if pipeline is None:
            try:
                pipeline = self.read_pipeline(read_pca=True)
                data_pc = pipeline.transform(normalized_data)
            except FileNotFoundError:
                print('No PCA pipeline predefined. Run PCA on data..')
                pca = PCA(n_components=n_features)
                pipeline = pca
                # print("{} features reduced to {} components.".format(
                #     n_features, n_components))
                data_pc = pipeline.fit_transform(normalized_data)

        # Read cluster shapes, transform to PC system
        prl, prp = np.empty((self.config.Clustering.n_clusters,
                             len(altitudes))), \
            np.empty((self.config.Clustering.n_clusters,
                      len(altitudes)))
        for i_c in range(self.config.Clustering.n_clusters):
            prl_i, prp_i, heights, scale_factor = \
                self.read_profiles(i_profile=int(i_c+1))
            prl[i_c, :] = prl_i / scale_factor
            prp[i_c, :] = prp_i / scale_factor
        clusters_pc = pipeline.transform(np.concatenate((prl, prp), 1))

        labels, backscaling, n_samples_per_loc, cluster_mapping = \
            self.read_labels()

        freq = np.zeros(self.config.Clustering.n_clusters)
        n_samples = len(labels)
        # Labels: Index of the cluster each sample belongs to.
        print(labels)
        for l in labels:
            print(l, type(l))
            freq[l] += 100. / n_samples

        # Full datetime information
        # Care: datetime is not reduced in preprocessing low wind speed cut
        data['datetime'] = np.array(list(
            data['datetime'])*self.config.Data.n_locs)
        visualise_patterns(
            self.config,
            data, labels,
            freq)
        # Predict labels reduced data:
        clustering_pipeline = self.read_pipeline()
        labels_reduced = self.predict_labels(data=wind_data,
                                             pipeline=clustering_pipeline,
                                             cluster_mapping=cluster_mapping,
                                             write_output=False)[0]

        print(labels_reduced.shape, data_pc.shape)
        projection_plot_of_clusters(self.config, data_pc,
                                    labels_reduced,
                                    clusters_pc)
        if return_data:
            return data


    def original_vs_cluster_wind_profile_shapes(self,
                                                loc=None,
                                                sample_id=None,
                                                x_lim=(-17, 17),
                                                y_lim=(-17, 17),):
        # Read sample wind data
        if loc is None:
            loc = self.config.Data.locations[0]
        if sample_id is None:
            sample_id = 0
        data = get_wind_data(self.config,
                             sel_sample_ids=[sample_id],
                             locs=[loc])
        data = self.preprocess_data(
            data,
            remove_low_wind_samples=False,
            return_copy=False,
            normalize=True)

        # Predict Cluster / read label
        if loc in self.config.Data.locations:
            # Read labels
            labels, backscaling, n_samples_per_loc, cluster_mapping = \
                self.read_labels()
            i_loc = self.config.Data.locations.index(loc)
            i = i_loc*n_samples_per_loc + sample_id
            i_cluster = labels[i]
            backscaling = backscaling[i]
        else:
            labels, backscaling, _ = self.predict_labels(
                data=data,
                remove_low_wind_samples=False,
                write_output=False)
            i_cluster = int(labels[0])

        prl, prp, heights, scale_factor = \
            self.read_profiles(i_profile=int(i_cluster+1))
        if heights != self.config.Data.height_range:
            print('Height ranges in m:')
            print('Clustering: ', heights)
            print('Data: ', self.config.Data.height_range)
            print('---')
            raise ValueError("Height ranges don't match!")

        # Plot matched cluster profile (with cluster tag),
        # backscaled to sample profile
        # and original sample profile in same plot
        wind_prl = data['wind_speed_parallel'][0, :]
        wind_prp = data['wind_speed_perpendicular'][0, :]
        training_prl, training_prp = data['training_data'][0, :][:len(heights)],\
            data['training_data'][0, :][len(heights):]
        print(wind_prl, training_prl*data['normalisation_value'][0])
        print('------------')
        print(wind_prp, training_prp*data['normalisation_value'][0])
        print('-----------------')
        print(backscaling, data['normalisation_value'][0], scale_factor[0],
              data['normalisation_value'][0]/scale_factor[0])
        print('------------')
        print('------------')
        wind_mag = np.sqrt(wind_prl**2 + wind_prp**2)
        cluster_prl = np.array(prl)*backscaling
        cluster_prp = np.array(prp)*backscaling
        cluster_mag = np.sqrt(cluster_prl**2 + cluster_prp**2)
        plot_original_vs_cluster_wind_profile_shapes(
            self.config,
            heights, wind_prl, wind_prp,
            cluster_prl, cluster_prp,
            int(i_cluster+1),
            wind_mag=wind_mag, cluster_mag=cluster_mag,
            x_lim=x_lim,
            y_lim=y_lim,
            loc_tag='_lat_{:.2f}_lon_{:.2f}_sample_{}'.format(
                loc[0], loc[1], sample_id))

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
        data = self.preprocess_data(
            data,
            remove_low_wind_samples=False,
            return_copy=False,
            normalize=True)
        # Read PCA pipeline
        pca_pipeline = self.read_pipeline(read_pca=True)

        # Predict Cluster / read label
        if loc in self.config.Data.locations:
            # Read labels
            labels, backscaling, n_samples_per_loc, cluster_mapping = \
                self.read_labels()
            i_loc = self.config.Data.locations.index(loc)
            i = i_loc*n_samples_per_loc + sample_id
            i_cluster = labels[i]
        else:
            labels, backscaling, _ = self.predict_labels(
                data=data,
                remove_low_wind_samples=False,
                write_output=False)
            i_cluster = int(labels[0])
        print('i cluster profile: ', i_cluster+1)

        prl, prp, heights, scale_factor = \
            self.read_profiles(i_profile=int(i_cluster+1))
        if heights != self.config.Data.height_range:
            print('Height ranges in m:')
            print('Clustering: ', heights)
            print('Data: ', self.config.Data.height_range)
            print('---')
            raise ValueError("Height ranges don't match!")

        plot_cluster_profile = (np.array(prl)/scale_factor,
                                np.array(prp)/scale_factor,
                                i_cluster)
        # Plot PCA - and Clustering flow after preprocessing on one sample
        from .principal_component_analysis import sample_profile_pc_sum
        sample_profile_pc_sum(self.config,
                              data,
                              n_pcs=self.config.Clustering.n_pcs,
                              i_sample=0,
                              pca_pipeline=pca_pipeline,
                              plot_cluster_profile=plot_cluster_profile,
                              loc_tag='_lat_{:.2f}_lon_{:.2f}_sample_{}'
                              .format(loc[0], loc[1], sample_id))

    # reshape labels -> locations file
    # plot same as cluster profiles for each i_cluster: freq map
    def cluster_frequency_maps(self, use_rel='cluster', read_input=True):  # or None...
        locations = self.config.Data.locations
        # Prepare the general map plot.
        file_name = self.config.IO.plot_output.format(
            title='cluster_map_projections_rel_to_{}'
            .format(use_rel)).replace('.pdf', '.pickle')
        try:
            if not read_input:
                raise FileNotFoundError
            with open(file_name, 'rb') as f:
                cluster_frequency = pickle.load(f)
            print('Frequency map output read.')
        except FileNotFoundError:
            # Read labels
            labels, backscaling, n_samples_per_loc, cluster_mapping = \
                self.read_labels()
            labels = labels.reshape(len(locations), n_samples_per_loc)
            cluster_frequency = np.empty((self.config.Clustering.n_clusters,
                                          len(locations)))
            for i_cluster in range(self.config.Clustering.n_clusters):
                if use_rel == 'cluster':
                    rel = np.sum(labels == i_cluster)
                elif use_rel == 'loc':
                    rel = n_samples_per_loc
                else:
                    raise ValueError('Relative basis not correctly specified,'
                                     ' neither "cluster" nor "loc"')
                cluster_frequency[i_cluster, :] = np.sum(
                    labels == i_cluster, axis=1)/rel*100

            pickle.dump(cluster_frequency, open(file_name, 'wb'))
            print('Frequency map output written.')
        plot_cluster_freq_maps(self.config,
                               cluster_frequency,
                               n_rows=self.config.Clustering.n_clusters//4,
                               tag=use_rel)

    def read_profiles(self, i_profile=None):
        profiles = pd.read_csv(
            self.config.IO.profiles, sep=";")
        if i_profile is None:
            return profiles
        else:
            i_profile = int(i_profile)
            scale_factor = profiles['scale factor{} [-]'.format(i_profile)]
            heights = list(profiles['height [m]'])
            prl = list(profiles['u{} [-]'.format(i_profile)])
            prp = list(profiles['v{} [-]'.format(i_profile)])
            return prl, prp, heights, scale_factor

    def read_pipeline(self, read_pca=False):
        if not read_pca:
            with open(self.config.IO.cluster_pipeline, 'rb') as f:
                pipeline = pickle.load(f)
        else:
            with open(self.config.IO.pca_pipeline, 'rb') as f:
                pipeline = pickle.load(f)

        return pipeline

    def read_labels(self, data_type='Data',
                    file_name=None,
                    return_file=False,
                    locs_slice=None):
        if file_name is None:
            if data_type in ['Data', 'data']:
                file_name = self.config.IO.labels
            elif data_type in ['Training', 'training']:
                file_name = self.config.IO.training_labels
        if locs_slice is not None:
            file_name = file_name.replace(
                '.pickle',
                '{}_n_{}.pickle'.format(locs_slice[0], locs_slice[1]))
            print('Reading labels of slice of locations:'
                  ' {} in {} locs slices'.format(locs_slice[0], locs_slice[1]))
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
