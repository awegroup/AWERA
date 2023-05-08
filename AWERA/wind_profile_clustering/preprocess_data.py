import numpy as np
from copy import copy

from .read_requested_data import get_wind_data


def express_profiles_wrt_ref_vector(data, ref_vector_height,
                                    use_memmap=False):
    # CCW w.r.t. East
    if use_memmap:
        wind_direction = np.memmap('tmp/wind_direction.memmap',
                                   dtype='float64', mode='w+',
                                   shape=data['wind_speed_north'].shape)
        wind_direction[:] = np.arctan2(data['wind_speed_north'],
                                       data['wind_speed_east'])
    else:
        wind_direction = np.arctan2(data['wind_speed_north'],
                                    data['wind_speed_east'])
        # data['original_wind_direction_atan_NE'] = copy(wind_direction)

    # TODO: check if interpolation can be done without the loop
    if use_memmap:
        wind_speed_ref = np.memmap('tmp/wind_speed_ref.memmap',
                                   dtype='float64', mode='w+',
                                   shape=(data['n_samples']))
        ref_dir = np.memmap('tmp/ref_dir.memmap',
                            dtype='float64', mode='w+',
                            shape=(data['n_samples']))
    else:
        wind_speed_ref = np.zeros(data['n_samples'])
        ref_dir = np.zeros(data['n_samples'])
    for i in range(data['n_samples']):
        wind_speed_ref[i] = np.interp(ref_vector_height, data['altitude'],
                                      data['wind_speed'][i, :])
        wind_speed_east_ref = np.interp(ref_vector_height, data['altitude'],
                                        data['wind_speed_east'][i, :])
        wind_speed_north_ref = np.interp(ref_vector_height, data['altitude'],
                                         data['wind_speed_north'][i, :])
        ref_dir[i] = np.arctan2(wind_speed_north_ref, wind_speed_east_ref)
    if use_memmap:  # FOXME Maybe do this after the reshapes? what is better for RAM?
        del wind_speed_ref
        del ref_dir
        wind_speed_ref = np.memmap('tmp/wind_speed_ref.memmap',
                                   dtype='float64', mode='r',
                                   shape=(data['n_samples']))
        ref_dir = np.memmap('tmp/ref_dir.memmap',
                            dtype='float64', mode='r',
                            shape=(data['n_samples']))
    data['reference_vector_speed'] = wind_speed_ref
    data['reference_vector_direction'] = ref_dir

    # Express wind direction with respect to the reference vector.
    # TODO This will load to RAM - as all reshapes will
    wind_direction = wind_direction - ref_dir.reshape((-1, 1))

    # Modify values such that angles are -pi < dir < pi.
    wind_direction = np.where(wind_direction < -np.pi,
                              wind_direction + 2*np.pi,
                              wind_direction)
    wind_direction = np.where(wind_direction > np.pi,
                              wind_direction - 2*np.pi,
                              wind_direction)
    if use_memmap:
        del wind_direction
        data['wind_direction'] = np.memmap('tmp/wind_direction.memmap',
                                           dtype='float64', mode='r',
                                           shape=data['wind_speed'].shape)
    else:
        data['wind_direction'] = wind_direction

    if use_memmap:
        wind_speed_parallel = np.memmap('tmp/wind_speed_parallel.memmap',
                                        dtype='float64', mode='w+',
                                        shape=data['wind_speed'].shape)
        wind_speed_parallel[:, :] = \
            data['wind_speed_east']*np.cos(ref_dir).reshape((-1, 1)) \
            + data['wind_speed_north']*np.sin(ref_dir).reshape((-1, 1))
        del wind_speed_parallel
        data['wind_speed_parallel'] = np.memmap(
            'tmp/wind_speed_parallel.memmap',
            dtype='float64', mode='r',
            shape=data['wind_speed'].shape)

        wind_speed_perpendicular = np.memmap(
            'tmp/wind_speed_perpendicular.memmap',
            dtype='float64', mode='w+',
            shape=data['wind_speed'].shape)
        wind_speed_perpendicular[:, :] = \
            -data['wind_speed_east']*np.sin(ref_dir).reshape((-1, 1)) + \
            data['wind_speed_north']*np.cos(ref_dir).reshape((-1, 1))
        del wind_speed_perpendicular
        data['wind_speed_perpendicular'] = np.memmap(
            'tmp/wind_speed_perpendicular.memmap',
            dtype='float64', mode='r',
            shape=data['wind_speed'].shape)
    else:
        data['wind_speed_parallel'] = \
            data['wind_speed_east']*np.cos(ref_dir).reshape((-1, 1)) \
            + data['wind_speed_north']*np.sin(ref_dir).reshape((-1, 1))
        data['wind_speed_perpendicular'] = \
            -data['wind_speed_east']*np.sin(ref_dir).reshape((-1, 1)) + \
            data['wind_speed_north']*np.cos(ref_dir).reshape((-1, 1))

    # Pickle Data
    # import pickle
    # with open('original_wind_data_and_wrt_ref.pickle', 'wb') as f:
    #     pickle.dump(data, f)
    return data


def reduce_wind_data(data, mask_keep, return_copy=False,
                     use_memmap=False):
    if return_copy:
        data = copy(data)
    n_samples_after_filter = np.sum(mask_keep)
    print("{:.1f}% of data/{} samples remain after filtering.".format(
        n_samples_after_filter/data['n_samples'] * 100.,
        n_samples_after_filter))
    skip_filter = ['altitude', 'n_samples', 'n_locs',
                   'years', 'n_samples_per_loc', 'locations',
                   'datetime_full']
    if len(data['locations']) > 1:
        skip_filter += ['datetime']
        # TODO datetime for all locations the same
        # -> masking for all locations at the same time cannot
        # be applied this way - fix: save datetime*len(locations)
        # the same datetime for all locations?
    if use_memmap:
        shape = (n_samples_after_filter, data['wind_speed'].shape[1])
    for k, val in data.items():
        if k in skip_filter:
            continue
        else:
            if use_memmap and k in ['wind_speed_east',
                                    'wind_speed_north']:
                new_memmap = np.memmap('tmp/{}_copy.memmap'.format(k),
                                       dtype='float64', mode='w+',
                                       shape=shape)
                new_memmap[:, :] = val[mask_keep]
                del new_memmap
                data[k] = np.memmap('tmp/{}_copy.memmap'.format(k),
                                    dtype='float64', mode='r',
                                    shape=shape)

            else:
                data[k] = val[mask_keep]
    data['n_samples'] = n_samples_after_filter
    return data


def remove_lt_mean_wind_speed_value(data, min_mean_wind_speed,
                                    use_memmap=False):
    sample_mean_wind_speed = np.mean(data['wind_speed'], axis=1)
    mask_keep = sample_mean_wind_speed > min_mean_wind_speed
    data = reduce_wind_data(data, mask_keep, use_memmap=use_memmap)

    return data


def normalize_data(data, use_memmap=False):
    if use_memmap:
        norm_ref = np.memmap('tmp/norm_ref.memmap',
                             dtype='float64', mode='w+',
                             shape=(data['n_samples']))
        norm_ref[:] = np.percentile(data['wind_speed'], 90., axis=1)
        if np.sum(norm_ref == 0) > 0:
            print('Non-Normalised components (zero 90th percentile): ',
                  np.sum(norm_ref == 0))
        norm_ref[norm_ref == 0] = 1
        # TODO need this? .reshape((-1, 1))
        # print('shape_single', data['wind_speed_parallel'].shape)
        # print('shape_norm', norm_ref.shape)
        training_data = np.memmap('tmp/training_data.memmap',
                                  dtype='float64', mode='w+',
                                  shape=(data['n_samples'],
                                         data['wind_speed'].shape[1]*2))

        training_data[:, :data['wind_speed'].shape[1]] = \
            data['wind_speed_parallel']/norm_ref[:, np.newaxis]
        training_data[:, data['wind_speed'].shape[1]:] = \
            data['wind_speed_perpendicular']/norm_ref[:, np.newaxis]
        del training_data
        del norm_ref
        training_data = np.memmap('tmp/training_data.memmap',
                                  dtype='float64', mode='r',
                                  shape=(data['n_samples'],
                                         data['wind_speed'].shape[1]*2))
        norm_ref = np.memmap('tmp/norm_ref.memmap',
                             dtype='float64', mode='r',
                             shape=(data['n_samples']))
        data['normalisation_value'] = norm_ref
        data['training_data'] = training_data
        # TODO need this? .reshape(-1)
        print('shape_single', data['training_data'].shape)
    else:
        norm_ref = np.percentile(data['wind_speed'], 90., axis=1).reshape((-1, 1))
        if np.sum(norm_ref == 0) > 0:
            print('Non-Normalised components (zero 90th percentile): ',
                  np.sum(norm_ref == 0))
        norm_ref[norm_ref == 0] = 1
        # print('shape_single', data['wind_speed_parallel'].shape)
        # print('shape_norm', norm_ref.shape)
        training_data_prl = data['wind_speed_parallel']/norm_ref
        training_data_prp = data['wind_speed_perpendicular']/norm_ref

        data['training_data'] = np.concatenate((training_data_prl,
                                                training_data_prp), 1)
        data['training_data'] = data['training_data'].astype(np.double)  # TODO is this astype necessary? no...?
        data['normalisation_value'] = norm_ref.reshape(-1)
        # print('shape_single', data['training_data'].shape)
    return data


def preprocess_data(config,
                    data,
                    remove_low_wind_samples=True,
                    return_copy=True,
                    normalize=True):
    if config.General.use_memmap:
        if return_copy:
            # Copy data info that isn't kept in a memmap
            data['datetime_full'] = copy(data['datetime'])
        else:
            if 'datetime_full' in data:
                data['datetime'] = copy(data['datetime_full'])
            data['n_samples'] = \
                data['n_samples_per_loc']*len(data['locations'])
            data['wind_speed_east'] = np.memmap(
                'tmp/v_east.memmap', dtype='float64', mode='r',
                shape=(data['n_samples'],
                       len(config.Data.height_range)))
            data['wind_speed_north'] = np.memmap(
                'tmp/v_north.memmap', dtype='float64', mode='r',
                shape=(data['n_samples'],
                       len(config.Data.height_range)))

        wind_speed = np.memmap('tmp/v.memmap', dtype='float64', mode='w+',
                               shape=data['wind_speed_east'].shape)
        wind_speed[:, :] = (data['wind_speed_east']**2
                            + data['wind_speed_north']**2)**.5
        del wind_speed
        wind_speed = np.memmap('tmp/v.memmap', dtype='float64', mode='r+',
                               shape=data['wind_speed_east'].shape)
        data['wind_speed'] = wind_speed
    else:
        data['wind_speed'] = (data['wind_speed_east']**2
                              + data['wind_speed_north']**2)**.5
        if return_copy:
            data = copy(data)
    if remove_low_wind_samples:
        data = remove_lt_mean_wind_speed_value(
            data, 5., use_memmap=config.General.use_memmap)
        # TODO here is now always memmap copied
    data = express_profiles_wrt_ref_vector(
        data,
        config.General.ref_height,
        use_memmap=config.General.use_memmap)
    if normalize:
        data = normalize_data(data, use_memmap=config.General.use_memmap)
    else:
        # TODO memmap
        # Non-normalized data :
        # imitate data structure with norm factor set to 1
        data['normalisation_value'] = np.zeros(
            (data['wind_speed'].shape[0])) + 1
        data['training_data'] = np.concatenate(
            (data['wind_speed_parallel'],
             data['wind_speed_perpendicular']),
            1)

    return data


if __name__ == '__main__':
    from ..config import Config
    config = Config()
    # Read data
    wind_data = get_wind_data(config)
    preprocess_data(wind_data)
