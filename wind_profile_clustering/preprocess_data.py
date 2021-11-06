import numpy as np
from copy import copy

try:
    from .read_requested_data import get_wind_data
    from .config_clustering import locations
except (ImportError, ModuleNotFoundError):
    from read_requested_data import get_wind_data
    from config_clustering import locations

ref_vector_height = 100.


def express_profiles_wrt_ref_vector(data):
    # CCW w.r.t. East
    data['wind_direction'] = np.arctan2(data['wind_speed_north'],
                                        data['wind_speed_east'])

    # TODO: check if interpolation can be done without the loop
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
    data['reference_vector_speed'] = wind_speed_ref
    data['reference_vector_direction'] = ref_dir

    # Express wind direction with respect to the reference vector.
    data['wind_direction'] = data['wind_direction'] - ref_dir.reshape((-1, 1))

    # Modify values such that angles are -pi < dir < pi.
    data['wind_direction'] = np.where(data['wind_direction'] < -np.pi,
                                      data['wind_direction'] + 2*np.pi,
                                      data['wind_direction'])
    data['wind_direction'] = np.where(data['wind_direction'] > np.pi,
                                      data['wind_direction'] - 2*np.pi,
                                      data['wind_direction'])

    data['wind_speed_parallel'] = \
        data['wind_speed_east']*np.cos(ref_dir).reshape((-1, 1)) \
        + data['wind_speed_north']*np.sin(ref_dir).reshape((-1, 1))
    data['wind_speed_perpendicular'] = \
        -data['wind_speed_east']*np.sin(ref_dir).reshape((-1, 1)) + \
        data['wind_speed_north']*np.cos(ref_dir).reshape((-1, 1))
    return data


def reduce_wind_data(data, mask_keep, return_copy=False):
    if return_copy:
        data = copy(data)
    n_samples_after_filter = np.sum(mask_keep)
    print("{:.1f}% of data/{} samples remain after filtering.".format(
        n_samples_after_filter/data['n_samples'] * 100.,
        n_samples_after_filter))
    skip_filter = ['altitude', 'n_samples', 'n_locs',
                   'years', 'n_samples_per_loc', 'locations']
    if len(locations) > 1:
        skip_filter += ['datetime']
        # TODO datetime for all locations the same
        # -> masking for all locations at the same time cannot
        # be applied this way - fix: save datetime*len(locations)
        # the same datetime for all locations TODO?
    for k, val in data.items():
        if k in skip_filter:
            continue
        else:
            data[k] = val[mask_keep]
    data['n_samples'] = n_samples_after_filter
    return data


def remove_lt_mean_wind_speed_value(data, min_mean_wind_speed):
    sample_mean_wind_speed = np.mean(data['wind_speed'], axis=1)
    mask_keep = sample_mean_wind_speed > min_mean_wind_speed
    data = reduce_wind_data(data, mask_keep)

    return data


def normalize_data(data):
    norm_ref = np.percentile(data['wind_speed'], 90., axis=1).reshape((-1, 1))
    # print('shape_single', data['wind_speed_parallel'].shape)
    # print('shape_norm', norm_ref.shape)
    training_data_prl = data['wind_speed_parallel']/norm_ref
    training_data_prp = data['wind_speed_perpendicular']/norm_ref

    data['training_data'] = np.concatenate((training_data_prl,
                                            training_data_prp), 1)
    data['normalisation_value'] = norm_ref.reshape(-1)
    # print('shape_single', data['training_data'].shape)
    return data


def preprocess_data(data,
                    remove_low_wind_samples=True,
                    return_copy=True,
                    normalize=True):
    if return_copy:
        data = copy(data)
    data['wind_speed'] = (data['wind_speed_east']**2
                          + data['wind_speed_north']**2)**.5
    if remove_low_wind_samples:
        data = remove_lt_mean_wind_speed_value(data, 5.)
    data = express_profiles_wrt_ref_vector(data)
    if normalize:
        data = normalize_data(data)

    return data


if __name__ == '__main__':
    # Read data
    wind_data = get_wind_data()
    preprocess_data(wind_data)
