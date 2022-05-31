import numpy as np

def sliding_window_avg(data, i_window):
    avg = np.empty(data.shape)

    n_samples = data.shape[1]
    for i_sample in range(n_samples):
        if i_sample % 10000 == 0:
            print('{}/{} samples'.format(i_sample + 1, n_samples))
        if i_sample < i_window//2:
            i_low = 0
            i_high = i_window
        else:
            if i_sample >= (n_samples - i_window//2 - i_window % 2):
                i_low = n_samples - i_window
                i_high = n_samples
            else:
                i_low = i_sample - i_window//2
                i_high = i_sample + i_window//2 + i_window % 2

        avg[:, i_sample] = np.mean(data[:, i_low:i_high], axis=1)
    return avg

    # TODO make processing of i_sample parallel

def count_consecutive_bool(data, return_all_counts=False):
    max_True_time = np.empty(data.shape[0])
    max_False_time = np.empty(data.shape[0])
    n_locs = data.shape[0]
    for i_loc in range(n_locs):
        condition = data[i_loc, :]
        # TODO does this work with masked
        consec = np.diff(np.where(np.concatenate(
            ([True],
             condition[:-1] != condition[1:],
             [True])))[0])
        if condition[0]:
            consec_True = consec[::2]
            consec_False = consec[1::2]
        else:
            consec_True = consec[1::2]
            consec_False = consec[::2]
        if len(consec_True) > 0:
            max_True_time[i_loc] = np.max(consec_True)
        else:
            print('No True in i_loc', i_loc)
            max_True_time[i_loc] = 0
        if len(consec_False) > 0:
            max_False_time[i_loc] = np.max(consec_False)
        else:
            print('No False in i_loc', i_loc)
            max_False_time[i_loc] = 0
    # TODO option for time resolved return? -> radius eval?
    # , time-distance to change, ... ?
    if return_all_counts:
        return consec_True, consec_False
    else:
        return max_True_time, max_False_time

