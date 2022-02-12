import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# !!! from config_production import plots_interactive
def get_velocity_bin_mask(wind_speed,
                          split_velocities=[0, 1.5, 3, 5, 10, 20, 0]):
    """Find masks for the samples matching each velocity bin.

    Parameters
    ----------
    wind_speed : ndarray
        1D (or nD) array containing wind speed values for each sample
        (and each location, ...).
    split_velocities : list, optional
        List of wind speeds defining the velocity bin intervals. The last value
        is used as lower bound to inf. If the last value is lower than the
        second to last value, both are used as lower bound to inf.
        The default is [0, 1.5, 3, 5, 10, 20, 0].

    Returns
    -------
    masks : list(ndarray)
        List of mask array containing the masks for each wind speed bin.
    tags : list(string)
        List of name tags combined via '_' for each wind speed bin.

    """
    masks = [wind_speed]*len(split_velocities)
    tags = ['']*len(split_velocities)
    for vel_idx, vel in enumerate(split_velocities):
        # TODO similar structure here and in clustering validation possible?
        # Evaluate velocity bin steps
        if split_velocities[-1] < split_velocities[-2]:
            # Last two steps not increasing: take velocity-up for both
            idx_vel_up = [len(split_velocities)-2, len(split_velocities)-1]
        else:
            idx_vel_up = [len(split_velocities)-1]
        # Find mask
        if vel_idx in idx_vel_up:
            masks[vel_idx] = wind_speed < vel
            tags[vel_idx] = ' vel {} up'.format(vel)
        else:
            masks[vel_idx] = np.logical_or(
                (wind_speed < vel),
                (wind_speed >= split_velocities[vel_idx+1]))
            tags[vel_idx] = ' vel {} to {}'.format(
                vel, split_velocities[vel_idx+1])
    return masks, tags


def get_abs_rel_diff(reference, test):
    abs_diff = np.ma.array(test) - reference
    rel_diff = abs_diff/reference
    return abs_diff, rel_diff


def diff_original_vs_reco(original,
                          reco):
    """Assess differences between original and fit reconstruction.

    Parameters
    ----------
    original : array
        Original data before fit.
    reco : array
        Reconstructed wind data via fit.

    Returns
    -------
    diffs : dict
        Dictionary of difference types containing the respective
        resulting difference for each sample.

    """
    # Absolute difference:
    absolute_difference = ma.array(reco - original)

    # Relative difference:
    # Mask div by 0 in relative difference (explicitly coded,
    # ma.masked_values(data, 0) for some data (low values?) masks all)
    print('Masking original vertical wind profile data with zero wind speed: ',
          np.sum(original == 0, axis=0),
          'samples masked in each part of vertical wind profile.')
    original_data_masked = ma.array(original)
    original_data_masked[original == 0] = ma.masked
    # Set ignore underflow - otherwise masked array error
    np.seterr(all='raise', under='ignore')
    relative_difference = absolute_difference/original_data_masked

    # Average over all samples - mean and standard deviation:
    diffs = {
            'absolute': absolute_difference,
            'relative': relative_difference
            }

    return diffs


def diffs_original_vs_reco(original, reco, n_altitudes,
                           wind_type_eval=['abs']):
    """Evaluate precision of reconstructed data versus original data.

    Parameters
    ----------
    original : array
        Original data (parallel and perpendicular wind speed) before fit.
        Same normalisation as reco.
    reco : array
        Reconstructed wind data (parallel and perpendicular wind speed)
        via fit. Same normalisation as original.
    n_altitudes : int
        Number of height levels in the data.
    wind_type_eval : list, optional
        Only evaluate selected wind orientations (parallel,
        perpendicular, abs). The default is ['abs'].

    Returns
    -------
    diffs_sample_mean : dict
        Dictionary of difference types containing the respective resulting
        difference mean&standard deviation.
    diffs : dict
        Dictionary of difference types containing the respective resulting
        difference for each sample.


    """
    # Also compare absolute wind speeds:
    # Calc resulting reco absolute wind speed
    reco_abs_wind_speed = (reco[:, :n_altitudes]**2 +
                           reco[:, n_altitudes:]**2)**.5
    original_abs_wind_speed = (original[:, :n_altitudes]**2 +
                               original[:, n_altitudes:]**2)**.5
    # Average over all samples - mean and standard deviation:
    diffs = {'parallel': diff_original_vs_reco(original[:, :n_altitudes],
                                               reco[:, :n_altitudes]),
             'perpendicular': diff_original_vs_reco(original[:, n_altitudes:],
                                                    reco[:, n_altitudes:]),
             'abs': diff_original_vs_reco(original_abs_wind_speed,
                                          reco_abs_wind_speed)
             }

    diffs_sample_mean = {}

    for wind_orientation in diffs:
        if wind_orientation not in wind_type_eval:
            continue
        diffs_sample_mean[wind_orientation] = {}
        for diff_type, val in diffs[wind_orientation].items():
            diffs_sample_mean[wind_orientation][diff_type] = (np.mean(val,
                                                                      axis=0),
                                                              np.std(val,
                                                                     axis=0))
    return diffs_sample_mean, diffs
