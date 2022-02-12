import numpy as np
from ..power_production.aep_map import evaluate_aep
from .utils_validation import get_abs_rel_diff
from ..utils.plotting_utils import plot_abs_rel_step_wise


def aep_err_vs_n_locs(config,
                      prediction_settings=[],
                      set_labels=None,
                      i_ref=0):
    # evaluate special_locs_i predictions aep
    aep = []
    n_locs = []
    if set_labels is None:
        labels = []
    else:
        labels = set_labels

    # TODO check if all production/prediction is run -> try except FileNotFoundError

    for settings in prediction_settings:
        config.update({'Clustering': settings})
        # TODO drop this hard-coding... ?
        if set_labels is None:
            labels.append(
                settings['training']['location_type']
                .replace('europe_', '').replace('europe', ''))
        n_locs.append(settings['training']['n_locs'])
        if labels[-1] in ['', 'incl_ref']:
            labels[-1] = str(n_locs[-1])
        if config.Data.n_locs == 1:
            aep.append(evaluate_aep(config)[0][0])
        else:
            aep.append(evaluate_aep(config)[0])
    # Plot delta aep vs aep (training only on single loc)
    plot_config = {
        'title':  'Impact of the selected number of training locations on the AEP',
        'x_label': '# training locations',
        'x_ticks': labels,
        'output_file_name': config.IO.plot_output_data.format(
            title='diff_aep_vs_n_training_locs'),
        'plots_interactive': config.Plotting.plots_interactive,
        }

    reference = np.array(aep[i_ref])
    print(reference)
    print(aep)
    abs_diff, rel_diff = get_abs_rel_diff(reference, np.array(aep))
    print(abs_diff)
    print(abs_diff.shape)
    if config.Data.n_locs > 1:
        # Ignore Numpy Masked floating point error bug for now
        np.seterr(all='raise', under='ignore')
        abs_diff = np.mean(abs_diff, axis=1)
        rel_diff = np.mean(rel_diff, axis=1)
        np.seterr(all='raise')
    plot_abs_rel_step_wise(n_locs, abs_diff, rel_diff, **plot_config)
    # TODO add labels for first single locs
    # TODO add more n_clusters options -> 500, 2000, ...
    # TODO should be run on different 50 locations clustering runs -> standard deviation of AEP predicted by clustering

    # TODO show diffs for all ref locs in one plot?

    # TODO add functionality to validation -> also check sample-wise wind speed/ power differences? -> vs number of training locations

