#TODO import correct functions from within package

def aep_err_vs_n_locs(config, prediction_settings, i=0):
    # evaluate special_locs_i predictions aep
    aep = []
    n_locs = []
    labels = []
    # TODO hard-coded 10, i, prediction_settings ...
    eval_settings = prediction_settings[10*i: 10*i+10]

    # TODO check if all production is run -> try except FileNotFoundError
    from ..power_production.aep_map import evaluate_aep
    for settings in eval_settings:
        config.update(settings)
        # TODO drop this hard-coding, make optional
        labels.append(settings['Clustering']['training']['location_type'].replace('europe_', '').replace('europe', ''))
        n_locs.append(settings['Clustering']['training']['n_locs'])
        if labels[-1] in ['', 'incl_ref']:
            labels[-1] = str(n_locs[-1])
        aep.append(evaluate_aep(config)[0][0])
    # Plot delta aep vs aep (training only on single loc)
    from AWERA.wind_profile_clustering.utils_validation import plot_abs_rel_step_wise, get_abs_rel_diff
    plot_config = {'title':  'Impact of the selected number of training locations on the AEP',
                   'x_label': '# training locations',
                   'x_ticks': labels,
                   'output_file_name': config.IO.plot_output.format(title='diff_aep_vs_n_training_locs'),
                   'plots_interactive': config.Plotting.plots_interactive,
                   }

    reference = aep[i]
    abs_diff, rel_diff = get_abs_rel_diff(reference, aep)
    plot_abs_rel_step_wise(n_locs, abs_diff, rel_diff, **plot_config)
    # TODO add labels for first single locs
    # TODO add more n_clusters options -> 500, 2000, ...
    # TODO should be run on different 50 locations clustering runs -> standard deviation of AEP predicted by clustering

    # TODO show diffs for all ref locs in one plot?

    # TODO add functionality to validation -> also check sample-wise wind speed/ power differences? -> vs number of training locations

