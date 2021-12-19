import os
from AWERA import config
import AWERA

training_settings = []
prediction_settings = []

# Test run on 5 special locations
for i in range(5):
    for j in range(5):
        settings = {
            'Processing': {'n_cores': 10},
            'Data': {'n_locs': 1,
                     'location_type': 'europe_ref_{}'.format(i)},
            'Clustering': {
                'training': {'n_locs': 1,
                     'location_type': 'europe_ref_{}'.format(j)}},
            }
        if i == 0:
            training_settings.append(settings)
        prediction_settings.append(settings)

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'n_clusters': 8,
            'training': {'n_locs': 5,
                 'location_type': 'europe_ref'}},
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

    # multiple locations
    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {'n_locs': 50,
                 'location_type': 'europe'}},
        }

    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {'n_locs': 200,
                 'location_type': 'europe'}},
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)


    # multiple locations plus 5 special locs
    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {'n_locs': 55,
                 'location_type': 'europe_incl_ref'}},
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

    settings = {
        'Processing': {'n_cores': 10},
        'Data': {'n_locs': 1,
                 'location_type': 'europe_ref_{}'.format(i)},
        'Clustering': {
            'training': {'n_locs': 205,
                 'location_type': 'europe_incl_ref'}},
        }
    if i == 0:
        training_settings.append(settings)
    prediction_settings.append(settings)

# very many locations
settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {'n_locs': 5000,
             'location_type': 'europe'}},
    }
training_settings.append(settings)
# TODO include prediction
settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {'n_locs': 1000,
             'location_type': 'europe'}},
    }
training_settings.append(settings)
settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {'n_locs': 5005,
             'location_type': 'europe_incl_ref'}},
    }
training_settings.append(settings)

settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_1'},
    'Clustering': {
        'training': {'n_locs': 1005,
                     'location_type': 'europe_incl_ref'}},
    }
training_settings.append(settings)

# TODO make clear how each module runs with the inout of the previous
# power_model.load() .generate() how is the object made/passed

if __name__ == '__main__':
    # read config from jobnumber
    # 8 small jobs
    # 4 big jobs
    settings_id = int(os.environ['SETTINGS_ID'])

    settings = training_settings[(-1 - settings_id)]
    print(settings)
    # Update settings to config
    config.update(settings)

    # Code Profiling
    # TODO include in config -> optional
    # imports at top level / optional
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()

    # Run full clustering, production, aep estimation
    # depending on flags set in config
    # TODO include all important flags here to update conveniently
    #run_full(config)
    # TODO check if clustering etc has to be done?
    #AWERA.eval.optimal_harvesting_height.eval_wind_speed_at_harvesting_height(config)
    #AWERA.power_production.aep_map.compare_cf_AWE_turbine()
    #AWERA.resource_analysis.plot_maps.plot_all()
    import matplotlib.pyplot as plt
    #AWERA.power_production.aep_map.aep_map(config)
    #AWERA.wind_profile_clustering.plot_location_maps.plot_location_map(config)
    #plt.show()
    AWERA.chain.awera_chain.run_full(config)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.strip_dirs()
    stats.sort_stats('tottime').print_stats()
    # TODO only print/dump first 20 or so
    #stats.dump_stats(config.IO.result_dir + config.IO.format.plot_output.format(data_info=(config.Data.data_info + '_' + config.Clustering.training.data_info)).replace('.pdf', '.profile').format(title='run_profile', ))

    #import sys
    #sys.exit()

