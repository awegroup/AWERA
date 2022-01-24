from AWERA import config
from AWERA.validation.validation import ValidationProcessingPowerProduction,\
    ValidationProcessingClustering

#from .run_awera import training_settings
#settings = training_settings[5]

settings = {
    'Processing': {'n_cores': 10},
    # Data will be the one set in validation yaml
    # 'Data': {'n_locs': 1,
    #         'location_type': 'europe_ref_0'},
    'Clustering': {
        'n_clusters': 8,
        'training': {
            'n_locs': 5,
            'location_type': 'europe_ref'
            }
        },
    }
# TODO use more cores
settings['Processing']['n_cores'] = 50  # 23
config.update(settings)
# TODO sample ids in file names: validation only
# | later: clustering data selection
val = ValidationProcessingPowerProduction(config)
val.multiple_locations()

#val_cluster = ValidationProcessingClustering(config)
#val_cluster.process_all()