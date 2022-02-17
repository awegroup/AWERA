import os
from AWERA import config
from AWERA.validation.validation import ValidationProcessingPowerProduction,\
    ValidationProcessingClustering
import matplotlib.pyplot as plt
#from .run_awera import training_settings
#settings = training_settings[5]
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()
settings = {
    'General': {'use_memmap': False},
    'Processing': {'n_cores': 23},
    'Data': {
        'n_locs': 5000,
        'location_type': 'europe'},
    'Clustering': {
        'n_clusters': 8,
        'training': {
            'n_locs': 5000,
            'location_type': 'europe'
            }
        },
    }
config.update(settings)
# TODO sample ids in file names: validation only
# | later: clustering data selection
val = ValidationProcessingPowerProduction(config)

# Location from parameter
loc_id = int(os.environ['LOC_ID'])
loc = val.config.Data.locations[loc_id]
val.multiple_locations(locs=[loc])

# val.power_curve_spread()

# plt.show()
# val_cluster = ValidationProcessingClustering(config)
# val_cluster.process_all()

print('Done.')
print('------------------------------ Config:')
print(val.config)
print('------------------------------ Time:')
write_timing_info('Validation run finished.',
                  time.time() - since)