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
    'Processing': {'n_cores': 9},
    'Data': {
        'n_locs': 5000,
        'location_type': 'europe_ref'},
    'Clustering': {
        'n_clusters': 80,
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

# val.power_curve_spread(overwrite=True, ref='p_sample',
#                        read_sample_only=True)
# val.power_curve_spread(overwrite=True, ref='p_cc_opt',
#                        read_sample_only=False)

val.plot_power_diff_maps(read_sample_only=True,
                         ref='p_sample',
                         overwrite=False)

# AEP vs n_locs: run_awera_3

# plt.show()
# val_cluster = ValidationProcessingClustering(config)
# val_cluster.process_all()

print('Done.')
print('------------------------------ Config:')
print(val.config)
print('------------------------------ Time:')
write_timing_info('Validation run finished.',
                  time.time() - since)