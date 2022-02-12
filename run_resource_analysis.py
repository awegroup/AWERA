import os
import matplotlib.pyplot as plt
from AWERA import config
from AWERA.resource_analysis.resource_analysis import ResourceAnalysis

import matplotlib.pyplot as plt
#from .run_awera import training_settings
#settings = training_settings[5]
import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()
settings = {
    'Processing': {'n_cores': 3},
    'Data': {
        'n_locs': 500,  # 10,
        'location_type': 'europe_ref'},
    'Clustering': {
        'n_clusters': 8,
        'training': {
            'n_locs': 5000,
            'location_type': 'europe'
            }
        },
    }


print(config.Data.locations)
config.update(settings)

ra = ResourceAnalysis(config)
# ra.single_loc_plot(loc=(52.0, 5.0),  # Caubauw
#                    time_ids=None,
#                    ceiling=500,
#                    floor=50)
# ra.plot_all_maps()
ra.height_range_sanity_check()

plt.show()
print('Done.')
print('------------------------------ Config:')
print(ra.config)
print('------------------------------ Time:')
write_timing_info('Validation run finished.',
                  time.time() - since)