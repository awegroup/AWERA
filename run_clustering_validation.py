import os
from AWERA import config
from AWERA.validation.validation import ValidationProcessingClustering
import matplotlib.pyplot as plt

import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

settings = {
    'General': {'use_memmap': False},
    'Processing': {'n_cores': 30},
    'Data': {
        'n_locs': 500,
        'location_type': 'europe'},
    'Clustering': {
        'n_clusters': 8,
        'training': {
            'n_locs': 500,
            'location_type': 'europe'
            }
        },
    }
config.update(settings)
# TODO sample ids in file names: validation only
# | later: clustering data selection


# plt.show()
# run 3,4,5,6,7 pcs on 500 locs data and training
val = ValidationProcessingClustering(config)
val_settings = {'Clustering': {
    # Run validation processing on:
    'eval_n_clusters': [8, 16, 80],
    'eval_n_pc_up_to': 8,
    # Detailed analysis of:
    'eval_n_pcs': [5, 7],
    'eval_heights': [300, 400, 500],
    # TODO move to General is also used -> Power
    # Defining the wind speed bin ranges to the next element,
    # second to last till maximum velocity; the last 0 refers to full sample
    'split_velocities': [0, 1.5, 3, 5, 10, 20, 25, 0],
    'wind_type_eval': ['abs', 'parallel', 'perpendicular'],

    # Validation step config:
    # Test impact of normalisation in preprocessing
    'do_normalize_data': True,
    # Test impact of removal of low-wind speed samples on training
    'Validation_type': {  # 'cut' low  wind samples, 'full' all data
        'training': 'cut',
        'testing': 'full',
        },
    # 'cut_training_full_test', 'cut_training_cut_test':

    # Production configuration for validation
    # Clustering evaluation
    # Run clustering on training data and export cluster labels and
    # cluster vertical wind profiles
    'make_profiles': True,
    # For finished clustering export the frequency distribution matching the
    # velocity binning of the cut-in/out wind speed ranges from
    # power production simulation
    'make_freq_distr': True,
    'n_wind_speed_bins': 100,
    # Predict cluster labels for new data given in Data from already trained
    # clustering on the data given in Clustering-training
    'predict_labels': True,

    'save_pca_pipeline': True
    }}
val.config.update(val_settings)

res = val.process_all(min_n_pcs=3,
                      save_full_diffs=True)
import numpy as np
np.set_printoptions(threshold=50)
print(res)
print('Done.')
print('------------------------------ Config:')
print(val.config)
print('------------------------------ Time:')
write_timing_info('Validation run finished.',
                  time.time() - since)