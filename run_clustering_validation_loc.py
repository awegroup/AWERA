import os
from AWERA import config
from AWERA.validation.validation import ValidationProcessingClustering
import matplotlib.pyplot as plt

import time
from AWERA.utils.convenience_utils import write_timing_info
since = time.time()

import os
if not os.path.exists('tmp'):
    os.makedirs('tmp')

loc_id = int(os.environ['LOC_ID'])

# # read config from jobnumber
# settings_id = int(os.environ['SETTINGS_ID'])

# n_locs_settings = [200, 500, 1000, 5000]
# n_locs = n_locs_settings[settings_id]

use_memmap = True

settings = {
    'General': {'use_memmap': use_memmap},
    'Processing': {'n_cores': 20},
    'Data': {
        'n_locs': -1,
        'location_type': 'europe'},
    'Clustering': {
        'n_clusters': 80,
        'training': {
            'n_locs': 1000,
            'location_type': 'europe'
            }
        },
    }
config.update(settings)
# TODO sample ids in file names: validation only
# | later: clustering data selection
print('Config updated.')

# plt.show()
# run 3,4,5,6,7 pcs on 500 locs data and training
val = ValidationProcessingClustering(config)
print(val.config.Data.n_locs)
val_settings = {'Clustering': {
    # Run validation processing on:
    'eval_n_clusters': [8, 16, 80],
    'eval_n_pc_up_to': 5,
    # Detailed analysis of:
    'eval_n_pcs': [5],
    'eval_heights': [300, 400, 500],
    # TODO move to General is also used -> Power
    # Defining the wind speed bin ranges to the next element,
    # second to last till maximum velocity; the last 0 refers to full sample
    'split_velocities': [0, 1.5, 3, 5, 10, 20, 25, 0],
    'wind_type_eval': ['abs'],  # 'parallel', 'perpendicular'],

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

working_title = 'clustering_loc_validation_{}'.format(loc_id)  # 'run_profile'
res = val.process_all(min_n_pcs=5,
                      save_full_diffs=False,
                      loc_cluster_only=True,
                      locs_slice=(loc_id, 1000))

import numpy as np
np.set_printoptions(threshold=50)
print(res)
if use_memmap:
    import os
    import glob
    # Get a list of all the file paths that ends with .memmap from in specified directory
    fileList = glob.glob('tmp/*.memmap')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except OSError:
            print("Error while deleting file : ", filePath)
    folder_path = 'tmp'
    if os.path.exists(folder_path):

        # checking whether the folder is empty or not
        if len(os.listdir(folder_path)) == 0:
            # removing the file using the os.remove() method
            os.rmdir(folder_path)
            print('Tmp files cleaned up.')
        else:
            # messaging saying folder not empty
            print("Tmp folder is not empty")
    else:
        # file not found message
        print("Tmp Folder not found in the directory")

print('Done.')
print('------------------------------ Config:')
print(val.config)
print('------------------------------ Time:')
write_timing_info('Validation run finished.',
                  time.time() - since)