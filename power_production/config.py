# TODO Docstring
# TODO remove old

# Clustering chain

# File name definitions
from ..wind_profile_clustering.config import file_name_cluster_profiles, \
    file_name_freq_distr, cut_wind_speeds_file, data_info, data_info_training,\
    result_dir, plots_interactive, n_clusters, locations

# Optimizer:
optimizer_history_file_name = (
    result_dir
    + 'optimizer_history_{data_info}.hist'.format(data_info=data_info))

# power curves:
run_marker = ''  # default = ''
power_curve_output_file_name = (
    result_dir + 'power_curve_{{i_profile}}_{data_info}{run_info}.{{suffix}}'.
    format(data_info=data_info, run_info=run_marker))  # suffix = csv / pickle
training_power_curve_output_file_name = (
    result_dir + 'power_curve_{{i_profile}}_{data_info}{run_info}.{{suffix}}'.
    format(data_info=data_info_training, run_info=run_marker))  # suffix = csv / pickle

refined_cut_wind_speeds_file = cut_wind_speeds_file.replace('estimate',
                                                            'refined')

plot_output_file = result_dir + '{title}' + data_info + '.pdf'


# brute force config
#sample_ids = [0, 999, 70000]
#sample_selection = '3'

# TODO save in csv file
sample_ids = [22217,   295,   492,   895,   965,  1087,  1596,  1840,  2166, 2697,  2705,  3671,  4163,  4289,  4315,  4877,  5097,  5132, 5213,  5321,  5358,  5543,  6139,  6179,  6978,  7251,  7380, 7444,  7468,  7795,  7806,  8006,  8124,  8176,  8294,  8346, 8427,
              9015,  9501,  9645,  9924,  9993, 10334, 10354, 10572,       10614, 10725, 11394, 13052, 13108, 13324, 13351, 13728, 13729,      13791, 13900, 13904, 14392, 14393, 14816, 14833, 15302, 15531,       16121, 16232, 16461, 16562, 16970, 16974, 16987, 17098, 17519,       17982, 18100, 18339, 18479, 18771, 18891, 18948, 19068, 19089,       19202, 19260, 19517, 19986, 20119, 20656, 20942, 20957, 21023,       21457, 22093, 22762, 22806, 22899, 23771, 24055, 24081, 24316,       25023, 25205, 25240, 25352, 25562, 25779, 26403, 26421, 26484,       26582, 26633, 26707, 26741, 26798, 26856, 26906, 27443, 27484,       27840, 27923, 28143, 28603, 28871, 29078, 29245, 29322, 29435,       29528, 29550, 29563, 29679, 29814, 29864, 30653, 30744, 31220,       31697, 31811, 31853, 32060, 32782, 32889, 32960, 33039, 33244,       34398, 34518, 34574, 34796, 35023, 36217, 36347, 36533, 36571,       36662, 36887, 36942, 37570, 37872, 38111, 38848, 39242, 39336,       39468, 39862, 39939, 39950, 40227, 41335, 41454, 41950, 42295,       42389, 42505, 42586, 42907, 42984, 43126, 43786, 43859, 44486,       44611, 44659, 44807, 45092, 45707, 46059, 46291, 46421, 46535,       47245, 47825, 47961, 47963, 48185, 48273, 48403, 48939, 48957,       49010, 49226, 49362, 49604, 49739, 49825, 49968, 50343, 50361,       50557, 51688, 51870, 51939, 51997, 52030, 52169, 52465, 52531,       53052, 53905, 54420, 54479, 54531, 54996, 55505, 55777, 56119,       56177, 56255, 56287, 56311, 56406, 56820, 56936, 57401, 57508,       57576, 58097, 58150, 58558, 59218, 59362, 59599, 59904, 59977,       60447, 60562, 60637, 60715, 61509, 61634, 61693, 61934, 62255,       62454, 62578, 62843, 62899, 63824, 63857, 63885, 64806, 64883,       65731, 65850, 66005, 66113, 66173, 66351, 66827, 67373, 68531,       68533, 68563, 68681, 68812, 69111, 69474, 69683, 69949, 69959,       70383, 70445, 70711, 70945, 71226, 71441, 71629, 73189, 73202,       73554, 74097, 74154, 74822, 74933, 74964, 75015, 75758, 75839,       76070, 76254, 76715, 77266, 78934, 79191, 79512, 79576, 79910,       80093, 80334, 80347, 80448, 80737, 80844, 80855, 81429, 82086,       82335, 82349, 82521, 82651, 82968, 83077, 83110, 83162, 83213,       83455, 83553, 84294, 84367, 84716, 84756, 84762, 84781, 84806,       85331, 86034, 86294, 86967, 87033, 87304, 87581, 87614, 87628,       87835, 87911, 88005, 88811, 89231, 89376, 89643, 89890]
sample_selection = 'bi-weekly'


# generate sample ids
# import numpy as np
# sample_ids = np.sort(np.random.uniform(0,90000,132).astype(int))
# sample_ids = sample_ids[[True] + list(np.diff(sample_ids)!=0)]
# sample_ids = [  219,  3091,  3830,  5090,  5264,  5446,  6535,  7838,  7895, 8648,  9695, 10047, 11132, 12192, 14291, 14962, 15403, 15558, 17942, 18180, 21051, 21143, 21377, 21617, 23072, 23568, 24837, 25313, 25467, 25987, 26071, 26842, 28596, 31224, 31439, 31659, 33046, 33248, 33882, 35483, 36384, 36475, 37303, 39042, 40185, 40507, 41390, 41952, 42067, 42991, 44619, 46274, 46396, 46464, 48727, 49281, 49814, 49833, 49897, 52133, 52250, 52921, 53637, 54166, 55087, 55269, 55816, 56446, 56788, 57280, 57947, 58698, 59934, 60537, 61740, 63947, 64289, 64914, 65104, 65323, 65386, 65630, 65884, 66827, 67420, 67515, 67890, 68268, 68602, 68604, 69129, 69758, 71475, 71812, 72075, 72153, 72984, 73439, 74624, 75347, 75463, 76906, 77728, 77782, 78183, 78222, 79122, 79518, 79777, 79915, 80505, 80878, 81211, 82094, 82163, 82593, 83053, 83604, 84927, 85233, 85602, 85998, 86227, 86954, 87408, 87864, 87928, 88148, 88449, 89006, 89072, 89322]
# sample_selection = 'monthly'
# print(sample_ids)
# TODO implement with user choice: every 100th sample e.g.

# Running on a job:
import os
jobnum = int(os.environ['LOCNUM'])
step = 1
location_number = jobnum*step
# Select loctions up to location number, -1: all locs #up to +400

# single_fail_loc_ids = [469, 805, 863, 975]
# print('Set Jobnumber/Locnumber: ', jobnum, 'in steps of ', step)
# location_number = single_fail_loc_ids[jobnum]


def get_loc_brute_force_name_and_locs(location_number, data_info,
                                      mult_loc=False):
    # location_number=10
    # -1 #select loctions up to location number, -1: all locs in mult loc case

    if mult_loc:
        if location_number != -1:
            locs = locations[:location_number]
            data_info += 'first_{}_locs'.format(location_number)
        else:
            locs = locations
    else:
        locs = [locations[location_number]]
        data_info += 'loc_{}'.format(location_number)
    return data_info, locs


data_info, locs = get_loc_brute_force_name_and_locs(location_number, data_info)
# print('Locations: ', locs)

brute_force_files = 'brute_force_power_{}_samples_{}.{}'
brute_force_testing_file_name = (
    result_dir
    + brute_force_files.format(sample_selection, data_info, 'pickle'))
