# TODO Docstring

# Running on a job:
import os
jobnum = int(os.environ['LOCNUM'])
step = 1
location_number = jobnum*step
# Select loctions up to location number, -1: all locs #up to +400

# single_fail_loc_ids = [469, 805, 863, 975]
# print('Set Jobnumber/Locnumber: ', jobnum, 'in steps of ', step)
# location_number = single_fail_loc_ids[jobnum]

# TODO remove this function - put in config.py what is necessary
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
