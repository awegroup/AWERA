import numpy as np
import pickle
import os.path

# TODO Docstrings
reference_locs = [
    (52.0, 5.0),  # Cabauw (Netherlands)
    (52.5, 3.25),  # Ijmuiden (North Sea Close to Netherlands)
    (47.5, 9.5),  # Bodensee (Southern Germany, North of Alpes)
    (57.5, 8.25),  # Northern Denmark Sea
    (60.0, -5.0)  # Faroe (Sea South of Island)
    ]
# TODO automate inclusion in loc selection?

# for i in range(5):
#     get_locations(loc_file, 'n_loc_test_{}'.format(i), 1, [65, 30], [-20, 20], 0.25, init_locs=[special_locs[i]])

def grid_round(x, prec=2, base=0.25):     return (base * (np.array(x) / base).round()).round(prec)


def print_location_list(locations):
    print('[', end='')
    for i, loc in enumerate(locations):
        print('(', end ='')
        print(*loc, sep=',', end='')
        print(')', end = '')
        if not i == len(locations)-1: print(',', end='')
    print(']')


def remove_doubles(a):
    doubles = [j for i, a1 in enumerate(a)
     for j, a2 in enumerate(a)
     if (a1[0] == a2[0] and a1[1] == a2[1] and j > i)]
    a = [a1 for j, a1 in enumerate(a) if not j in doubles]
    return a


def abs_of_neg_zero(a):
    for j, (a1, a2) in enumerate(a):
        if -a1 == a1:
            a1 = np.abs(a1)
        if -a2 == a2:
            a2 = np.abs(a2)
        a[j] = (a1, a2)
    return a


def random_locations(n=10,
                     lat_range=[65.0, 30],
                     lon_range=[-20, 20],
                     base=0.25,
                     initial_locs=[]):
    # Randomly select n locations
    # Uniform draws from [min, max) -> slightly extend max
    epsilon = 0.0000001
    if lat_range[0] > lat_range[1]:
        lat_max = lat_range[0] + epsilon*base
        lat_min = lat_range[1]
    else:
        lat_max = lat_range[1] + epsilon*base
        lat_min = lat_range[0]
    if lon_range[0] > lon_range[1]:
        lon_max = lon_range[0] + epsilon*base
        lon_min = lon_range[1]
    else:
        lon_max = lon_range[1] + epsilon*base
        lon_min = lon_range[0]
    xy_min = (lat_min, lon_min)
    xy_max = (lat_max, lon_max)
    locations = np.random.default_rng().uniform(low=xy_min,
                                                high=xy_max,
                                                size=(n, 2))
    locations = grid_round(locations, base=base)
    locations = [(lat, lon) for lat, lon in locations]
    locations = abs_of_neg_zero(locations)
    if len(initial_locs) > 0:
        locations += initial_locs
    locations = remove_doubles(locations)
    n_diff = n - (len(locations) - len(initial_locs))
    if n_diff != 0:
        locations = random_locations(n=n_diff,
                                     lat_range=lat_range,
                                     lon_range=lon_range,
                                     base=base,
                                     initial_locs=locations)
    if len(initial_locs) == 0:
        # Print output to copy here in case new random locations are to be defined - maybe later write out as csv?
        print_location_list(locations)
        print('Test n:', n, len(locations), len(remove_doubles(locations)))
    return(locations)


def get_locations(file_name, location_type, n_locs, lat_range, lon_range,
                  grid_size, init_locs=[]):
    # TODO fix BAF error random generation xmax < xmin?
    n_max_loc = (((lat_range[1]-lat_range[0])/grid_size + 1)
                 * ((lon_range[1]-lon_range[0])/grid_size + 1))
    if n_locs == -1 or n_locs == n_max_loc:
        locations_file = file_name.format(
                location_type=location_type,
                n_locs='all')
    else:
        locations_file = file_name.format(
                location_type=location_type,
                n_locs=n_locs)
    if len(init_locs) > 0:
        # TODO include option for custom locations in chain config/...
        locations = init_locs
        res = {'n_locs': n_locs,
               'location_type': location_type,
               'lat_range': lat_range,
               'lon_range': lon_range,
               'grid_size': grid_size,
               'locations': locations,
               }
        # Pickle results
        with open(locations_file, 'wb') as f:
            pickle.dump(res, f)
    elif os.path.isfile(locations_file):
        # Locations already generated
        with open(locations_file, 'rb') as f:
            res = pickle.load(f)
        locations = res['locations']
    else:
        if n_locs != -1:
            # Random uniform selection of n_locs locations
            locations = random_locations(n=n_locs,
                                         lat_range=lat_range,
                                         lon_range=lon_range,
                                         base=grid_size)
        else:
            # Select all locations
            if lat_range[0] > lat_range[1]:
                all_lats = list(np.arange(lat_range[0], lat_range[1]-grid_size,
                                          -grid_size))
            else:
                all_lats = list(np.arange(lat_range[0], lat_range[1]+grid_size,
                                          grid_size))
            if lon_range[0] > lon_range[1]:
                all_lons = list(np.arange(lon_range[0], lon_range[1]-grid_size,
                                          -grid_size))
            else:
                all_lons = list(np.arange(lon_range[0], lon_range[1]+grid_size,
                                          grid_size))
            locations = [(lat, lon) for lat in all_lats for lon in all_lons]
        res = {'n_locs': n_locs,
               'location_type': location_type,
               'lat_range': lat_range,
               'lon_range': lon_range,
               'grid_size': grid_size,
               'locations': locations,
               }
        # Pickle results
        with open(locations_file, 'wb') as f:
            pickle.dump(res, f)
    return locations
