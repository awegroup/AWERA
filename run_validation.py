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
    'Processing': {'n_cores': 10},
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

# Location from parameter
loc_id = int(os.environ['LOC_ID'])
# Failed:  307
rerun_ids = [185, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 743, 1002, 1005, 1006, 1007, 1008, 1009, 1012, 1015, 1017, 1018, 1019, 1020, 1022, 1023, 1030, 1031, 1032, 1035, 1036, 1038, 1039, 1040, 1042, 1043, 1046, 1047, 1048, 1049, 1050, 1057, 1058, 1061, 1063, 1064, 1067, 1068, 1070, 1072, 1073, 1074, 1076, 1078, 1079, 1080, 1081, 1082, 1083, 1086, 1087, 1089, 1093, 1094, 1098, 1099, 1100, 1107, 1109, 1110, 1120, 1121, 1123, 1124, 1137, 1145, 1146, 1151, 1164, 1166, 1180, 1181, 1189, 1197, 1198, 1204, 1206, 1208, 1209, 1211, 1216, 1217, 1226, 1227, 1272, 1273, 1281, 1295, 1296, 1321, 1322, 1350, 1351, 1352, 1379, 1391, 1529, 1604, 1627, 1631, 1698, 1699, 1700, 1702, 1724, 1819, 1851, 1894, 1933, 1934, 1935, 1937, 1963, 1964, 1967, 1969, 1979, 1982, 1995, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100]
loc_id = rerun_ids[loc_id]
loc = val.config.Data.locations[loc_id]
val.multiple_locations(locs=[loc])

# val.power_curve_spread(overwrite=True)

# plt.show()
# val_cluster = ValidationProcessingClustering(config)
# val_cluster.process_all()

print('Done.')
print('------------------------------ Config:')
print(val.config)
print('------------------------------ Time:')
write_timing_info('Validation run finished.',
                  time.time() - since)