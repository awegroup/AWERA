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

# Location from parameter
loc_id = int(os.environ['LOC_ID'])
# Failed:  292
#rerun_ids = [376, 1057, 1058, 1061, 1063, 1064, 1067, 1068, 1070, 1072, 1073, 1074, 1076, 1078, 1079, 1080, 1081, 1082, 1083, 1086, 1087, 1089, 1093, 1094, 1098, 1099, 1100, 1107, 1109, 1110, 1120, 1121, 1123, 1124, 1137, 1145, 1146, 1151, 1164, 1166, 1180, 1181, 1189, 1197, 1198, 1204, 1206, 1208, 1209, 1211, 1216, 1217, 1226, 1227, 1272, 1273, 1281, 1295, 1296, 1321, 1322, 1350, 1351, 1352, 1379, 1391, 1529, 1604, 1627, 1631, 1698, 1699, 1700, 1702, 1724, 1819, 1851, 1894, 1933, 1934, 1935, 1937, 1963, 1964, 1967, 1969, 1979, 1982, 1995, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200]
#loc_id = rerun_ids[loc_id]
loc = val.config.Data.locations[loc_id]
val.multiple_locations(locs=[loc])

#val.power_curve_spread(overwrite=True)

# plt.show()
# val_cluster = ValidationProcessingClustering(config)
# val_cluster.process_all()

print('Done.')
print('------------------------------ Config:')
print(val.config)
print('------------------------------ Time:')
write_timing_info('Validation run finished.',
                  time.time() - since)