import pandas as pd

import numpy as np

from config import n_clusters, file_name_profiles, data_info
from wind_profile_clustering import plot_wind_profile_shapes


df = pd.read_csv(file_name_profiles, sep=";")
heights = df['height [m]']
prl = np.zeros((n_clusters, len(heights)))
prp = np.zeros((n_clusters, len(heights)))

for i in range(n_clusters):
    u = df['u{} [-]'.format(i+1)]
    v = df['v{} [-]'.format(i+1)]
    sf = df['scale factor{} [-]'.format(i+1)]
    prl[i,:] = u/sf
    prp[i,:] = v/sf

   
plot_wind_profile_shapes(heights, prl, prp, (prl ** 2 + prp ** 2) ** .5, plot_info=data_info)
#plt.show()

