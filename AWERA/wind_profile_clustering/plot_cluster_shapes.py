import pandas as pd

import numpy as np

from .wind_profile_clustering import plot_wind_profile_shapes

# TODO in utils / eval


def plot_cluster_shapes(config, scale_back_sf=False):
    # TODO this can only plot 8 cluster shapes for now
    df = pd.read_csv(config.IO.profiles, sep=";")
    heights = df['height [m]']
    prl = np.zeros((config.Clustering.n_clusters, len(heights)))
    prp = np.zeros((config.Clustering.n_clusters, len(heights)))
    for i in range(config.Clustering.n_clusters):
        u = df['u{} [-]'.format(i+1)]
        v = df['v{} [-]'.format(i+1)]
        sf = df['scale factor{} [-]'.format(i+1)]
        if scale_back_sf:
            prl[i, :] = u/sf
            prp[i, :] = v/sf

    plot_wind_profile_shapes(heights,
                             prl, prp,
                             (prl ** 2 + prp ** 2) ** .5)
