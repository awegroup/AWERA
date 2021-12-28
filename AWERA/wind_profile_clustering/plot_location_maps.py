import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pickle
import cartopy
import cartopy.crs as ccrs
#TODO put in utils/eval...? function necessary for what? -> eval?


def plot_location_map(config):
    locations = config.Data.locations
    # Prepare the general map plot.
    with open(config.IO.labels, 'rb') as f:
        labels_file = pickle.load(f)
    labels = labels_file['labels [-]']
    locations = labels_file['locations']
    n_samples_per_loc = labels_file['n_samples_per_loc']

    # Evaluate locations data
    loc_data_cluster = np.zeros(len(locations))
    max_cluster_found = np.zeros(config.Clustering.n_clusters)
    for i_loc in range(len(locations)):
        loc_labels = labels[
            n_samples_per_loc*i_loc:n_samples_per_loc*(i_loc+1)]
        cluster_frequency = np.zeros(config.Clustering.n_clusters)
        for i_cluster in range(config.Clustering.n_clusters):
            cluster_frequency[i_cluster] = sum(loc_labels == i_cluster+1)
        max_cluster = max(zip(cluster_frequency,
                              np.arange(config.Clustering.n_clusters)+1),
                          key=lambda t: t[0])[1]
        loc_data_cluster[i_loc] = max_cluster
        max_cluster_found[max_cluster-1] = 1

    # print(loc_data_cluster)

    # Compute relevant map projection coordinates
    lats = [lat for lat, _ in locations]
    lons = [lon for _, lon in locations]

    cm = 1/2.54
    fig = plt.figure(figsize=(13*cm, 14.5*cm))


    mrc = ccrs.Mercator()
    ax = plt.axes(projection=mrc)
    ax.coastlines()  # TODO resolution='50m', color='black', linewidth=1)
    plt.title("Cluster mapping")
    # Compute map projection coordinates.

    # TODO what do I use this for, improve plotting/labelling
    if sum(max_cluster_found) <= 8:
        color_map = plt.get_cmap('Dark2')
    elif sum(max_cluster_found) <= 20:
        color_map = plt.get_cmap('tab20')
    else:
        # continuous
        # TODO define custom discrete color map depending on found clusters
        color_map = plt.get_cmap('nipy_spectral')

    # Map found max clusters to cluster id
    max_cluster = [i+1 for i, m in enumerate(max_cluster_found) if m == 1]

    loc_data = [max_cluster.index(c)+1 for c in loc_data_cluster]

    normalize = mpl.colors.Normalize(vmin=1, vmax=len(max_cluster))

    plt.scatter(lons, lats, c=color_map(normalize(loc_data)),
                transform=cartopy.crs.PlateCarree(),
                zorder=0.5)

    cax, _ = mpl.colorbar.make_axes(ax)
    cbar = mpl.colorbar.ColorbarBase(cax,
                                     cmap=color_map,
                                     norm=normalize,
                                     ticks=np.arange(len(max_cluster))+1,
                                     label='Cluster')
    cbar.set_ticklabels(max_cluster)
    if not config.Plotting.plots_interactive:
        plt.savefig(config.IO.result_dir
                    + 'map_max_cluster_id'
                    + config.Data.data_info
                    + '.pdf')


if __name__ == '__main__':
    from ..config import Config
    config = Config()
    plot_location_map(config)
    plt.show()
