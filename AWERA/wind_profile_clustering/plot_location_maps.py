import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from ..utils.plotting_utils import match_loc_data_map_data
import pickle
import cartopy
import cartopy.crs as ccrs
#TODO put in utils/eval...? function necessary for what? -> eval?


def plot_cluster_freq_maps(config,
                           cluster_frequency,
                           n_rows=2,
                           tag='loc'):
    c_label = r"frequency [%]"

    n_cols = int(np.ceil(cluster_frequency.shape[0]/n_rows))
    cm = 1/2.54
    figsize = (n_cols*3.5*cm+.2, n_rows*7*cm+.2)

    mrc = ccrs.Mercator()
    color_map = plt.get_cmap('coolwarm')  # 'YlOrRd')
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize,
                           subplot_kw={'projection': mrc})
    wspace = 0.001
    plt.subplots_adjust(top=1-0.01*2/n_rows, bottom=0.04*2/n_rows, left=0.01, right=0.99,
                        hspace=0.01*2/n_rows, wspace=wspace)

    # Get min/max for each row -> colorbar
    min_max = np.zeros((n_rows, 2))
    min_freq = np.min(cluster_frequency, axis=1)
    max_freq = np.max(cluster_frequency, axis=1)
    for i in range(n_rows):
        min_max[i, :] = (np.min(min_freq[i*n_cols: (i+1)*n_cols]),
                         np.max(max_freq[i*n_cols: (i+1)*n_cols]))

    for i in range(cluster_frequency.shape[0]):
        freq = cluster_frequency[i]
        k = i % n_cols
        j = i // n_cols

        # ax[j, k].set_extent([config.Data.lon_range[0],
        #                     config.Data.lon_range[1],
        #                     config.Data.lat_range[0],
        #                     config.Data.lat_range[1]])
        ax[j, k].coastlines(zorder=4)
        # TODO resolution='50m', color='black', linewidth=1)

        normalize = mpl.colors.Normalize(vmin=min_max[j, 0],
                                         vmax=min_max[j, 1])

        lons_grid, lats_grid = np.meshgrid(config.Data.all_lons,
                                           config.Data.all_lats)
        data_grid = match_loc_data_map_data(config, freq)
        # Compute map projection coordinates.
        # TODO does this work now? threw error too many values to unpack
        # - works locally
        pcm = ax[j, k].pcolormesh(lons_grid, lats_grid, data_grid,
                                  cmap=color_map, norm=normalize,
                                  transform=cartopy.crs.PlateCarree(),
                                  zorder=3)
        # Colorbar below each row
        if k == n_cols-1 or i == cluster_frequency.shape[0]-1:
            fig.colorbar(pcm, ax=ax[j, :], shrink=0.6, location='bottom',
                         label=c_label, pad=0.001 + 0.05*2/n_rows)
        # fig.colorbar(pcm, ax=ax[j, k], shrink=0.6, location='bottom',
        #              label=c_label)

        # Add cluster profile ID tag
        txt = '${}$'.format(int(i+1))
        cmap = plt.get_cmap("gist_ncar")
        if config.Clustering.n_clusters > 25:
            if i % 2 == 1:
                if config.Clustering.n_clusters % 2 == 1:
                    shift = -1
                else:
                    shift = 0
                i_c = -i + shift
            else:
                i_c = i
        else:
            i_c = i
        clrs = cmap(np.linspace(0.03, 0.97, config.Clustering.n_clusters))
        ax[j, k].plot(0.1, 0.1, 'o', mfc=clrs[i_c], alpha=1, ms=14, mec='k',
                      transform=ax[j, k].transAxes, zorder=4)  # "white"
        ax[j, k].plot(0.1, 0.1, marker=txt, alpha=1, ms=7, mec='k',
                      transform=ax[j, k].transAxes, zorder=4)
        # TODO Add matching background color cluster name -> mfc to power curves colors
    if not config.Plotting.plots_interactive:
        plt.savefig(config.IO.plot_output
                    .format(title='cluster_map_projections_rel_to_{}'
                            .format(tag)))


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
        plt.savefig(config.IO.plot_output.format(title='map_max_cluster_id'))


if __name__ == '__main__':
    from ..config import Config
    config = Config()
    plot_location_map(config)
    plt.show()
