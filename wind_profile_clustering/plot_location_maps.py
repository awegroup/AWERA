import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.cbook import MatplotlibDeprecationWarning
from mpl_toolkits.basemap import Basemap

import pickle

import warnings

from config_clustering import file_name_cluster_labels, n_clusters

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# General plot settings.
map_resolution = 'i' # Options for resolution are c (crude), l (low), i (intermediate), h (high), f (full) or None


#Europe map 
map_lons = [-20, 20]
map_lats = [65, 30]

# Prepare the general map plot.
with open(file_name_cluster_labels, 'rb') as f:labels_file = pickle.load(f)
labels = labels_file['labels [-]']
n_samples = len(labels)
normalisation_values = labels_file['normalisation value [-]']
locations = labels_file['locations']
n_samples_per_loc = labels_file['n_samples_per_loc']

# Evaluate locations data
loc_data_cluster = np.zeros(len(locations))
max_cluster_found = np.zeros(n_clusters)
for i_loc in range(len(locations)):
    loc_labels = labels[n_samples_per_loc*i_loc:n_samples_per_loc*(i_loc+1)]
    cluster_frequency = np.zeros(n_clusters)
    for i_cluster in range(n_clusters):
        cluster_frequency[i_cluster] = sum(loc_labels == i_cluster+1)
    max_cluster = max(zip(cluster_frequency, np.arange(n_clusters)+1), key = lambda t: t[0])[1]
    loc_data_cluster[i_loc] = max_cluster
    max_cluster_found[max_cluster-1] = 1
    
#print(loc_data_cluster)

# Compute relevant map projection coordinates
lats = [lat for lat,_ in locations]
lons = [lon for _,lon in locations]

cm=1/2.54
fig = plt.figure(figsize=(13*cm, 14.5*cm))
ax = fig.add_subplot(111)

plt.title("Cluster mapping")

map_plot = Basemap(projection='merc', llcrnrlon=np.min(map_lons), llcrnrlat=np.min(map_lats), urcrnrlon=np.max(map_lons),
                   urcrnrlat=np.max(map_lats), resolution=map_resolution, ax=ax)
grid_x, grid_y = map_plot(lons, lats)  # Compute map projection coordinates.

if sum(max_cluster_found) <= 8:
    color_map = plt.get_cmap('Dark2')
elif sum(max_cluster_found) <= 20:
    color_map = plt.get_cmap('tab20')
else: 
    color_map = plt.get_cmap('nipy_spectral') #continuous  #TODO define custom discrete color map depending on found clusters

#map fpund max clusters to cluster id
max_cluster = [i+1 for i, m in enumerate(max_cluster_found) if m == 1]

loc_data = [max_cluster.index(c)+1 for c in loc_data_cluster]

normalize = mpl.colors.Normalize(vmin=1, vmax=len(max_cluster))

map_plot.scatter(grid_x, grid_y, c=color_map(normalize(loc_data)))

map_plot.drawcoastlines(linewidth=.4)
#map_plot.fillcontinents(color='#CCCCCC')

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=color_map, norm=normalize, ticks=np.arange(len(max_cluster))+1, label='Cluster') #
cbar.set_ticklabels(max_cluster)

plt.show()



