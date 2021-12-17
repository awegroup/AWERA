# Wind-profile-clustering

This repository contains the Python code for analysing vertical wind profile patterns in any dataset containing time series of wind speeds and directions for multiple altitudes. This analysis is the backbone of the [Clustering wind profile shapes to estimate airborne wind energy production](https://doi.org/10.5194/wes-5-1097-2020) paper, which has been published in Wind Energy Science.

The code has been originally developed for analysing the Dutch offshore wind atlas (DOWA) dataset. The DOWA file reading functionalities are compatible with the [time series files from 2008-2017 at 10-600 meter height at individual 2,5 km grid location](https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=dowa_netcdf_ts_singlepoint&x-dataset-version=1). Additionally, file reading functionalities are provided for the raw output files of the Leosphere WindCube v.2.1.8 lidar. Measurements of this machine at a location near Köln in Germany for the first three months of 2020 are provided by GWU Umwelttechnik and are under analysis by an airborne wind energy (AWE) resource consortium, which aims for developing AWE system design load case standards. A request to publish the hour-averaged measurement in this repository is pending. 

## Installing the environment and running the code

The code is tested in an Anaconda environment with Python 3.9.1, which can be created using the lower command:

```
conda create --name [env_name] --file requirements.txt python=3.9.1
```
replacing [env_name] by a name of your choice. Download the DOWA files of the desired location. Point with the `data_dir` variable in **dowa.py** to the download directory. In `main` of **wind_profile_clustering.py** change the grid point coordinates that are passed to the `read_data` function to those of the downloaded location. Activate the new environment:

```commandline
conda activate [env_name]
```

Finally, run the script to perform the clustering:

```commandline
python wind_profile_clustering.py
```
The export of the clustering results is performed by running the script:
```commandline
python export_profiles_and_probabilit.py -p
```
When cut-in and cut-out wind speeds are given, the respective frequency distributions are exported via
```commandline
python export_profiles_and_probabilit.py -f
```
