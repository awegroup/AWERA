# AWERA - AWE Resource Analysis 


## Installing and running the code
### Creating the conda environment
To create the conda environment run
```commandline
conda create -c conda-forge -p [path/envName] --file requirements.txt
```

Then activate the environment to be able to work with AWERA.

### Run AWERA
There are a few example scripts on how to run AWERA. The structure is always: import, initialise with configuration (Config() class) and call functions as needed. 

## Components
If only parts of the toolchain are needed, other independent parts can be excluded from the import in AWERA/__init__.

### Wind Resource Analysis 
(See README in folder for now.)

### Wind Profile Clustering
Developed from (https://github.com/markschelbergen/wind-profile-clustering)

### QSM Power Production 
Using Kitepower V3 prototype specifications (or kitepower 100kW or kitepower 500kW), a quasi-steady model simulation is run for a given wind profile. Power curve and single profile power production functionality is available. 

### Evaluation
