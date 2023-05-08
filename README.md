# AWERA - AWE Resource Analysis 

An extensive discussion of AWERA, its concepts and results can be found in the master's thesis by Lavinia Thimm
http://resolver.tudelft.nl/uuid:ba0c7fb2-baff-4110-9a51-c27a8498663b called The Airborne Wind Energy Resource Analysis Tool AWERA - 
and the paper on the clustering approach for wind profile clustering by Mark Schelbergen. 


## Installing and running the code
### Creating the conda environment
To create the conda environment run
```commandline
conda create -c conda-forge -p [path/envName] --file requirements.txt
```

Then activate the environment to be able to work with AWERA.
### Download ERA5
There is a script in the resource analysis subfolder thatworked for downloading ERA5 data - but as the web structure changed considerably, it is pribaby not up-to-date anymore. NETCDF files are needed for the analysis, of different model levels. For the resource analysis also the surface pressure files are used.

### Run AWERA
There are a few example scripts on how to run AWERA. The structure is always: import, initialise with configuration (Config() class) and call functions as needed. 

Always check the config.yaml for the settings done and changed - any comments on what settings can and can't change. Some manual labor is still required, even if minimal - see generator power in config.yaml.

## Components
If only parts of the toolchain are needed, other independent parts can be excluded from the import in AWERA/__init__.

### Wind Resource Analysis 
(See README in folder for now.) Can partially be run from within AWERA, mostly the evaluation - see run_resource analysis.py

For processing and plotting maps, run within the folder. Check own config file then. Set mpl type to not PDF for this, oherwise plt.show() and plotting maps don't work properly.  

### Wind Profile Clustering
Developed from (https://github.com/markschelbergen/wind-profile-clustering) As all of AWERA, also the wind profile clustering is based on Mark's work - Thank you! Check out his repo if you want to see the origin. 
And his paper as well, while you're at it. 

### QSM Power Production 
Using Kitepower V3 prototype specifications (or kitepower 100kW or kitepower 500kW), a quasi-steady model simulation is run for a given wind profile. Power curve and single profile power production functionality is available. 

A full example an be found for different locations, but in general in run_awera_production_8.py for the 8 cluster standard evaluation chain. Comment in/out as required.

Running only the power production can be see in run_power_production.py for a LogProfile wind profile. 

### Evaluation

# References


Schelbergen, M., Kalverla, P. C., Schmehl, R., and Watson, S. J.: Clustering wind profile shapes to estimate airborne wind energy production, Wind Energ. Sci., 5, 1097–1120, https://doi.org/10.5194/wes-5-1097-2020, 2020. 


