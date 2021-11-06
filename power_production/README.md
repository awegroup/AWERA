# Production estimation from input wind profile using quasi steady model for AWE soft wing


## Installing and running the code
### Creating the conda environment
To create the conda environment run
```commandline
conda create -p [path/envName] --file requirements.txt
```

#### Install pyoptsparse into the environment
*Full:*
Activate the conda environment.
```commandline
pip install git+https://github.com/mdolab/pyoptsparse@v2.5.1
```
*Light:*
This installs less modules, and does not require swig.
Clone the pyoptsparse git repository. 
Select version via 
```commandline
git checkout v2.5.1
```
Comment out the lines concerning NOMAD and NSGA2 in the `pyoptsparse/pyoptsparse/setup.py` and 
`pyoptsparse/pyoptsparse/__init__.py`.
Activate the conda environment, navigate to the pyoptsparse top level folder, run
```commandline
pip install .
``` 
OR install with OptView to visualize the optimization output:
```commandline
pip install .[optview]
```

The pyoptsparse package should now be available in the environment. 

## Chain connection to wind profile clustering (https://github.com/markschelbergen/wind-profile-clustering)
The configuration set for the clustering can be imported to the aep estimation via creating a symlink
to the config.py file in the clustering repo called config_clustering.py. In the production-estimation folder call:
```commandline
ln -s "path to local clone of clustering-repo"/config.py config_clustering.py
```
The wind profile output from the clustering is needed, so run the `export_profiles_and_probability.py -p` function (-p option, only profiles). Then estimate the cut-in and cut-out velocity in `power_curves.py -c`, rerun clustering (-f option for frequency distributions) then run the `power_curves.py -p` to obtain the power curves for all cluster profiles and a range of absolute velocities (e.g. 5 to 20 m/s at 100m). 
Then the `aep.py` can estimate the annual energy production for the initial samples using the frequency of absolute velocities and the power curves. 
