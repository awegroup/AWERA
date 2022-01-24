from .config import Config
config = Config()
if not config.Plotting.plots_interactive:
    import matplotlib as mpl
    mpl.use('Pdf')
    print('Initial config with non-interactive plots'
          ' - setting "Pdf" mode in matplotlib.')
    # TODO this rules out local config
    # read from program sarting dir if possible, not interpreted
    # -> import settings
    # read from config-yaml in AWERA
    # read program starting dir again
    # interpret this time
from . import resource_analysis
from . import wind_profile_clustering
from . import power_production
from . import eval
from . import utils
from . import validation
from .awera import ChainAWERA
