from .wind_profile_clustering.clustering import Clustering
from .power_production.power_production import PowerProduction


class ChainAWERA(Clustering, PowerProduction):
    def __init__(self, config):
        """Initialise Clustering and Production classes."""
        super().__init__(config)

    def run(self):
        make_freq = self.config.Clustering.make_freq_distr
        if make_freq:
            setattr(self.config.Clustering, 'make_freq_distr', False)
        self.run_clustering()

        self.run_curves()

        if make_freq:
            setattr(self.config.Clustering, 'make_freq_distr', True)
            self.get_frequency()

#    def evaluate(config):
#        print('To be done :D')
#        # include inheritance from eval?
#        #aep_map(config)
#        # !!! then: eval
#        # - validation
#        # - result plots
#        # TODO make dependent on config what eval to do -> eval class

