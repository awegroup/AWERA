from ..wind_profile_clustering.export_profiles_and_probability import export_profiles_and_probability
from ..power_production.power_curves import get_power_curves
#from ..power_production.aep_map import aep_map

def processing(config):
    make_freq = config.Clustering.make_freq_distr
    if make_freq:
        setattr(config.Clustering, 'make_freq_distr', False)
        export_profiles_and_probability(config)

    get_power_curves(config)

    if make_freq:
        setattr(config.Clustering, 'make_freq_distr', True)
        reset_profiles, reset_labels = (False, False)
        if config.Clustering.make_profiles:
            setattr(config.Clustering, 'make_profiles', False)
            reset_profiles = True
        if config.Clustering.predict_labels:
            setattr(config.Clustering, 'predict_labels', False)
            reset_labels = True
        export_profiles_and_probability(config)
        if reset_profiles:
            setattr(config.Clustering, 'make_profiles', True)
        if reset_labels:
            setattr(config.Clustering, 'predict_labels', True)

def evaluate(config):
    aep_map(config)
    # !!! then: eval
    # - validation
    # - result plots
    # TODO make dependent on config what eval to do


def run_full(config):
    processing(config)
    #evaluate(config)
