from .power_curves import estimate_wind_speed_operational_limits

class PowerProduction:
    def __init__(self, config):  # TODO is this useable?, read_input=True):
        # Set configuration from Config class object
        # TODO not standalone then? own config class object & yaml in production?
        setattr(self, 'config', config)

    def estimate_wind_speed_operational_limits(self,
                                               export_operational_limits=True,
                                               input_profiles=None):
        res = estimate_wind_speed_operational_limits(
            self.config,
            export_operational_limits=True,
            input_profiles=None)
        return res



def run(self):
    # TODO include multiprocessing option here

    if self.config.Power.estimate_cut_in_out:
        # TODO this is estimated every time for all profiles, but
        # also if only one profile is run at a time
        self.estimate_wind_speed_operational_limits()


    if config.Power.make_power_curves:
        if config.Processing.parallel:
            # TODO import not here
            from multiprocessing import Pool
            from tqdm import tqdm
            # TODO tqdm is not useful here - all 8 profile run, no updates
            # until opt is finished
            i_profile = [[i+1] for i in range(config.Clustering.n_clusters)]
            import functools
            funct = functools.partial(generate_power_curves,
                                      config)
            with Pool(config.Processing.n_cores) as p:
                if config.Processing.progress_out == 'stdout':
                    file = sys.stdout
                else:
                    file = sys.stderr
                res = list(tqdm(p.imap(funct, i_profile),
                                total=len(i_profile),
                                file=file))
            # Interpret res: funct returns a list of the result for each
            # process
            pcs = [res_n[0] for res_n in res]
            combine_separate_profile_files(
                config,
                io_file='refined_cut_wind_speeds')
            # TODO remove combined (old) fiiles
        else:
            run_profiles = list(range(config.Clustering.n_clusters))
            pcs = generate_power_curves(
                config,
                run_profiles)
        compare_kpis(config, pcs)

# TODO include option for brute forcing