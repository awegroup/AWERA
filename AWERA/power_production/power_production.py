import sys

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import copy
from .qsm import TractionPhase, TractionPhaseHybrid, NormalisedWindTable1D,\
    SteadyStateError, OperationalLimitViolation, PhaseError

from .kitepower_kites import sys_props_v3
from .cycle_optimizer import OptimizerCycle, OptimizerError

from .power_curves import estimate_wind_speed_operational_limits,\
    generate_power_curves, combine_separate_profile_files, compare_kpis

from .power_curve_constructor import PowerCurveConstructor


from ..utils.wind_profile_shapes import export_wind_profile_shapes

# TODO include option for brute forcing: run_single, run_curve -> own class,
# power production inherits
# include production cycle settings here - extract later in power curves?


class SingleProduction:
    def __init__(self, ref_height=100):
        """Initialise using from Config class object."""
        # TODO not standalone then? own config class object
        # & yaml in production?
        super().__init__()
        setattr(self, 'ref_height', ref_height)  # m
        setattr(self, 'prod_errors', (SteadyStateError,
                                      PhaseError,
                                      OperationalLimitViolation,
                                      OptimizerError))

    def single_profile_power(self, heights, wind_u, wind_v,
                             x0=[4100., 850., 0.5, 240., 200.0],
                             ref_wind_speed=1,
                             oc=None,
                             return_optimizer=False,
                             raise_errors=True):
        # ref wind speed = 1 - No backscaling of non-normalised wind profiles
        # Starting control parameters from mean of ~180k optimizations
        # unbiased (starting controls independent of clustering)

        if oc is None:
            # Create optimization wind environment
            env_state = self.create_environment(wind_u, wind_v, heights)
            env_state.set_reference_wind_speed(ref_wind_speed)
            ref_wind_speed = env_state.calculate_wind(self.ref_height)
            oc = self.create_optimizer(env_state, ref_wind_speed)

        # Optimize individual profile
        try:
            x0_opt, x_opt, op_res, cons, kpis = oc.run_optimization(x0)
        except self.prod_errors as e:
            print("Clustering output point evaluation finished with a "
                  "simulation error: {}".format(e))
            if raise_errors:
                raise e
            else:
                x0_opt, x_opt, op_res, cons, kpis = \
                    x0, [-1]*len(x0), -1, -1, -1
        # TODO use? optHist = History(optimizer_history_file_name)
        # print(optHist.getValues(major=True, scale=False,
        #                         stack=False, allowSens=True)['isMajor'])
        if return_optimizer:
            return x0_opt, x_opt, op_res, cons, kpis, oc
        else:
            return x0_opt, x_opt, op_res, cons, kpis

    def create_environment(self, wind_profile_u, wind_profile_v, heights):
        """Use wind profile shape to create the environment object."""
        env = NormalisedWindTable1D()
        env.heights = list(heights)
        env.normalised_wind_speeds = list(
            (wind_profile_u**2 + wind_profile_v**2)**.5)
        return env


    def create_optimizer(self, env_state, ref_wind_speed):
        # TODO use this in power_curves
        phi_ro = 13 * np.pi / 180.
        chi_ro = 100 * np.pi / 180.
        # Cycle simulation settings for different phases of the power curves.
        cycle_sim_settings_pc = {
            'cycle': {
                'traction_phase': TractionPhase,
                'include_transition_energy': False,
            },
            'retraction': {},
            'transition': {
                'time_step': 0.25,
            },
            'traction': {
                'azimuth_angle': phi_ro,
                'course_angle': chi_ro,
            },
        }

        # Modify settings in accordance with clustering settings
        # TODO test what happen if always use Hybrid?
        # TODO this refers to the clustering v_100m classification of
        # settings, change/ remove difference?
        if ref_wind_speed > 7:
            cycle_sim_settings_pc['cycle']['traction_phase'] = \
                TractionPhaseHybrid
        oc = OptimizerCycle(cycle_sim_settings_pc, sys_props_v3,
                            env_state, reduce_x=np.array([0, 1, 2, 3]))
        # !!! optional reduce_x, settingsm ref wind speed -> also in class?
        if ref_wind_speed <= 7:
            oc.bounds_real_scale[2][1] = 30*np.pi/180.
        return oc


class PowerProduction(SingleProduction):
    def __init__(self, config):
        """Initialise using from Config class object."""
        # TODO not standalone then? own config class object
        # & yaml in production?
        super().__init__(ref_height=config.General.ref_height)
        setattr(self, 'config', config)

    def as_input_profile(self, heights, u, v,
                         do_scale=True):
        if self.config.General.write_output:
            output_file = self.config.IO.profiles
        else:
            output_file = None
        profile = export_wind_profile_shapes(
            heights,
            u, v,
            output_file=output_file,
            do_scale=do_scale,
            ref_height=self.config.General.ref_height)
        return profile

    def estimate_wind_speed_operational_limits(self,
                                               input_profiles=None):
        """Estimate the cut-in/out wind speeds for each wind profile shape.

        These wind speeds are refined when determining the power curves.
        """
        # TODO include description of input profiles
        res = estimate_wind_speed_operational_limits(
            self.config,
            export_operational_limits=self.config.General.write_output,
            input_profiles=input_profiles)
        return res

    def make_power_curves(self,
                          input_profiles=None,
                          run_profiles=None,
                          limit_estimates=None):
        """
        Determine power curve(s) of relected run_profiles of input profiles.

        Parameters
        ----------
        input_profiles : pandas DataFrame, optional
            Absolute wind profiles scaled at reference height.
            The default is None, wind profiles are read from file in config.
        run_profiles : list, optional
            Select wind profile id(s) to generate power curves for,
            starting at 1.
            The default is None, all available profiles are run.
        limit_estimates : pandas DataFrame, optional
            Cut-in and cut-out wind speeds matchting the input_profiles.
            The default is None. Default requires estimates of the cut-in
            and cut-out wind speed to be available to read from file in config.

        Returns
        -------
        pcs : list(PowerCurveConstructor)
            Power curves for each wind profile shape.
            Wind speed gives the scaling of the wind speed at reference height.

        """
        if run_profiles is None:
            if input_profiles is None:
                run_profiles = list(
                    range(1, self.config.Clustering.n_clusters+1))
            else:
                run_profiles = list(
                    range(1, int((input_profiles.shape[1]-1)/3) + 1))
                # TODO len( pandas)
        elif not isinstance(run_profiles, list):
            run_profiles = [run_profiles]

        if self.config.Processing.parallel:
            # TODO import not here
            from multiprocessing import Pool
            from tqdm import tqdm
            import functools
            funct = functools.partial(
                generate_power_curves,
                self.config,
                input_profiles=input_profiles,
                limit_estimates=limit_estimates)
            with Pool(self.config.Processing.n_cores) as p:
                if self.config.Processing.progress_out == 'stdout':
                    file = sys.stdout
                else:
                    file = sys.stderr
                res = list(tqdm(p.imap(funct, [[i] for i in run_profiles]),
                                total=len(run_profiles),
                                file=file))
            # Interpret res: funct returns a list of the result for each
            # process
            pcs = []
            for i, res_n in enumerate(res):
                pcs.append(res_n[0][0])
                refined_limits_n = res_n[1]
                if i == 0:
                    refined_limits = copy.deepcopy(refined_limits_n)
                else:
                    for key, val in refined_limits_n.items():
                        refined_limits[key].append(val)

            if run_profiles == \
                    list(range(1, self.config.Clustering.n_clusters+1)):
                self.combine_separate_profile_files(
                    io_file='refined_cut_wind_speeds')
        else:
            pcs, refined_limits = generate_power_curves(
                self.config,
                run_profiles,
                input_profiles=input_profiles,
                limit_estimates=limit_estimates)
        return pcs, refined_limits

    def combine_separate_profile_files(self,
                                       io_file='refined_cut_wind_speeds'):
        combine_separate_profile_files(self.config, io_file=io_file)

    def run_curves(self,
                   input_profiles=None,
                   run_profiles=None):
        # TODO make Doctring with more input
        """Estimate cut wind speed, generate all power curves set in config."""
        if self.config.Power.estimate_cut_in_out:
            # TODO this is estimated every time for all profiles, but
            # also if only one profile is run at a time
            limit_estimates = self.estimate_wind_speed_operational_limits(
                input_profiles=input_profiles)

            if self.config.Power.make_power_curves:
                pcs, limit_refined = self.make_power_curves(
                    input_profiles=input_profiles,
                    run_profiles=run_profiles,
                    limit_estimates=limit_estimates)

                self.compare_kpis(pcs)
                return pcs, limit_refined
            else:
                return limit_estimates

        elif self.config.Power.make_power_curves:
            pcs = self.make_power_curves(
                input_profiles=input_profiles,
                run_profiles=run_profiles)
            self.compare_kpis(pcs)
            return pcs

    def compare_kpis(self, pcs):
        """Plot changing performance indicators with wind speed for all pcs."""
        compare_kpis(self.config, pcs)

    def plot_trajectories(self, pc, i_profile=None):
        """Plot trajectories from previously generated power curve."""
        if i_profile is None:
            plot_info = 'single_profile'
        else:
            plot_info = '_profile_{}'.format(i_profile)
        pc.plot_optimal_trajectories(wind_speed_ids=[0, 9, 18, 33, 48, 64],
                                     plot_info=plot_info)
        plt.gcf().set_size_inches(5.5, 3.5)
        plt.subplots_adjust(top=0.99, bottom=0.1, left=0.12, right=0.65)
        pc.plot_optimization_results(plot_info=plot_info)

    def plot_power_curves(self,
                          pcs=None,
                          n_profiles=None,
                          labels=None,
                          lim=[5, 21]):
        """Plot power curve(s) of previously generated power curve(s)."""
        fig, ax_pcs = plt.subplots(1, 1)
        ax_pcs.grid()

        if pcs is not None:
            if isinstance(pcs, list):
                n_profiles = len(pcs)
            else:
                pcs = [pcs]
                n_profiles = 1
        elif n_profiles is None:
            try:
                n_profiles = self.config.Clustering.n_clusters
            except AttributeError:
                raise ValueError('No valid option to get '
                                 'power curve input found.')

        for i_profile in range(n_profiles):
            if pcs is None:
                df_profile = pd.read_csv(self.config.IO.power_curve.format(
                    i_profile=i_profile+1, suffix='csv'), sep=";")
                wind_speeds = df_profile['v_100m [m/s]']
                power = df_profile['P [W]']
            else:
                # Get power and wind speeds from pc input
                pc = pcs[i_profile]
                if isinstance(pc, PowerCurveConstructor):
                    wind_speeds, power = pc.curve()
                else:
                    wind_speeds, power = pc[0], pc[1]
            # Plot power
            if labels is None:
                label = i_profile+1
            elif isinstance(labels, list):
                label = labels[i_profile]
            else:
                label = labels

            ax_pcs.plot(wind_speeds, power/1000, label=label)

        ax_pcs.set_xlabel(('$v_{w,' +
                           str(self.config.General.ref_height) + 'm}$ [m/s]'))
        ax_pcs.set_ylabel('Mean cycle Power [kW]')

        ax_pcs.set_xlim(lim)
        # TODO automatise limits: min, round, ...?, make optional
        # plt.show()

    def read_curve(self,
                   i_profile=None,
                   file_name=None,
                   return_constructor=False):
        """Read power curve pickle file for either profile id or file name."""
        if file_name is None:
            if i_profile is not None:
                if return_constructor:
                    suffix = 'pickle'
                else:
                    suffix = 'csv'
                file_name = self.config.IO.power_curve.format(
                    i_profile=i_profile,
                    suffix=suffix)
            else:
                raise ValueError('No profile id (i_profile) or file_name given')

        if return_constructor:
            pc = PowerCurveConstructor(None)
            setattr(pc, 'plots_interactive',
                    self.config.Plotting.plots_interactive)
            setattr(pc, 'plot_output_file',
                    self.config.IO.training_plot_output)
            pc.import_results(file_name)
        else:
            pc = pd.read_csv(file_name, sep=";")
        return pc

    def read_profiles(self, file_name=None, sep=';'):
        """
        Read existing profiles from csv-file to pandas DataFrame.

        Parameters
        ----------
        file_name : String, optional
            Filename of the saved profiles. The default is None, profiles
            according to config are read.
        sep : String, optional
            CSV separator in profiles csv file. The default is ';'.
        Returns
        -------
        profiles : pandas DataFrame
            Vertical wind profiles and scale factors.

        """
        if file_name is None:
            file_name = self.config.IO.profiles
        profiles = pd.read_csv(file_name, sep=sep)
        return profiles

    def read_limits(self, file_name=None, sep=',', refined=False):
        """
        Read existing limit estimates from csv-file to pandas DataFrame.

        Parameters
        ----------
        file_name : String, optional
            Filename of the saved estimated limits. The default is None, profiles
            according to config are read.
        sep : String, optional
            CSV separator in limits csv file. The default is ','.
        Returns
        -------
        limits : pandas DataFrame
            Cut-in and cut-out wind speeds DataFrame.

        """
        if file_name is None:
            if refined:
                file_name = self.config.IO.refined_cut_wind_speeds
            else:
                file_name = self.config.IO.cut_wind_speeds
        limits = pd.read_csv(file_name, sep=sep)
        return limits