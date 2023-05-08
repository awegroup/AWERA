import sys

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import copy
from .qsm import TractionPhase, TractionPhaseHybrid, NormalisedWindTable1D,\
    SteadyStateError, OperationalLimitViolation, PhaseError, LogProfile, \
    SystemProperties

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
                             bounds=[None],
                             raise_errors=True,
                             sys_props=sys_props_v3, cycle_sim_settings=None):
        # ref wind speed = 1 - No scaling of non-normalised wind profiles
        # Starting control parameters from mean of ~180k optimizations
        # unbiased (starting controls independent of clustering)

        if oc is None:
            # Create optimization wind environment
            env_state = self.create_environment(wind_u, wind_v, heights)
            env_state.set_reference_wind_speed(ref_wind_speed)
            ref_wind_speed = env_state.calculate_wind(self.ref_height)
            oc = self.create_optimizer(env_state, ref_wind_speed,
                                       sys_props=sys_props,
                                       cycle_sim_settings=cycle_sim_settings,
                                       bounds=bounds)

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

    def create_optimizer(self, env_state, ref_wind_speed,
                         sys_props=sys_props_v3,
                         reduce_x=np.array([0, 1, 2, 3, 5]),
                         cycle_sim_settings=None,
                         print_details=False,
                         bounds=[None]):
        # TODO use this in power_curves
        phi_ro = 13 * np.pi / 180.
        chi_ro = 100 * np.pi / 180.
        if cycle_sim_settings is None:
            # Cycle simulation settings for different phases of the power curves.
            cycle_sim_settings = {
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
        if True:  # ref_wind_speed > 7:
            cycle_sim_settings['cycle']['traction_phase'] = \
                TractionPhaseHybrid

        if not isinstance(sys_props, SystemProperties):
            sys_props = SystemProperties(sys_props)

        oc = OptimizerCycle(cycle_sim_settings, sys_props,
                            env_state, reduce_x=reduce_x,  # , 5]),
                            print_details=print_details,
                            bounds=bounds)
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

    def single_power_curve(self,
                           wind_speeds,
                           x0=[4100., 850., 0.5, 240., 200.0, 1],
                           sys_props=sys_props_v3,
                           cycle_sim_settings=None,
                           plot_output_file=None,
                           env_state=None,
                           ref_height=10,
                           return_optimizer=False,
                           print_details=False):
        # Define optimizer
        if env_state is None:
            # Create environment object.
            env_state = LogProfile()
            env_state.set_reference_height(ref_height)
            env_state.set_reference_wind_speed(wind_speeds[0])

        reduce_x = np.array([0, 1, 2, 3])
        oc = self.create_optimizer(env_state, wind_speeds[0],
                                   sys_props=sys_props,
                                   reduce_x=reduce_x,
                                   cycle_sim_settings=
                                   copy.deepcopy(cycle_sim_settings),
                                   bounds=copy.deepcopy(
                                       self.config.Power.bounds),
                                   print_details=print_details)
        if wind_speeds[0] <= 7:
            oc_g_7 = self.create_optimizer(env_state, 8,
                                           sys_props=sys_props,
                                           reduce_x=np.array([0, 1, 2, 3]),
                                           cycle_sim_settings=
                                           copy.deepcopy(cycle_sim_settings),
                                           bounds=copy.deepcopy(
                                               self.config.Power.bounds),
                                           print_details=print_details)
        else:
            oc_g_7 = oc
        # Configuration of the sequential optimizations for which is
        # differentiated between the wind speed ranges
        # bounded above by the wind speed of the dictionary key. If dx0
        # does not contain only zeros, the starting point
        # of the new optimization is not the solution of the
        # preceding optimization.
        op_seq = {
            7.: {'power_optimizer': oc,
                 'dx0': np.array([0., 0., 0., 0., 0., 0.])},
            17.: {'power_optimizer': oc_g_7,
                  'dx0': np.array([0., 0., 0., 0., 0., 0.])},
            np.inf: {'power_optimizer': oc_g_7,
                     'dx0': np.array([0., 0., 0.1, 0., 0., 0.])},
            }

        reduce_x = np.array([0, 1, 3, 5])
        oc_p = self.create_optimizer(env_state, wind_speeds[0],
                                   sys_props=sys_props,
                                   reduce_x=reduce_x,
                                   cycle_sim_settings=
                                   copy.deepcopy(cycle_sim_settings),
                                   bounds=copy.deepcopy(
                                       self.config.Power.bounds),
                                   print_details=print_details)
        if wind_speeds[0] <= 7:
            oc_g_7_p = self.create_optimizer(env_state, 8,
                                           sys_props=sys_props,
                                           reduce_x=np.array([0, 1, 3, 5]),
                                           cycle_sim_settings=
                                           copy.deepcopy(cycle_sim_settings),
                                           bounds=copy.deepcopy(
                                               self.config.Power.bounds),
                                           print_details=print_details)
        else:
            oc_g_7_p = oc_p
        op_seq_powering = {
            7.: {'power_optimizer': oc_p,
                 'dx0': np.array([0., 0., 0., 0., 0., 0.])},
            17.: {'power_optimizer': oc_g_7_p,
                  'dx0': np.array([0., 0., 0., 0., 0., 0.])},
            np.inf: {'power_optimizer': oc_g_7_p,
                     'dx0': np.array([0., 0., 0.1, 0., 0., 0.])},
            }

        # Start optimizations.
        pc = PowerCurveConstructor(wind_speeds, print_details=print_details)
        setattr(pc, 'plots_interactive',
                self.config.Plotting.plots_interactive)
        if plot_output_file is None:
            plot_output_file = self.config.IO.training_plot_output
        setattr(pc, 'plot_output_file', plot_output_file)

        pc.run_predefined_sequence(op_seq, x0, depowering_seq=op_seq_powering)
        if return_optimizer:
            return pc, oc
        else:
            return pc

    def as_input_profile(self, heights, u, v,
                         do_scale=True):
        if self.config.General.write_output:
            output_file = self.config.IO.profiles
        else:
            output_file = None
        profile, scale_factors = export_wind_profile_shapes(
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
        print('Operational limits estimated.')
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

    def compare_kpis(self, pcs, compare_profiles=None):
        """Plot changing performance indicators with wind speed for all pcs."""
        compare_kpis(self.config, pcs, compare_profiles=compare_profiles)

    def plot_trajectories(self, pc, i_profile=None, circle_radius=200):
        """Plot trajectories from previously generated power curve."""
        if i_profile is None:
            plot_info = 'single_profile'
        else:
            plot_info = '_profile_{}'.format(i_profile)
        i_v_max = len(pc.wind_speeds) - 1
        wind_speed_ids = [i for i in [0, 9, 18, 33, 48, 64] if i <= i_v_max]

        pc.plot_optimal_trajectories(wind_speed_ids=wind_speed_ids,
                                     plot_info=plot_info,
                                     circle_radius=circle_radius
                                     )
        plt.gcf().set_size_inches(5.5, 3.5)
        plt.subplots_adjust(top=0.99, bottom=0.1, left=0.12, right=0.65)
        pc.plot_optimization_results(plot_info=plot_info)

    def plot_power_curves(self,
                          pcs=None,
                          n_profiles=None,
                          labels=None,
                          lim=[3, 30], # was [5, 21]
                          save_plot=True,
                          plot_full_electrical=False,
                          speed_at_op_height=False):
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

        if speed_at_op_height:
            tag = ''
        else:
            tag = '_ref'

        no_loss = [True, False]
        loss_labels = ['Mechanical Power', 'Electrical Power']
        loss_styles = ['solid', 'dashdot']
        for i_profile in range(n_profiles):
            for loss_i, use_mech in enumerate(no_loss):
                if n_profiles > 1 and not plot_full_electrical and loss_i > 0:
                    # TODO this always plot electrical for single profile
                    continue
                if n_profiles > 1 and plot_full_electrical and loss_i == 0:
                    # TODO this only plots electrical for mult profiles if el.
                    continue
                if not plot_full_electrical:
                    tag = tag + 'mech'
                if n_profiles > 1:
                    cmap = plt.get_cmap("gist_ncar")
                    if n_profiles > 25:
                        if i_profile % 2 == 1:
                            if n_profiles % 2 == 1:
                                shift = -1
                            else:
                                shift = 0
                            i_c = - i_profile + shift
                        else:
                            i_c = i_profile
                    else:
                        i_c = i_profile
                    clrs = cmap(np.linspace(0.03, 0.97, n_profiles))
                    color = clrs[i_c]
                else:
                    color = 'orangered'
                if pcs is None:
                    pc = self.read_curve(i_profile=i_profile+1,
                                         return_constructor=True)
                else:
                    # Get power and wind speeds from pc input
                    pc = pcs[i_profile]
                if isinstance(pc, PowerCurveConstructor):
                    wind_speeds, power = pc.curve(return_mech=use_mech)
                    if speed_at_op_height:
                        wind_speeds = [kpis[
                            'wind_speed_at_avg_traction_height']
                            for kpis in pc.performance_indicators]
                        power = np.array([x for _, x in sorted(zip(wind_speeds, power))])
                        wind_speeds = np.array(sorted(wind_speeds))
                else:
                    if not plot_full_electrical and loss_i > 0:
                        continue
                    wind_speeds, power = pc[0], pc[1]
                    if speed_at_op_height:
                        print('Wind Speed input interpreted'
                              ' as at operational height')

                # Plot power
                if labels is None:
                    label = str(i_profile+1)
                elif isinstance(labels, list):
                    label = labels[i_profile]
                else:
                    label = labels
                if n_profiles == 1:
                    # Single labeling
                    label = loss_labels[loss_i] + ' ' + label
                elif loss_i > 0 and not plot_full_electrical and n_profiles > 1:
                    # No labels - no legend entries for multple profile plots
                    label = ''
                    # TODO is this needed? :
                    # ax_pcs.plot(wind_speeds[0], power[0]/1000*loss_f,
                    #             label=loss_labels[loss_i],
                    #             linestyle=loss_styles[loss_i], marker='.')
                if n_profiles > 17:
                    # Cycle linestyles fpr better readability
                    lines = ["-", "-", "--", "--", "-.", "-.", ":", ":",
                             (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5))]
                    linestyle = lines[(i_profile+1) % 10]
                else:
                    linestyle = loss_styles[loss_i]
                ax_pcs.plot(wind_speeds, power/1000, label=label,
                            linestyle=linestyle, zorder=2,
                            color=color)

            if n_profiles > 17 and \
                    ((i_profile+1) % 10 == 0 or i_profile == n_profiles - 1) \
                    and save_plot:
                # Plot every 10 popwer curves together
                if speed_at_op_height:
                    ax_pcs.set_xlabel('$v_{w, @ avg. traction height}$ [m/s]')
                else:
                    ax_pcs.set_xlabel(('$v_{w,' +
                                       str(self.config.General.ref_height)
                                       + 'm}$ [m/s]'))
                ax_pcs.set_ylabel('Mean Cycle Power [kW]')

                ax_pcs.set_xlim(lim)
                print(lim)

                plt.legend()
                # Save plot
                title = 'power_curve_up_to_{}{}'.format(i_profile+1, tag)
                plt.savefig(self.config.IO.plot_output.format(title=title))

                fig, ax_pcs = plt.subplots(1, 1)
                ax_pcs.grid()

        if speed_at_op_height:
            ax_pcs.set_xlabel('$v_{w, @ avg. traction\ height}$ [m/s]')
        else:
            ax_pcs.set_xlabel(('$v_{w,' +
                               str(self.config.General.ref_height)
                               + 'm}$ [m/s]'))
        ax_pcs.set_ylabel('Mean Cycle Power [kW]')

        ax_pcs.set_xlim(lim)
        print(lim)
        if save_plot:
            plt.legend()
            # Save plot
            title = 'power_curve' + tag
            plt.savefig(self.config.IO.plot_output.format(title=title))
        # TODO automatise limits: min, round, ...?, make optional
        # plt.show()

    def plot_cut_wind_speeds(self, limit_estimates=None, limits_refined=None):
        """Plot power curve(s) of previously generated power curve(s)."""
        fig, ax = plt.subplots(1, 1)
        ax.grid()

        if limit_estimates is None:
            limit_estimates = self.read_limits()
        if limits_refined is None:
            limits_refined = self.read_limits(refined=True)

        limits = [limit_estimates, limits_refined]
        limit_labels = ['estimated', 'refined']
        limit_styles = ['solid', 'dashdot']
        limit_colors = ['darkcyan', 'orangered']
        for lim_i, lim in enumerate(limits):
            cut_in = lim['vw_100m_cut_in']
            cut_out = lim['vw_100m_cut_out']
            x = list(range(1, len(cut_in)+1))

            # Plot cut wind speeds
            label = limit_labels[lim_i]

            ax.plot(x, cut_in, label=label,
                    linestyle=limit_styles[lim_i], zorder=2,
                    color=limit_colors[lim_i])
            ax.plot(x, cut_out, label='',
                    linestyle=limit_styles[lim_i], zorder=2,
                    color=limit_colors[lim_i])

        ax.set_xlabel(('Profile Id'))
        ax.set_ylabel('Cut Wind Speeds [m/s]')

        plt.legend(loc='center right', bbox_to_anchor=(0.98, 0.25))
        # Save plot
        title = 'cut_in_out_range_profiles'
        if not self.config.Plotting.plots_interactive:
            plt.savefig(self.config.IO.plot_output.format(title=title))

    def plot_scaled_cut_wind_speed_limits(self,
                                          input_limits=None,
                                          input_profiles=None,
                                          limits_type='estimates'):

        # TODO include descrition of inpuf profiles
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 3), sharey=True)
        plt.subplots_adjust(top=0.92, bottom=0.164, left=0.11,
                            right=0.788, wspace=0.13)

        # TODO format with reference height everywhere

        if input_limits is None:
            input_limits = self.read_limits(refined=(limits_type == 'refined'))
        if input_profiles is None:
            input_profiles = self.read_profiles()
        # 1 height column, 3 columns each profile (u,v,scale factor)

        n_profiles = int((input_profiles.shape[1]-1)/3)
        # TODO option to read arbitrary profile, n_profiles: len(profiles)
        for i_profile in range(1, n_profiles+1):
            cmap = plt.get_cmap("gist_ncar")
            i_profile = i_profile - 1
            if n_profiles > 25:
                if i_profile % 2 == 1:
                    if n_profiles % 2 == 1:
                        shift = -1
                    else:
                        shift = 0
                    i_c = - i_profile + shift
                else:
                    i_c = i_profile
            else:
                i_c = i_profile
            i_profile = i_profile + 1
            clrs = cmap(np.linspace(0.03, 0.97, n_profiles))
            color = clrs[i_c]
            # TODO logging? / timing info print('Profile {}'.format(i_profile))
            from .power_curves import create_environment
            env = create_environment(input_profiles, i_profile)

            # Get cut-in wind speed.
            vw_cut_in = input_limits['vw_100m_cut_in'][i_profile-1]
            vw_cut_out = input_limits['vw_100m_cut_out'][i_profile-1]

            # Plot the wind profiles corresponding to the wind speed operational
            # limits and the profile shapes.
            env.set_reference_height(
                self.config.General.ref_height)
            env.set_reference_wind_speed(vw_cut_in)
            plt.sca(ax[0])
            env.plot_wind_profile(color=color)

            env.set_reference_wind_speed(vw_cut_out)
            plt.sca(ax[1])
            if n_profiles > 20 and i_profile > 8:
                env.plot_wind_profile(color=color)
            else:
                env.plot_wind_profile("{}".format(i_profile), color=color)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.ylabel('')

        ax[0].set_title("Cut-in")
        ax[0].set_xlim([0, None])
        ax[0].set_ylim([0, 400])
        ax[1].set_title("Cut-out")
        ax[1].set_xlim([0, None])
        ax[1].set_ylim([0, 400])
        if not self.config.Plotting.plots_interactive:
            plt.savefig(self.config.IO.training_plot_output.format(
                title='{}_wind_speed_operational_limits'.format(limits_type)))

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