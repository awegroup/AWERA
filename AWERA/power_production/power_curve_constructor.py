import matplotlib as mpl
import numpy as np
import pickle
import copy
from scipy.stats import truncnorm

from .qsm import SteadyStateError, OperationalLimitViolation, PhaseError

from .cycle_optimizer import OptimizerError

from .utils import flatten_dict

#TODO is this needed here?
#if not plots_interactive:
#    mpl.use('Pdf')
import matplotlib.pyplot as plt


class PowerCurveConstructor:
    def __init__(self, wind_speeds, print_details=False):
        self.wind_speeds = wind_speeds

        self.x_opts = []
        self.x0 = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

        self.print_details = print_details

        self.plots_interactive = False
        self.plot_output_file = '{title}.pdf'

    def run_predefined_sequence(self, seq, x0_start, depowering_seq=None):
        wind_speed_tresholds = iter(sorted(list(seq)))
        vw_switch = next(wind_speed_tresholds)

        x_opt_last, vw_last = None, None
        failed_wind_speeds = []
        n_wind_speeds = len(self.wind_speeds)
        for i, vw in enumerate(self.wind_speeds):
            if self.print_details:
                print('================')
                print('{}: wind speed {}m/s...'.format(i, vw))
            i = i - len(failed_wind_speeds)
            if vw > vw_switch:
                vw_switch = next(wind_speed_tresholds)

            power_optimizer = seq[vw_switch]['power_optimizer']
            # TODO set in settings?

            dx0 = seq[vw_switch].get('dx0', None)
            print('optimise on first: ', power_optimizer.reduce_x)
            if x_opt_last is None:
                x0_next = x0_start
            else:
                x0_next = x_opt_last + dx0*(vw - vw_last)
            if x0_next[2] > power_optimizer.bounds_real_scale[2][1]*0.9\
                    and depowering_seq is not None and vw > 7:
                seq = copy.deepcopy(depowering_seq)
                depowering_seq = None
                power_optimizer = seq[vw_switch]['power_optimizer']
                print('next: switch to depowering', seq)

            print('optimise on: ', power_optimizer.reduce_x)
            power_optimizer.smear_x0 = True
            # TODO log? print("[{}] Processing v={:.2f}m/s".format(i, vw))
            power_optimizer.environment_state.set_reference_wind_speed(vw)
            if self.print_details:
                print('Real scale bounds: ', power_optimizer.bounds_real_scale)
            if i+1 == len(self.wind_speeds):
                dv = 1e-2
            else:
                dv = (self.wind_speeds[i+1]-self.wind_speeds[i])/3
            try:
                print('try run first optimisation ')
                x0_opt, x_opt, op_res, cons, kpis = \
                    power_optimizer.run_optimization(x0_next)
            except (OperationalLimitViolation, SteadyStateError,
                    PhaseError, OptimizerError, FloatingPointError) as e:
                # Include FloatingPointError in except
                # - QSM simulation sometimes runs into this
                # File qsm.py", line 2339, in run_simulation:
                # self.average_power = self.energy / self.time[-1]
                # FloatingPointError: divide by zero encountered
                # in double_scalars
                if self.print_details:
                    print('Error msg in power curve constructor:\n', e)
                    print('-----------------------')
                try:  # Retry for a slightly different wind speed.
                    # TODO log? print('first optimization/simulation ended
                    # in error: {}'.format(e))
                    # TODO log? print('run with varied wind speed:', vw+1e-2)
                    vw = vw + dv
                    print('2: Try wind speed: ', vw)
                    power_optimizer.environment_state.set_reference_wind_speed(
                        vw)
                    x0_opt, x_opt, op_res, cons, kpis = \
                        power_optimizer.run_optimization(x0_next,
                                                         n_x_test=2,  # TODO change again
                                                         second_attempt=True)
                    self.wind_speeds[i] = vw
                except (OperationalLimitViolation, SteadyStateError,
                        PhaseError, OptimizerError, FloatingPointError) as e:
                    if self.print_details:
                        print('{}: wind speed {}m/s...'
                              .format(i + len(failed_wind_speeds), vw))
                        print('Error msg in power curve constructor:\n', e)
                        print('-----------------------')
                    try:  # Retry for a slightly different wind speed.
                        # TODO log? print('first optimization/simulation ended
                        # in error: {}'.format(e))
                        # TODO log? print('run with varied wind speed:')
                        vw = vw + dv
                        print('3: Try wind speed: ', vw)
                        power_optimizer.environment_state\
                            .set_reference_wind_speed(vw)
                        x0_opt, x_opt, op_res, cons, kpis = \
                            power_optimizer.run_optimization(
                                x0_next,
                                n_x_test=2,  # TODO change again
                                second_attempt=True)
                        self.wind_speeds[i] = vw
                    except (OperationalLimitViolation, SteadyStateError,
                            PhaseError, OptimizerError, FloatingPointError
                            ) as e:
                        if self.print_details:
                            print('{}: wind speed {}m/s...'
                                  .format(i + len(failed_wind_speeds), vw))
                            print('Error msg in power curve constructor:\n', e)
                            print('-----------------------')
                        err = e
                        if len(failed_wind_speeds) == n_wind_speeds:
                            self.wind_speeds = []
                            print('Optimisation wind speeds failed at step'
                                  ' {}, wind speed{} m/s'.format(i, vw))
                            # TODO log? print("Optimization sequence stopped
                            # prematurely due to failed optimization. {:.2f}
                            # m/s is
                            # the "
                            #       "highest wind speed for which the
                            # optimization
                            # was successful.".format(self.wind_speeds[-1]))
                            break
                        else:
                            failed_wind_speeds.append(vw)
                            self.wind_speeds = list(np.delete(self.wind_speeds,
                                                              i))
                            continue
            self.x0.append(x0_opt)
            self.x_opts.append(x_opt)
            self.optimization_details.append(op_res)
            self.constraints.append(cons)
            self.performance_indicators.append(kpis)
            x_opt_last = x_opt
            vw_last = vw
            # TODO log? print(self.wind_speeds[:i+1], [kpi['sim_successful']
            # for kpi in self.performance_indicators])
            # TODO log? print(len(self.wind_speeds[:i+1]), sum([kpi['
            # sim_successful'] for kpi in self.performance_indicators]))
        if len(failed_wind_speeds) == n_wind_speeds:
            # First tested wind speed ran into break:
            print('No working solutions for any wind speed found.')
            raise err

    def plot_optimal_trajectories(self,
                                  wind_speed_ids=None,
                                  ax=None,
                                  circle_radius=200,
                                  elevation_line=25*np.pi/180,
                                  plot_info=''):
        if ax is None:
            plt.figure(figsize=(6, 3.5))
            plt.subplots_adjust(right=0.65)
            ax = plt.gca()

        mask = [kpis['sim_successful'] for kpis in self.performance_indicators]
        all_kpis = [kpi
                    for i, kpi in enumerate(self.performance_indicators)
                    if mask[i]]
        # Mask power discontinuities #TODO still necessary?
        p_cycle = np.array([kpis['average_power']['cycle']
                            for kpis in all_kpis])
        # mask_power = p_cycle > 0
        # TODO  mask negative jumps in power
        # - check reason for negative power/ strong jumps
        # p_cycle = p_cycle[mask_power]
        all_kpis = [kpi for i, kpi in enumerate(all_kpis)]  # if mask_power[i]]
        wind_speeds = self.wind_speeds  # [mask_power]

        # masking_counter = 0
        # TODO resolve source of problems
        # while True:
        #     mask_power_disc = [True] + list((np.diff(p_cycle) > -1000))
        #     if sum(mask_power_disc) == len(mask_power_disc):
        #         # No more discontinuities
        #         break
        #     print('Masking {} power runs'.format(len(mask_power_disc)
        #                                          - sum(mask_power_disc)))
        #     p_cycle = p_cycle[mask_power_disc]
        #     all_kpis = [kpi
        #                 for i, kpi in enumerate(all_kpis)
        #                 if mask_power_disc[i]]
        #     wind_speeds = wind_speeds[mask_power_disc]
        #     masking_counter += 1
        # TODO drop?print('TotalMasking {} power runs'.format(masking_counter))

        if wind_speed_ids is None:
            if len(wind_speeds) > 8:
                wind_speed_ids = [int(a)
                                  for a in np.linspace(0,
                                                       len(wind_speeds)-1, 6)]
            else:
                wind_speed_ids = range(len(wind_speeds))

        for i in wind_speed_ids:
            v = wind_speeds[i]
            kpis = all_kpis[i]
            if kpis is None:
                print("No trajectory available for"
                      " {} m/s wind speed.".format(v))
                continue

            x_kite, z_kite = zip(*[(kp.x, kp.z) for kp in kpis['kinematics']])
            # TODO log? print('min x, z: ', min(x_kite), min(z_kite))
            # try:
            #     z_traj = [kp.z for kp in kite_positions['trajectory']]
            # except AttributeError:
            #     z_traj = [np.sin(kp.elevation_angle)*kp.straight_
            # tether_length for kp in kite_positions['trajectory']] #???
            # TODO v_100m or v_ref
            ax.plot(x_kite, z_kite,
                    label="$v_{ref}$="+"{:.1f} ".format(v) + "m s$^{-1}$")

        # Plot semi-circle at constant tether length bound.
        phi = np.linspace(0, 2*np.pi/3, 40)
        x_circle = np.cos(phi) * circle_radius
        z_circle = np.sin(phi) * circle_radius
        ax.plot(x_circle, z_circle, 'k--', linewidth=1)

        # Plot elevation line.
        x_elev = np.linspace(0, 400, 40)
        z_elev = np.tan(elevation_line)*x_elev
        ax.plot(x_elev, z_elev, 'k--', linewidth=1)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_xlim([0, None])
        ax.set_ylim([0, None])
        ax.grid()
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if not self.plots_interactive:
            plt.savefig(self.plot_output_file.format(
                title='optimal_trajectories{}'.format(plot_info)))

    def plot_optimization_results(self,
                                  opt_variable_labels=None,
                                  opt_variable_bounds=None,
                                  tether_force_limits=None,
                                  reeling_speed_limits=None,
                                  plot_info=''):
        assert self.x_opts, "No optimization results available for plotting."

        # mask unsuccessful simulations
        mask = [kpis['sim_successful'] for kpis in self.performance_indicators]
        kpis = [kpi
                for i, kpi in enumerate(self.performance_indicators)
                if mask[i]]
        opt_details = [od
                       for i, od in enumerate(self.optimization_details)
                       if mask[i]]
        xf = [xopt for i, xopt in enumerate(self.x_opts) if mask[i]]
        x0 = [x for i, x in enumerate(self.x0) if mask[i]]
        cons = [c for i, c in enumerate(self.constraints) if mask[i]]

        n_opt_vars = len(xf[0])
        fig, ax = plt.subplots(max([n_opt_vars, 6]), 2, sharex=True,
                               figsize=(15, 15))
        # TODO fig size in pdf too small

        # In the left column plot each optimization variable
        # against the wind speed.
        for i in range(n_opt_vars):
            # Plot optimal and starting points.
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in xf], label='x_opt')
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in x0], 'o',
                          markerfacecolor='None', label='x0')

            if opt_variable_labels:
                label = opt_variable_labels[i]
                ax[i, 0].set_ylabel(label)
            else:
                ax[i, 0].set_ylabel("x[{}]".format(i))

            if opt_variable_bounds is not None:
                ax[i, 0].axhline(opt_variable_bounds[i, 0],
                                 linestyle='--', color='k')
                ax[i, 0].axhline(opt_variable_bounds[i, 1],
                                 linestyle='--', color='k')

            ax[i, 0].grid()
        ax[0, 0].legend()

        # In the right column plot the number of iterations in the upper panel.

        nits = np.array([od['nit'] for od in opt_details])
        ax[0, 1].plot(self.wind_speeds, nits)
        # TODO this is now obsolete / to be done differently
        # mask_opt_failed = np.array([~od['success'] for od in opt_details])
        # ax[0, 1].plot(self.wind_speeds[mask_opt_failed],
        # nits[mask_opt_failed], 'x', label='opt failed')
        # mask_sim_failed = np.array([~kpi['sim_successful'] for kpi in kpis])
        # ax[0, 1].plot(self.wind_speeds[mask_sim_failed],
        # nits[mask_sim_failed], 'x', label='sim failed')
        ax[0, 1].grid()
        # ax[0, 1].legend()
        ax[0, 1].set_ylabel('Optimization iterations [-]')

        # In the second panel, plot the optimal power.
        # TODO this is now obsolete / to be done differently
        # cons_treshold = -.1
        # mask_cons_adhered = np.array([all([c >= cons_treshold for c in con])
        # for con in cons])
        # mask_plot_power = ~mask_sim_failed & mask_cons_adhered
        power = np.array([kpi['average_power']['cycle'] for kpi in kpis])
        # power[~mask_plot_power] = np.nan
        ax[1, 1].plot(self.wind_speeds, power)
        ax[1, 1].grid()
        ax[1, 1].set_ylabel('Mean power [W]')

        # In the third panel, plot the tether force related
        # performance indicators.
        max_force_in = np.array([kpi['max_tether_force']['in']
                                 for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_in,
                      label='max_tether_force.in')
        max_force_out = np.array([kpi['max_tether_force']['out']
                                  for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_out,
                      label='max_tether_force.out')
        max_force_trans = np.array([kpi['max_tether_force']['trans']
                                    for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_trans,
                      label='max_tether_force.trans')
        if tether_force_limits:
            ax[3, 1].axhline(tether_force_limits[0], linestyle='--', color='k')
            ax[3, 1].axhline(tether_force_limits[1], linestyle='--', color='k')
        ax[2, 1].grid()
        ax[2, 1].set_ylabel('Tether force [N]')
        ax[2, 1].legend(loc=2)
        ax[2, 1].annotate(
            'Violation occurring before\nswitch to force controlled',
            xy=(0.05, 0.10), xycoords='axes fraction')

        # Plot reeling speed related performance indicators.
        max_speed_in = np.array([kpi['max_reeling_speed']['in']
                                 for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_in,
                      label='max_reeling_speed.in')
        max_speed_out = np.array([kpi['max_reeling_speed']['out']
                                  for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_out,
                      label='max_reeling_speed.out')
        min_speed_in = np.array([kpi['min_reeling_speed']['in']
                                 for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_in,
                      label='min_reeling_speed.in')
        min_speed_out = np.array([kpi['min_reeling_speed']['out']
                                  for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_out,
                      label='min_reeling_speed.out')
        if reeling_speed_limits:
            ax[3, 1].axhline(reeling_speed_limits[0],
                             linestyle='--', color='k')
            ax[3, 1].axhline(reeling_speed_limits[1],
                             linestyle='--', color='k')
        ax[3, 1].grid()
        ax[3, 1].set_ylabel('Reeling speed [m/s]')
        ax[3, 1].legend(loc=2)

        # Plot constraint matrix.
        cons_matrix = np.array(cons).transpose()
        n_cons = cons_matrix.shape[0]

        cons_treshold_magenta = -.1
        cons_treshold_red = -1e-6

        # Assign color codes based on the constraint values.
        color_code_matrix = np.where(cons_matrix < cons_treshold_magenta,
                                     -2, 0)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_magenta) &
                                     (cons_matrix < cons_treshold_red),
                                     -1, color_code_matrix)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_red) &
                                     (cons_matrix < 1e-3),
                                     1, color_code_matrix)
        color_code_matrix = np.where(cons_matrix == 0.,
                                     0, color_code_matrix)
        color_code_matrix = np.where(cons_matrix >= 1e-3,
                                     2, color_code_matrix)

        # Plot color code matrix.
        cmap = mpl.colors.ListedColormap(['r', 'm', 'y', 'g', 'b'])
        bounds = [-2, -1, 0, 1, 2]
        mpl.colors.BoundaryNorm(bounds, cmap.N)
        im1 = ax[4, 1].matshow(color_code_matrix, cmap=cmap,
                               vmin=bounds[0], vmax=bounds[-1],
                               extent=[self.wind_speeds[0],
                                       self.wind_speeds[-1], n_cons, 0])
        ax[4, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[4, 1].set_yticklabels(range(n_cons))
        ax[4, 1].set_ylabel('Constraint id\'s')

        # Add colorbar.
        ax_pos = ax[4, 1].get_position()
        h_cbar = ax_pos.y1 - ax_pos.y0
        w_cbar = .012
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = np.arange(-2+4/10., 2., 4/5.)
        cbar_ticks_labels = ['<-.1', '<0', '0', '~0', '>0']
        cbar = fig.colorbar(im1, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        # Plot constraint matrix with linear mapping the colors from
        # data values between plot_cons_range.
        plot_cons_range = [-.1, .1]
        im2 = ax[5, 1].matshow(cons_matrix, vmin=plot_cons_range[0],
                               vmax=plot_cons_range[1], cmap=mpl.cm.YlGnBu_r,
                               extent=[self.wind_speeds[0],
                                       self.wind_speeds[-1], n_cons, 0])
        ax[5, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[5, 1].set_yticklabels(range(n_cons))
        ax[5, 1].set_ylabel('Constraint id\'s')

        # Add colorbar.
        ax_pos = ax[5, 1].get_position()
        cbar_ax = fig.add_axes([ax_pos.x1, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = plot_cons_range[:]
        cbar_ticks_labels = [str(v) for v in cbar_ticks]
        if plot_cons_range[0] < np.min(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(1, np.min(cons_matrix))
            cbar_ticks_labels.insert(1, 'min: {:.2E}'
                                     .format(np.min(cons_matrix)))
        if plot_cons_range[0] < np.max(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(-1, np.max(cons_matrix))
            cbar_ticks_labels.insert(-1, 'max: {:.2E}'
                                     .format(np.max(cons_matrix)))
        cbar = fig.colorbar(im2, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        ax[-1, 0].set_xlabel('Wind speeds [m/s]')
        ax[-1, 1].set_xlabel('Wind speeds [m/s]')
        ax[0, 0].set_xlim([self.wind_speeds[0], self.wind_speeds[-1]])
        if not self.plots_interactive:
            plt.savefig(self.plot_output_file.format(
                title='optimization_results{}'.format(plot_info)))

    def curve(self, return_mech=False):
        wind_speeds = self.wind_speeds
        power = np.array([kpis['average_power']['cycle']
                          for kpis in self.performance_indicators])
        if not return_mech:
            eff = [kpis['generator']['eff']['cycle']
                   for kpis in self.performance_indicators]
            for i, e in enumerate(eff):
                if e is None:
                    eff[i] = 0
            eff = np.array(eff)
            power = eff * power
        return wind_speeds, power

    def export_results(self, file_name):
        export_dict = self.__dict__
        # for k, v in export_dict.items():
        #     if isinstance(v, np.ndarray):
        #         export_dict[k] = v.copy().tolist()
        with open(file_name, 'wb') as f:
            pickle.dump(export_dict, f)

    def import_results(self, file_name):
        with open(file_name, 'rb') as f:
            import_dict = pickle.load(f)
        for k, v in import_dict.items():
            setattr(self, k, v)
