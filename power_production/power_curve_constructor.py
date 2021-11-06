import matplotlib as mpl
import numpy as np
import pickle
from scipy.stats import truncnorm

from .qsm import SteadyStateError, OperationalLimitViolation, PhaseError

from .cycle_optimizer import OptimizerError

from .utils import flatten_dict

from .config import plots_interactive, plot_output_file

if not plots_interactive:
    mpl.use('Pdf')
import matplotlib.pyplot as plt


class PowerCurveConstructor:
    def __init__(self, wind_speeds):
        self.wind_speeds = wind_speeds

        self.x_opts = []
        self.x0 = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []
        # TODO remove?
        self.optimization_rounds = {'total_opts': [],
                                    'successful_opts': [],
                                    'opt_and_sim_successful': [],
                                    'unstable_results': [],
                                    'second_wind_speed_test': [],
                                    'optimizer_error_starting_vals': [],
                                    'optimizer_error_wind_speed': []}

    def run_optimization(self, wind_speed, power_optimizer, x0,
                         second_attempt=False,
                         save_initial_value_scan_output=True,
                         n_x_test=2, test_until_n_succ=3):
        # TODO set save scan output to False by default
        power_optimizer.environment_state.set_reference_wind_speed(wind_speed)

        print("x0:", x0)
        # Optimize around x0
        # perturb x0:
        x0_range = [x0]
        # Optimization variables bounds defining the search space.
        bounds = power_optimizer.bounds_real_scale
        reduce_x = power_optimizer.reduce_x
        for n_test in range(n_x_test):
            # Gaussian random selection of x0 within bounds
            # TODO or just use uniform distr, mostly at bounds anyways...?
            x0_range.append([truncnorm(a=bounds[i][0]/bounds[i][1],
                                        b=1, scale=bounds[i][1]).rvs()
                              if i in reduce_x else x0[i]
                              for i in range(len(x0))])

            # Gaussian random smearing of x0 within bounds
            smearing = 0.1 # 10% smearing of the respective values

            def get_smeared_x0():
                return [np.random.normal(x0[i], x0[i]*smearing)
                        if i in reduce_x else x0[i] for i in range(len(x0))]

            def test_smeared_x0(test_x0, precision=0):
                return np.all([np.logical_and(
                    test_x0[i] >= (bounds[i][0]-precision),
                    test_x0[i] <= (bounds[i][1]+precision))
                    for i in range(len(test_x0))])

            def smearing_x0():
                test_smearing = get_smeared_x0()
                bounds_adhered = test_smeared_x0(test_smearing)
                while not bounds_adhered:
                    test_smearing = get_smeared_x0()
                    bounds_adhered = test_smeared_x0(test_smearing)
                return test_smearing
            # Test on two smeared variations of x0
            x0_range.append(smearing_x0())
            x0_range.append(smearing_x0())

        x0_range = np.array(x0_range)
        print('Testing x0 range: ', x0_range)
        n_x0 = x0_range.shape[0]
        x_opts = []
        op_ress = []
        conss = []
        kpiss = []
        sim_successfuls = []
        opt_successfuls = []

        for i in range(n_x0):
            x0_test = x0_range[i]
            power_optimizer.x0_real_scale = x0_test
            try:
                print("Testing the {}th starting values: {}".format(i,
                                                                    x0_test))
                x_opts.append(power_optimizer.optimize())
                if test_smeared_x0(power_optimizer.x_opt_real_scale,
                                   precision=power_optimizer.precision):
                    # Safety check if variable bounds are adhered
                    op_ress.append(power_optimizer.op_res)
                    opt_successfuls.append(True)
                    try:
                        cons, kpis = power_optimizer.eval_point()
                        conss.append(cons)
                        kpiss.append(kpis)
                        sim_successfuls.append(True)
                        print('Simulation successful')
                        if sum(sim_successfuls) == test_until_n_succ:
                            x0_range = x0_range[:i+1]
                            break
                    except (SteadyStateError, OperationalLimitViolation,
                            PhaseError) as e:
                        print("Error occurred while evaluating the "
                              "resulting optimal point: {}".format(e))
                        # Evaluate results with relaxed errors
                        # relaxed errors only relax OperationalLimitViolation
                        cons, kpis = power_optimizer.eval_point(
                            relax_errors=True)
                        conss.append(cons)
                        kpiss.append(kpis)
                        sim_err = e
                        sim_successfuls.append(False)
                        print('Simulation failed -> errors relaxed')
                else:
                    print("Optimization number "
                          "{} finished with an error: {}".format(
                              i, 'Optimization bounds violated'))
                    opt_err = OptimizerError("Optimization bounds violated.")
                    # Drop last x_opts, bonds are not adhered
                    x_opts = x_opts[:-1]
                    opt_successfuls.append(False)

            except (OptimizerError) as e:
                print("Optimization number "
                      "{} finished with an error: {}".format(i, e))
                opt_err = e
                opt_successfuls.append(False)
                continue
            except (SteadyStateError, PhaseError,
                    OperationalLimitViolation) as e:
                print("Optimization number "
                      "{} finished with a simulation error: {}".format(i, e))
                opt_err = e
                opt_successfuls.append(False)
                continue
            except (FloatingPointError) as e:
                print("Optimization number "
                      "{} finished due to a mathematical simulation error: {}"
                      .format(i, e))
                opt_err = e
                opt_successfuls.append(False)

        # TODO Include test for correct power?
        # TODO Output handling different? own function?

        self.optimization_rounds['total_opts'].append(len(opt_successfuls))
        self.optimization_rounds['successful_opts'].append(sum(opt_successfuls))
        self.optimization_rounds['opt_and_sim_successful'].append(sum(sim_successfuls)) # good results
        self.optimization_rounds['second_wind_speed_test'].append(second_attempt)

        if save_initial_value_scan_output:
            print('Saving optimizer scan output')
            #TODO scan optimizer output / sim results to file - dep on wind_speed

            #TODO independent of this: optvis history output?

        if sum(sim_successfuls) > 0:
            # Optimization and Simulation successful at least once:
            # append to results
            # consistency check sim results - both optimization and simulation work
            x0_success = x0_range[opt_successfuls][sim_successfuls]
            #x0_failed = list(x0_range[np.logical_not(opt_successfuls)]) + list(x0_range[opt_successfuls][np.logical_not(sim_successfuls)])
            #print('Failed starting values: ', x0_failed)
            #print('Successful starting values: ', x0_success)

            # consistency check function values
            # corresponding eval function values from the optimizer
            flag_unstable_opt_result = False

            #print('Optimizer x point results: ', x_opts)
            #print(' Leading to a successful simulation:', sim_successfuls)
            x_opts_succ = np.array(x_opts)[sim_successfuls]
            (x_opt_mean, x_opt_std) = (np.mean(x_opts_succ, axis=0), np.std(x_opts_succ, axis=0))
            #print('  The resulting mean {} with a standard deviation of {}'.format(x_opt_mean, x_opt_std))
            if (x_opt_std > np.abs(0.1*x_opt_mean)).any(): #TODO: lower/higher, different check? - make this as debug output?
                #print('  More than 1% standard deviation - unstable result')
                flag_unstable_opt_result = True

            # corresponding eval function values from the optimizer
            op_ress_succ = [op_ress[i] for i in range(len(op_ress)) if sim_successfuls[i]]
            f_opt = [op_res['fun'] for op_res in op_ress_succ]
            #print('Successful optimizer eval function results: ', f_opt)
            (f_opt_mean, f_opt_std) = (np.mean(f_opt), np.std(f_opt))
            #print('  The resulting mean {} with a standard deviation of {}'.format(f_opt_mean, f_opt_std))
            if f_opt_std > np.abs(0.1*f_opt_mean):
                #print('  More than 1% standard deviation - unstable result')
                flag_unstable_opt_result = True

            self.optimization_rounds['unstable_results'].append(flag_unstable_opt_result)

            # Chose best optimization result:
            minimal_f_opt = np.argmin(f_opt) # maching index in sim_successfuls

            self.x0.append(x0_success[minimal_f_opt])
            x_opt = x_opts_succ[minimal_f_opt]
            self.x_opts.append(x_opt)
            self.optimization_details.append(op_ress_succ[minimal_f_opt])

            cons = [conss[i] for i in range(len(kpiss)) if sim_successfuls[i]][minimal_f_opt]
            kpis = [kpiss[i] for i in range(len(kpiss)) if sim_successfuls[i]][minimal_f_opt]
            print("cons:", cons)
            self.constraints.append(cons)
            # Failed simulation results are later masked
            kpis['sim_successful'] = True
            self.performance_indicators.append(kpis)

            return x_opt, True # x_opt result, sim_successful

        elif sum(opt_successfuls) > 0:
            # simulations failed (run again with loose errors) but optimization worked
            print('All simulations failed, save flagged loose error simulation output')
            self.x0.append(x0_range[opt_successfuls][-1])
            self.x_opts.append(x_opts[-1])
            self.optimization_details.append(op_ress[-1])

            print("cons:", conss[-1])
            self.constraints.append(conss[-1])
            # Failed simulation results are later masked
            kpis = kpiss[-1]
            kpis['sim_successful'] = False
            self.performance_indicators.append(kpis)

            print('Output appended, raise simulation error: ')
            raise sim_err
        else:
            # optimizatons all failed
            self.optimization_rounds['optimizer_error_starting_vals'].append(x0_range)
            self.optimization_rounds['optimizer_error_wind_speed'].append(wind_speed)
            print('All optimizations failed, raise optimizer Error: ')
            raise opt_err




    def run_predefined_sequence(self, seq, x0_start):
        wind_speed_tresholds = iter(sorted(list(seq)))
        vw_switch = next(wind_speed_tresholds)

        x_opt_last, vw_last = None, None
        for i, vw in enumerate(self.wind_speeds):
            if vw > vw_switch:
                vw_switch = next(wind_speed_tresholds)

            power_optimizer = seq[vw_switch]['power_optimizer']
            dx0 = seq[vw_switch].get('dx0', None)

            if x_opt_last is None:
                x0_next = x0_start
            else:
                x0_next = x_opt_last + dx0*(vw - vw_last)

            print("[{}] Processing v={:.2f}m/s".format(i, vw))
            try:
                x_opt, sim_successful = self.run_optimization(vw, power_optimizer, x0_next)
            except (OperationalLimitViolation, SteadyStateError, PhaseError, OptimizerError) as e:
                try:  # Retry for a slightly different wind speed.
                    print('first optimization/simulation ended in error: {}'.format(e))
                    print('run with varied wind speed:', vw+1e-2)
                    x_opt, sim_successful = self.run_optimization(vw+1e-2, power_optimizer, x0_next, second_attempt=True)
                    self.wind_speeds[i] = vw+1e-2
                except (OperationalLimitViolation, SteadyStateError, PhaseError, OptimizerError):
                    self.wind_speeds = self.wind_speeds[:i]
                    print("Optimization sequence stopped prematurely due to failed optimization. {:.2f} m/s is the "
                          "highest wind speed for which the optimization was successful.".format(self.wind_speeds[-1]))
                    break

            if sim_successful:
                x_opt_last = x_opt
                vw_last = vw
            print(self.wind_speeds[:i+1], [kpi['sim_successful'] for kpi in self.performance_indicators])
            print(len(self.wind_speeds[:i+1]), sum([kpi['sim_successful'] for kpi in self.performance_indicators]))

    def plot_optimal_trajectories(self, wind_speed_ids=None, ax=None, circle_radius=200, elevation_line=25*np.pi/180,
                                  plot_info=''):
        if ax is None:
            plt.figure(figsize=(6, 3.5))
            plt.subplots_adjust(right=0.65)
            ax = plt.gca()

        mask = [kpis['sim_successful'] for kpis in self.performance_indicators]
        all_kpis = [kpi for i,kpi in enumerate(self.performance_indicators) if mask[i]]
        #mask power discontinuities
        p_cycle = np.array([kpis['average_power']['cycle'] for kpis in all_kpis])
        mask_power = p_cycle > 0  #TODO  mask negative jumps in power - check reason for negative power/ strong jumps
        p_cycle = p_cycle[mask_power]
        all_kpis = [kpi for i, kpi in enumerate(all_kpis) if mask_power[i]]
        wind_speeds = self.wind_speeds[mask_power]

        masking_counter = 0
        #TODO resolve source of problems
        while True:
            mask_power_disc = [True] + list((np.diff(p_cycle) > -1000))
            if sum(mask_power_disc) == len(mask_power_disc):
                # No more discontinuities
                break
            print('Masking {} power runs'.format(len(mask_power_disc) - sum(mask_power_disc)))
            p_cycle = p_cycle[mask_power_disc]
            all_kpis = [kpi for i, kpi in enumerate(all_kpis) if mask_power_disc[i]]
            wind_speeds = wind_speeds[mask_power_disc]
            masking_counter += 1
        print('Total Masking {} power runs'.format(masking_counter))

        if wind_speed_ids is None:
            if len(wind_speeds) > 8:
                wind_speed_ids = [int(a) for a in np.linspace(0, len(wind_speeds)-1, 6)]
            else:
                wind_speed_ids = range(len(wind_speeds))


        for i in wind_speed_ids:
            v = wind_speeds[i]
            kpis = all_kpis[i]
            if kpis is None:
                print("No trajectory available for {} m/s wind speed.".format(v))
                continue

            x_kite, z_kite = zip(*[(kp.x, kp.z) for kp in kpis['kinematics']])
            print('min x, z: ', min(x_kite), min(z_kite))
            # try:
            #     z_traj = [kp.z for kp in kite_positions['trajectory']]
            # except AttributeError:
            #     z_traj = [np.sin(kp.elevation_angle)*kp.straight_tether_length for kp in kite_positions['trajectory']]
            ax.plot(x_kite, z_kite, label="$v_{100m}$="+"{:.1f} ".format(v) + "m s$^{-1}$")

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
        if not plots_interactive: plt.savefig(plot_output_file.format(title='optimal_trajectories{}'.format(plot_info)))


    def plot_optimization_results(self, opt_variable_labels=None, opt_variable_bounds=None, tether_force_limits=None,
                                  reeling_speed_limits=None, plot_info=''):
        assert self.x_opts, "No optimization results available for plotting."

        # mask unsuccessful simulations
        mask = [kpis['sim_successful'] for kpis in self.performance_indicators]
        kpis = [kpi for i,kpi in enumerate(self.performance_indicators) if mask[i]]
        opt_details = [od for i, od in enumerate(self.optimization_details) if mask[i]]
        xf = [xopt for i, xopt in enumerate(self.x_opts) if mask[i]]
        x0 = [x for i, x in enumerate(self.x0) if mask[i]]
        cons = [c for i,c in enumerate(self.constraints) if mask[i]]

        try: #TODO this seems to be ineffective code?
            performance_indicators = next(list(flatten_dict(kpi)) for kpi in kpis if kpi is not None)
        except StopIteration:
            performance_indicators = []

        n_opt_vars = len(xf[0])
        fig, ax = plt.subplots(max([n_opt_vars, 6]), 2, sharex=True) #TODO fig size in pdf too small

        # In the left column plot each optimization variable against the wind speed.
        for i in range(n_opt_vars):
            # Plot optimal and starting points.
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in xf], label='x_opt')
            ax[i, 0].plot(self.wind_speeds, [a[i] for a in x0], 'o', markerfacecolor='None', label='x0')

            if opt_variable_labels:
                label = opt_variable_labels[i]
                ax[i, 0].set_ylabel(label)
            else:
                ax[i, 0].set_ylabel("x[{}]".format(i))

            if opt_variable_bounds is not None:
                ax[i, 0].axhline(opt_variable_bounds[i, 0], linestyle='--', color='k')
                ax[i, 0].axhline(opt_variable_bounds[i, 1], linestyle='--', color='k')

            ax[i, 0].grid()
        ax[0, 0].legend()

        # In the right column plot the number of iterations in the upper panel.

        nits = np.array([od['nit'] for od in opt_details])
        ax[0, 1].plot(self.wind_speeds, nits)
        #TODO this is now obsolete / to be done differently
        #mask_opt_failed = np.array([~od['success'] for od in opt_details])
        #ax[0, 1].plot(self.wind_speeds[mask_opt_failed], nits[mask_opt_failed], 'x', label='opt failed')
        #mask_sim_failed = np.array([~kpi['sim_successful'] for kpi in kpis])
        #ax[0, 1].plot(self.wind_speeds[mask_sim_failed], nits[mask_sim_failed], 'x', label='sim failed')
        ax[0, 1].grid()
        #ax[0, 1].legend()
        ax[0, 1].set_ylabel('Optimization iterations [-]')

        # In the second panel, plot the optimal power.
        #TODO this is now obsolete / to be done differently
        #cons_treshold = -.1
        #mask_cons_adhered = np.array([all([c >= cons_treshold for c in con]) for con in cons])
        #mask_plot_power = ~mask_sim_failed & mask_cons_adhered
        power = np.array([kpi['average_power']['cycle'] for kpi in kpis])
        #power[~mask_plot_power] = np.nan
        ax[1, 1].plot(self.wind_speeds, power)
        ax[1, 1].grid()
        ax[1, 1].set_ylabel('Mean power [W]')

        # In the third panel, plot the tether force related performance indicators.
        max_force_in = np.array([kpi['max_tether_force']['in'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_in, label='max_tether_force.in')
        max_force_out = np.array([kpi['max_tether_force']['out'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_out, label='max_tether_force.out')
        max_force_trans = np.array([kpi['max_tether_force']['trans'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_trans, label='max_tether_force.trans')
        if tether_force_limits:
            ax[3, 1].axhline(tether_force_limits[0], linestyle='--', color='k')
            ax[3, 1].axhline(tether_force_limits[1], linestyle='--', color='k')
        ax[2, 1].grid()
        ax[2, 1].set_ylabel('Tether force [N]')
        ax[2, 1].legend(loc=2)
        ax[2, 1].annotate('Violation occurring before\nswitch to force controlled',
                          xy=(0.05, 0.10), xycoords='axes fraction')

        # Plot reeling speed related performance indicators.
        max_speed_in = np.array([kpi['max_reeling_speed']['in'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_in, label='max_reeling_speed.in')
        max_speed_out = np.array([kpi['max_reeling_speed']['out'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_out, label='max_reeling_speed.out')
        min_speed_in = np.array([kpi['min_reeling_speed']['in'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_in, label='min_reeling_speed.in')
        min_speed_out = np.array([kpi['min_reeling_speed']['out'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_out, label='min_reeling_speed.out')
        if reeling_speed_limits:
            ax[3, 1].axhline(reeling_speed_limits[0], linestyle='--', color='k')
            ax[3, 1].axhline(reeling_speed_limits[1], linestyle='--', color='k')
        ax[3, 1].grid()
        ax[3, 1].set_ylabel('Reeling speed [m/s]')
        ax[3, 1].legend(loc=2)

        # Plot constraint matrix.
        cons_matrix = np.array(cons).transpose()
        n_cons = cons_matrix.shape[0]

        cons_treshold_magenta = -.1
        cons_treshold_red = -1e-6

        # Assign color codes based on the constraint values.
        color_code_matrix = np.where(cons_matrix < cons_treshold_magenta, -2, 0)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_magenta) & (cons_matrix < cons_treshold_red), -1,
                                     color_code_matrix)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_red) & (cons_matrix < 1e-3), 1, color_code_matrix)
        color_code_matrix = np.where(cons_matrix == 0., 0, color_code_matrix)
        color_code_matrix = np.where(cons_matrix >= 1e-3, 2, color_code_matrix)

        # Plot color code matrix.
        cmap = mpl.colors.ListedColormap(['r', 'm', 'y', 'g', 'b'])
        bounds = [-2, -1, 0, 1, 2]
        mpl.colors.BoundaryNorm(bounds, cmap.N)
        im1 = ax[4, 1].matshow(color_code_matrix, cmap=cmap, vmin=bounds[0], vmax=bounds[-1],
                                    extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
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

        # Plot constraint matrix with linear mapping the colors from data values between plot_cons_range.
        plot_cons_range = [-.1, .1]
        im2 = ax[5, 1].matshow(cons_matrix, vmin=plot_cons_range[0], vmax=plot_cons_range[1], cmap=mpl.cm.YlGnBu_r,
                               extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
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
            cbar_ticks_labels.insert(1, 'min: {:.2E}'.format(np.min(cons_matrix)))
        if plot_cons_range[0] < np.max(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(-1, np.max(cons_matrix))
            cbar_ticks_labels.insert(-1, 'max: {:.2E}'.format(np.max(cons_matrix)))
        cbar = fig.colorbar(im2, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        ax[-1, 0].set_xlabel('Wind speeds [m/s]')
        ax[-1, 1].set_xlabel('Wind speeds [m/s]')
        ax[0, 0].set_xlim([self.wind_speeds[0], self.wind_speeds[-1]])
        if not plots_interactive: plt.savefig(plot_output_file.format(title='optimization_results{}'.format(plot_info)))


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
