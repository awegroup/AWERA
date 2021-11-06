from pyoptsparse import Optimization, SLSQP, Gradient
from pyoptsparse import History
from scipy import optimize as op
import numpy as np
import matplotlib as mpl
mpl.use('Pdf')
import matplotlib.pyplot as plt

from qsm import Cycle
from utils import flatten_dict

from config_production import optimizer_history_file_name

class OptimizerError(Exception):
    """Exception raised for optimizations not finishing successfully with 0."""
    pass

def read_slsqp_output_file(print_details=True):
    """Read relevant information from pyOpt's output file for the SLSQP algorithm."""
    i_iter = 0
    with open('SLSQP.out') as f:
        for line in f:
            if line[:11] == "     ITER =":
                i_iter += 1
                x_iter = []
                while True:
                    line = next(f)
                    xi = line.strip()
                    if not xi:
                        break
                    elif line[:38] == "        NUMBER OF FUNC-CALLS:  NFUNC =":
                        nfunc_line = line
                        ngrad_line = next(f)
                        break
                    else:
                        x_iter.append(float(xi[7:15]))
                if print_details:
                    print("Iter {}: x=".format(i_iter) + str(x_iter))
            elif line[:38] == "        NUMBER OF FUNC-CALLS:  NFUNC =":
                nfunc_line = line
                ngrad_line = next(f)
                break

    nit = i_iter
    nfev = nfunc_line.split()[5]
    njev = ngrad_line.split()[5]

    return nit, nfev, njev


def convert_optimization_result(op_sol, nit, nfev, njev, print_details, iprint):
    """Write pyOpt's optimization results to the same format as the output of SciPy's minimize function."""
    x = []
    for varname in op_sol.variables:
        for var in op_sol.variables[varname]:
            if var.type in ["c", "i"]:
                x.append(var.value)
    op_res = {
        'x': x,
        'success': op_sol.optInform['value'] == 0,
        'message': op_sol.optInform['text'],
        'fun': op_sol.objectives['obj'].value,
        'nit': nit,
        'nfev': nfev,
        'njev': njev,
    }
    if print_details:
        print("{}    (Exit mode {})".format(op_res['message'], op_sol.optInform['value']))
        print("            Current function value: {}".format(op_res['fun']))
        if iprint:
            print("            Iterations: {}".format(nit))
            print("            Function evaluations: {}".format(nfev))
            print("            Gradient evaluations: {}".format(njev))

    return op_res


class Optimizer:
    """Class collecting useful functionalities for solving an optimization problem and evaluating the results using
    different settings and, thereby, enabling assessing the effect of these settings."""
    def __init__(self, x0_real_scale, bounds_real_scale, scaling_x, reduce_x, reduce_ineq_cons,
                 system_properties, environment_state):
        assert isinstance(x0_real_scale, np.ndarray)
        assert isinstance(bounds_real_scale, np.ndarray)
        assert isinstance(scaling_x, np.ndarray)
        assert isinstance(reduce_x, np.ndarray)

        # Simulation side conditions.
        self.system_properties = system_properties
        self.environment_state = environment_state

        # Optimization configuration.
        self.use_library = 'pyOptSparse' #  'scipy' #   # Either 'pyOptSparse' or 'scipy' can be opted. pyOptSparse is in general faster, however more
        # cumbersome to install.
        self.use_parallel_processing = False  #TODO Only compatible with pyOpt: used for determining the gradient. Script
        # should be run using: mpiexec -n 4 python script.py, when using parallel processing. Parallel processing does
        # not speed up solving the problem when only a limited number of processors are available.

        self.scaling_x = scaling_x  # Scaling the optimization variables will affect the optimization. In general, a
        # similar search range is preferred for each variable.
        self.x0_real_scale = x0_real_scale  # Optimization starting point.
        self.bounds_real_scale = bounds_real_scale  # Optimization variables bounds defining the search space.
        self.reduce_x = reduce_x  # Reduce the search space by providing a tuple with id's of x to keep. Set to None for
        # utilizing the full search space.
        if isinstance(reduce_ineq_cons, int):
            self.reduce_ineq_cons = np.arange(reduce_ineq_cons)  # Reduces the number of inequality constraints used for
            # solving the problem.
        else:
            self.reduce_ineq_cons = reduce_ineq_cons

        # Settings inferred from the optimization configuration.
        self.x0 = None  # Scaled starting point.
        self.x_opt_real_scale = None  # Optimal solution for the optimization vector.

        # Optimization operational attributes.
        self.x_last = None  # Optimization vector used for the latest evaluation function call.
        self.obj = None  # Value of the objective/cost function of the latest evaluation function call.
        self.ineq_cons = None  # Values of the inequality constraint functions of the latest evaluation function call.
        self.x_progress = []  # Evaluated optimization vectors of every conducted optimization iteration - only tracked
        # when using Scipy.

        # Optimization result.
        self.op_res = None  # Result dictionary of optimization function.

    def clear_result_attributes(self):
        """Clear the inferred optimization settings and results before re-running the optimization."""
        self.x0 = None
        self.x_last = None
        self.obj = None
        self.ineq_cons = None
        self.x_progress = []
        self.x_opt_real_scale = None
        self.op_res = None

    def eval_point(self, plot_result=False, relax_errors=False, x_real_scale=None):
        """Evaluate simulation results using the provided optimization vector. Uses either the optimization vector
        provided as argument, the optimal vector, or the starting point for the simulation."""
        if x_real_scale is None:
            if self.x_opt_real_scale is not None:
                x_real_scale = self.x_opt_real_scale
            else:
                x_real_scale = self.x0_real_scale
        kpis = self.eval_performance_indicators(x_real_scale, plot_result, relax_errors)
        cons = self.eval_fun(x_real_scale, False, relax_errors=relax_errors)[1]
        return cons, kpis

    def eval_fun_pyopt(self, x, *args):
        """PyOpt's implementation of SLSQP can produce NaN's in the optimization vector or contain values that violate
        the bounds. true for pyoptsparse? #TODO"""
        x_vals = [v for k,v in x.items()]
        if np.isnan(x_vals).any():
            raise OptimizerError("Optimization vector contains NaN's.")

        if self.reduce_x is not None:
            x_full = self.x0.copy()
            x_full[self.reduce_x] = x_vals
        else:
            x_full = x_vals

        bounds_adhered = (x_full - self.bounds_real_scale[:, 0]*self.scaling_x >= -1e6).all() and \
                         (x_full - self.bounds_real_scale[:, 1]*self.scaling_x <= 1e6).all()
        if not bounds_adhered:
            #raise OptimizerError("Optimization bounds violated.") #TODO keep this y/n?
            print('Optimization bounds VIOLATED. Diff to bounds:')
            print('lower:', (x_full - self.bounds_real_scale[:, 0]*self.scaling_x >= -1e6))
            print('upper:', (x_full - self.bounds_real_scale[:, 1]*self.scaling_x <= 1e6))

        obj, ineq_cons = self.eval_fun(x_full, *args)
        #print(x_full, ineq_cons)
        funcs = {}
        funcs['obj'] = obj
        for idx, i_c in enumerate(self.reduce_ineq_cons):
            funcs['g{}'.format(i_c)] = ineq_cons[idx]
        self.op_eval_func_calls += 1
        return funcs, 0

    def obj_fun(self, x, *args):
        """Scipy's implementation of SLSQP uses separate functions for the objective and constraints. Since the
        objective and constraints result from the same simulation, a work around is provided to prevent running the same
        simulation twice."""
        if self.reduce_x is not None:
            x_full = self.x0.copy()
            x_full[self.reduce_x] = x
        else:
            x_full = x
        if not np.array_equal(x_full, self.x_last):
            self.obj, self.ineq_cons = self.eval_fun(x_full, *args)
            self.x_last = x_full.copy()

        return self.obj

    def cons_fun(self, x, return_i=-1, *args):
        """Scipy's implementation of SLSQP uses separate functions for the objective and every constraint. Since the
        objective and constraints result from the same simulation, a work around is provided to prevent running the same
        simulation twice."""
        if self.reduce_x is not None:
            x_full = self.x0.copy()
            x_full[self.reduce_x] = x
        else:
            x_full = x
        if not np.array_equal(x_full, self.x_last):
            self.obj, self.ineq_cons = self.eval_fun(x_full, *args)
            self.x_last = x_full.copy()

        if return_i > -1:
            return self.ineq_cons[return_i]
        else:
            return self.ineq_cons

    def callback_fun_scipy(self, x):
        """Function called when using Scipy for every optimization iteration (does not include function calls for
        determining the gradient)."""
        if np.isnan(x).any():
            raise OptimizerError("Optimization vector contains nan's.")
        self.x_progress.append(x.copy())

    def optimize(self, *args, maxiter=30, iprint=-1): # 0):
        """Perform optimization."""
        self.clear_result_attributes()
        # Construct scaled starting point and bounds
        self.x0 = self.x0_real_scale*self.scaling_x
        bounds = self.bounds_real_scale.copy()
        bounds[:, 0] = bounds[:, 0]*self.scaling_x
        bounds[:, 1] = bounds[:, 1]*self.scaling_x

        if self.reduce_x is None:
            starting_point = self.x0
        else:
            starting_point = self.x0[self.reduce_x]
            bounds = bounds[self.reduce_x]

        print_details = True
        ftol, eps = 1e-6, 1e-6
        self.precision = ftol
        if self.use_library == 'scipy':
            con = {
                'type': 'ineq',  # g_i(x) >= 0
                'fun': self.cons_fun,
            }
            cons = []
            for i in self.reduce_ineq_cons:
                cons.append(con.copy())
                cons[-1]['args'] = (i, *args)

            options = {
                'disp': print_details,
                'maxiter': maxiter,
                'ftol': ftol,
                'eps': eps,
                'iprint': iprint, # 1: show final summary
            }
            self.op_res = dict(op.minimize(self.obj_fun, starting_point, args=args, bounds=bounds, method='SLSQP',
                                           options=options, callback=self.callback_fun_scipy, constraints=cons))
        elif self.use_library == 'pyOptSparse':
            op_problem = Optimization('Pumping cycle power', self.eval_fun_pyopt)
            op_problem.addObj('obj')

            if self.reduce_x is None:
                x_range = range(len(self.x0))
            else:
                x_range = self.reduce_x
            for i_x, xi0, b in zip(x_range, starting_point, bounds):
                op_problem.addVar('x{}'.format(i_x), 'c', lower=b[0], upper=b[1], value=xi0)

            for i_c in self.reduce_ineq_cons:
                op_problem.addCon('g{}'.format(i_c), lower=0)
                # force_out_setpoint_min, force_in_setpoint_max, ineq_cons_traction_max_force, ineq_cons_cw_patterns
            if self.use_parallel_processing:
                sens_mode = 'pgc' #TODO pyOptSparse implementation?
            else:
                sens_mode = ''
            # TODO update for pyoptsparse
            # grad = Gradient(op_problem, sens_type='FD', sens_mode=sens_mode, sens_step=eps)
            # f0, g0, _ = self.eval_fun_pyopt(starting_point)
            # grad_fun = lambda f, g: grad.getGrad(starting_point, {}, [f], g)
            # dff0, dgg0 = grad_fun(f0, g0)
            # if np.any(dff0 == 0.):
            #     print("!!! Gradient contains zero component !!!")

            optimizer = SLSQP()
            optimizer.setOption('IPRINT', iprint)  # -1 - None, 0 - Screen, 1 - File
            optimizer.setOption('MAXIT', maxiter)
            optimizer.setOption('ACC', ftol)

            self.op_eval_func_calls = 0
            op_sol = optimizer(op_problem, sens='FD', sensMode=sens_mode, sensStep=eps, *args)  # TODO , storeHistory=optimizer_history_file_name)
            # print(op_sol)  # TODO make optional
            nit, nfev, njev = op_sol.userObjCalls , self.op_eval_func_calls, op_sol.userSensCalls    #TODO old: read_slsqp_output_file(print_details) from iprint = 1


            self.op_res = convert_optimization_result(op_sol, nit, nfev, njev, print_details, iprint)
        else:
            raise ValueError("Invalid library provided.")
        # Check if optimization terminated successfully
        if not self.op_res['success']:
            raise OptimizerError(self.op_res['message'])

        # Extract optimization results in real scale
        if self.reduce_x is None:
            res_x = self.op_res['x']
        else:
            res_x = self.x0.copy()
            res_x[self.reduce_x] = self.op_res['x']
        self.x_opt_real_scale = res_x/self.scaling_x

        return self.x_opt_real_scale

    def plot_opt_evolution(self):
        """Method can be called after finishing optimizing using Scipy to plot how the optimization evolved and arrived
        at the final solution."""
        fig, ax = plt.subplots(len(self.x_progress[0])+1, 2, sharex=True)
        for i in range(len(self.x_progress[0])):
            # Plot optimization variables.
            ax[i, 0].plot([x[i] for x in self.x_progress])
            ax[i, 0].grid(True)
            ax[i, 0].set_ylabel('x[{}]'.format(i))

            # Plot step size.
            tmp = [self.x0]+self.x_progress
            step_sizes = [b[i] - a[i] for a, b in zip(tmp[:-1], tmp[1:])]

            ax[i, 1].plot(step_sizes)
            ax[i, 1].grid(True)
            ax[i, 1].set_ylabel('dx[{}]'.format(i))

        # Plot objective.
        obj_res = [self.obj_fun(x) for x in self.x_progress]
        ax[-1, 0].plot([res for res in obj_res])
        ax[-1, 0].grid()
        ax[-1, 0].set_ylabel('Objective [-]')

        # Plot constraints.
        cons_res = [self.cons_fun(x, -1)[self.reduce_ineq_cons] for x in self.x_progress]
        cons_lines = ax[-1, 1].plot([res for res in cons_res])

        # add shade when one of the constraints is violated
        active_cons = [any([c < -1e-6 for c in res]) for res in cons_res]
        ax[-1, 1].fill_between(range(len(active_cons)), 0, 1, where=active_cons, alpha=0.4,
                               transform=ax[-1, 1].get_xaxis_transform())

        ax[-1, 1].legend(cons_lines, ["constraint {}".format(i) for i in range(len(cons_lines))],
                         bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.subplots_adjust(right=0.7)

        ax[-1, 1].grid()
        ax[-1, 1].set_ylabel('Constraint [-]')

        ax[-1, 0].set_xlabel('Iteration [-]')
        ax[-1, 1].set_xlabel('Iteration [-]')

    def check_gradient(self, x_real_scale=None):
        """Evaluate forward finite difference gradient of objective function at given point. Logarithmic sensitivities
        are evaluated to assess how influential each parameter is at the evaluated point."""
        self.x0 = self.x0_real_scale*self.scaling_x
        step_size = 1e-6

        # Evaluate the gradient at the point set by either the optimization vector provided as argument, the optimal
        # vector, or the starting point for the simulation.
        if x_real_scale is None:
            if self.x_opt_real_scale is not None:
                x_real_scale = self.x_opt_real_scale
            else:
                x_real_scale = self.x0_real_scale
        if self.scaling_x is not None:
            x_ref = x_real_scale*self.scaling_x
        else:
            x_ref = x_real_scale

        obj_ref = self.obj_fun(x_ref)
        ffd_gradient, log_sensitivities = [], []
        for i, xi_ref in enumerate(x_ref):
            x_ref_perturbed = x_ref.copy()
            x_ref_perturbed[i] += step_size
            grad = (self.obj_fun(x_ref_perturbed) - obj_ref) / step_size
            ffd_gradient.append(grad)
            log_sensitivities.append(xi_ref/obj_ref*grad)

        return ffd_gradient, log_sensitivities

    def perform_local_sensitivity_analysis(self):
        """Sweep search range of one of the variables at the time and calculate objective and constraint functions.
        Plot the results of each sweep in a separate panel."""
        ref_point_label = "x_ref"
        if self.reduce_x is None:
            red_x = np.arange(len(self.x0_real_scale))
        else:
            red_x = self.reduce_x
        n_plots = len(red_x)
        bounds = self.bounds_real_scale[red_x]

        # Perform the sensitivity analysis around the intersection point set by either the optimization vector provided
        # as argument, the optimal vector, or the starting point for the simulation.
        if self.x_opt_real_scale is not None:
            x_ref_real_scale = self.x_opt_real_scale
        else:
            x_ref_real_scale = self.x0_real_scale
        f_ref, cons_ref = self.eval_fun(x_ref_real_scale, scale_x=False)

        fig, ax = plt.subplots(n_plots)
        if n_plots == 1:
            ax = [ax]
        fig.subplots_adjust(hspace=.3)

        for i, b in enumerate(bounds):
            # Determine objective and constraint functions along given variable.
            lb, ub = b
            xi_sweep = np.linspace(lb, ub, 50)
            f, g, active_g = [], [], []
            for xi in xi_sweep:
                x_full = list(x_ref_real_scale)
                x_full[red_x[i]] = xi

                try:
                    res_eval = self.eval_fun(x_full, scale_x=False)
                    f.append(res_eval[0])
                    cons = res_eval[1][self.reduce_ineq_cons]
                    g.append(res_eval[1])
                    active_g.append(any([c < -1e-6 for c in cons]))
                except:
                    f.append(None), g.append(None), active_g.append(False)

            # Plot objective function and marker at the reference point.
            ax[i].plot(xi_sweep, f, '--', label='objective')
            x_ref = x_ref_real_scale[red_x[i]]
            ax[i].plot(x_ref, f_ref, 'x', label=ref_point_label, markersize=12)

            # Plot constraint functions.
            for i_cons in self.reduce_ineq_cons:
                cons_line = ax[i].plot(xi_sweep, [c[i_cons] if c is not None else None for c in g],
                                       label='constraint {}'.format(i_cons))
                clr = cons_line[0].get_color()
                ax[i].plot(x_ref, cons_ref[i_cons], 's', markerfacecolor='None', color=clr)

            # Mark ranges where constraint is active with a background color.
            ax[i].fill_between(xi_sweep, 0, 1, where=active_g, alpha=0.4, transform=ax[i].get_xaxis_transform())

            ax[i].set_xlabel(self.OPT_VARIABLE_LABELS[red_x[i]])
            ax[i].set_ylabel("Response [-]")
            ax[i].grid()

        ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax[0].set_title("v={:.1f}m/s".format(self.environment_state.wind_speed))
        plt.subplots_adjust(right=0.7)

    def plot_sensitivity_efficiency_indicators(self, i_x=0):
        """Sweep search range of the requested variable and calculate and plot efficiency indicators."""
        ref_point_label = "x_ref"

        # Perform the sensitivity analysis around the intersection point set by either the optimization vector provided
        # as argument, the optimal vector, or the starting point for the simulation.
        if self.x_opt_real_scale is not None:
            x_real_scale = self.x_opt_real_scale
        else:
            x_real_scale = self.x0_real_scale

        # Reference point
        x_ref = x_real_scale[i_x]
        power_cycle_ref = self.eval_performance_indicators(x_real_scale, scale_x=False)['average_power']['cycle']
        power_out_ref = self.eval_performance_indicators(x_real_scale, scale_x=False)['average_power']['out']
        xlabel = self.OPT_VARIABLE_LABELS[i_x]

        # Sweep between limits and write results to
        lb, ub = self.bounds_real_scale[i_x]
        xi_sweep = np.linspace(lb, ub, 100)
        power_cycle_norm, power_out_norm, g, active_g, duty_cycle, pumping_eff = [], [], [], [], [], []
        for xi in xi_sweep:
            x_full = list(x_real_scale)
            x_full[i_x] = xi

            try:
                res_eval = self.eval_fun(x_full, scale_x=False)
                kpis = self.eval_performance_indicators(x_full, scale_x=False)
                power_cycle_norm.append(kpis['average_power']['cycle']/power_cycle_ref)
                if kpis['average_power']['out']:
                    power_out_norm.append(kpis['average_power']['out']/power_out_ref)
                else:
                    power_out_norm.append(None)
                cons = res_eval[1][self.reduce_ineq_cons]
                g.append(res_eval[1])
                active_g.append(any([c < -1e-6 for c in cons]))
                duty_cycle.append(kpis['duty_cycle'])
                pumping_eff.append(kpis['pumping_efficiency'])
            except:
                power_cycle_norm.append(None), power_out_norm.append(None)
                duty_cycle.append(None), pumping_eff.append(None)
                g.append(None), active_g.append(False)

        fig, ax = plt.figure()
        ax.plot(xi_sweep, power_cycle_norm, '--', label='normalized cycle power')
        ax.plot(xi_sweep, power_out_norm, '--', label='normalized traction power')
        ax.plot(xi_sweep, duty_cycle, label='duty cycle')
        ax.plot(xi_sweep, pumping_eff, label='pumping efficiency')

        # Plot marker at the reference point.
        ax.plot(x_ref, 1, 'x', label=ref_point_label, markersize=12)
        ax.fill_between(xi_sweep, 0, 1, where=active_g, alpha=0.4, transform=ax.get_xaxis_transform())
        ax.set_xlabel(xlabel.replace('\n', ' '))
        ax.set_ylabel("Response [-]")
        ax.grid()

        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_title("v={:.1f}m/s".format(self.environment_state.wind_speed))
        plt.subplots_adjust(right=0.7)


class OptimizerCycle(Optimizer):
    """Tether force controlled cycle optimizer. Zero reeling speed is used as setpoint for transition phase."""
    OPT_VARIABLE_LABELS = [
        "Reel-out\nforce [N]",
        "Reel-in\nforce [N]",
        "Elevation\nangle [rad]",
        "Reel-in tether\nlength [m]",
        "Minimum tether\nlength [m]"
    ]
    X0_REAL_SCALE_DEFAULT = np.array([5000, 500, 0.523599, 120, 150])
    SCALING_X_DEFAULT = np.array([1e-4, 1e-4, 1, 1e-3, 1e-3])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan],
        [25*np.pi/180, 60.*np.pi/180.],
        [150, 250],
        [150, 250],
    ])

    def __init__(self, cycle_settings, system_properties, environment_state, reduce_x=None, reduce_ineq_cons=None):
        # Initiate attributes of parent class.
        bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        bounds[0, :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        bounds[1, :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        if reduce_ineq_cons is None:
            reduce_ineq_cons = np.arange(4)
        super().__init__(self.X0_REAL_SCALE_DEFAULT.copy(), bounds, self.SCALING_X_DEFAULT.copy(),
                         reduce_x, reduce_ineq_cons, system_properties, environment_state)

        # Set cycle settings after printing the settings that may be overruled by the optimization.
        cycle_settings.setdefault('cycle', {})
        cycle_keys = list(flatten_dict(cycle_settings))
        overruled_keys = []
        for k in ['cycle.elevation_angle_traction', 'cycle.tether_length_start_retraction',
                  'cycle.tether_length_end_retraction', 'retraction.control', 'transition.control', 'traction.control']:
            if k in cycle_keys:
                overruled_keys.append(k)
        if overruled_keys:
            print("Overruled cycle setting: " + ", ".join(overruled_keys) + ".")
        self.cycle_settings = cycle_settings

    def eval_fun(self, x, scale_x=True, **kwargs):
        """Method calculating the objective and constraint functions from the eval_performance_indicators method output.
        """
        # Convert the optimization vector to real scale values and perform simulation.
        if scale_x and self.scaling_x is not None:
            x_real_scale = x/self.scaling_x
        else:
            x_real_scale = x
        res = self.eval_performance_indicators(x_real_scale, **kwargs)
        # Prepare the simulation by updating simulation parameters.
        env_state = self.environment_state
        env_state.calculate(100.)
        power_wind_100m = .5 * env_state.air_density * env_state.wind_speed ** 3

        # Determine optimization objective and constraints.
        obj = -res['average_power']['cycle']/power_wind_100m/self.system_properties.kite_projected_area
        # When speed limits are active during the optimization (see determine_new_steady_state method of Phase
        # class in qsm.py), the setpoint reel-out/reel-in forces are overruled. For special cases, the respective
        # optimization variables won't affect the simulation. The lower constraints avoid random steps between
        # iterations and drives the variables towards an extremum value of the force during the respective phase.
        if res['min_tether_force']['out'] == np.inf:
            res['min_tether_force']['out'] = 0.
        force_out_setpoint_min = (res['min_tether_force']['out'] - x_real_scale[0])*1e-2 + 1e-6
        force_in_setpoint_max = (res['max_tether_force']['in'] - x_real_scale[1])*1e-2 + 1e-6

        # The maximum reel-out tether force can be exceeded when the tether force control is overruled by the maximum
        # reel-out speed limit and the imposed reel-out speed yields a tether force exceeding its set point. This
        # scenario is prevented by the lower constraint.
        force_max_limit = self.system_properties.tether_force_max_limit
        max_force_violation_traction = res['max_tether_force']['out'] - force_max_limit
        ineq_cons_traction_max_force = -max_force_violation_traction/force_max_limit + 1e-6

        # Constraint on the number of cross-wind patterns. It is assumed that a realistic reel-out trajectory should
        # include at least one crosswind pattern.
        if res["n_crosswind_patterns"] is not None:
            ineq_cons_cw_patterns = res["n_crosswind_patterns"] - 1
        else:
            ineq_cons_cw_patterns = 0.  # Constraint set to 0 does not affect the optimization.

        ineq_cons = np.array([force_out_setpoint_min, force_in_setpoint_max, ineq_cons_traction_max_force,
                              ineq_cons_cw_patterns])
        return obj, ineq_cons

    def eval_performance_indicators(self, x_real_scale, plot_result=False, relax_errors=True):
        """Method running the simulation and returning the performance indicators needed to calculate the objective and
        constraint functions."""
        # Map the optimization vector to the separate variables.
        tether_force_traction, tether_force_retraction, elevation_angle_traction, tether_length_diff, \
        tether_length_min = x_real_scale

        # Configure the cycle settings and run simulation.
        self.cycle_settings['cycle']['elevation_angle_traction'] = elevation_angle_traction
        self.cycle_settings['cycle']['tether_length_start_retraction'] = tether_length_min + tether_length_diff
        self.cycle_settings['cycle']['tether_length_end_retraction'] = tether_length_min

        self.cycle_settings['retraction']['control'] = ('tether_force_ground', tether_force_retraction)
        self.cycle_settings['transition']['control'] = ('reeling_speed', 0.)
        self.cycle_settings['traction']['control'] = ('tether_force_ground', tether_force_traction)

        cycle = Cycle(self.cycle_settings)
        iterative_procedure_config = {
            'enable_steady_state_errors': not relax_errors,
        }
        cycle.run_simulation(self.system_properties, self.environment_state, iterative_procedure_config,
                             not relax_errors)

        if plot_result:  # Plot the simulation results.
            cycle.trajectory_plot(steady_state_markers=True)
            phase_switch_points = [cycle.transition_phase.time[0], cycle.traction_phase.time[0]]
            cycle.time_plot(['straight_tether_length', 'reeling_speed', 'tether_force_ground', 'power_ground'],
                            plot_markers=phase_switch_points)

        res = {
            'average_power': {
                'cycle': cycle.average_power,
                'in': cycle.retraction_phase.average_power,
                'trans': cycle.transition_phase.average_power,
                'out': cycle.traction_phase.average_power,
            },
            'min_tether_force': {
                'in': cycle.retraction_phase.min_tether_force,
                'trans': cycle.transition_phase.min_tether_force,
                'out': cycle.traction_phase.min_tether_force,
            },
            'max_tether_force': {
                'in': cycle.retraction_phase.max_tether_force,
                'trans': cycle.transition_phase.max_tether_force,
                'out': cycle.traction_phase.max_tether_force,
            },
            'min_reeling_speed': {
                'in': cycle.retraction_phase.min_reeling_speed,
                'out': cycle.traction_phase.min_reeling_speed,
            },
            'max_reeling_speed': {
                'in': cycle.retraction_phase.max_reeling_speed,
                'out': cycle.traction_phase.max_reeling_speed,
            },
            'n_crosswind_patterns': getattr(cycle.traction_phase, 'n_crosswind_patterns', None),
            'min_height': min([cycle.traction_phase.kinematics[0].z, cycle.traction_phase.kinematics[-1].z]),
            'max_elevation_angle': cycle.transition_phase.kinematics[0].elevation_angle,
            'duration': {
                'cycle': cycle.duration,
                'in': cycle.retraction_phase.duration,
                'trans': cycle.transition_phase.duration,
                'out': cycle.traction_phase.duration,
            },
            'duty_cycle': cycle.duty_cycle,
            'pumping_efficiency': cycle.pumping_efficiency,
            'kinematics': cycle.kinematics,
        }
        return res


def test():
    from qsm import LogProfile, TractionPhaseHybrid
    from kitepower_kites import sys_props_v3

    import time
    since = time.time()
    env_state = LogProfile()
    env_state.set_reference_wind_speed(12.)

    cycle_sim_settings = {
        'cycle': {
            'traction_phase': TractionPhaseHybrid,
            'include_transition_energy': False,
        },
        'retraction': {},
        'transition': {
            'time_step': 0.25,
        },
        'traction': {
            'azimuth_angle': 13 * np.pi / 180.,
            'course_angle': 100 * np.pi / 180.,
        },
    }
    oc = OptimizerCycle(cycle_sim_settings, sys_props_v3, env_state, reduce_x=np.array([0, 1, 2, 3]))
    oc.x0_real_scale = np.array([4500, 1000, 30*np.pi/180., 150, 230])
    print('Optimization on:', oc.x0_real_scale)
    print(oc.optimize())
    oc.eval_point(True)
    time_elapsed = time.time() - since
    print('Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #optHist = History(optimizer_history_file_name)
    #print(optHist.getValues(major=True, scale=False, stack=False, allowSens=True)['isMajor'])

    plt.show()


if __name__ == "__main__":
    test()
