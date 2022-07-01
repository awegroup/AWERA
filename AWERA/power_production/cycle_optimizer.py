from pyoptsparse import Optimization, SLSQP, Gradient
from pyoptsparse import History
from scipy import optimize as op
import copy
import numpy as np
import matplotlib as mpl
#mpl.use('Pdf')
#TODO inlclude config
import matplotlib.pyplot as plt

from .qsm import Cycle, \
    SteadyStateError, OperationalLimitViolation, PhaseError
from scipy.stats import truncnorm

from .utils import flatten_dict

# TODO use optimizer history?
# from .config import optimizer_history_file_name


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
        x_vals = [v for k, v in x.items()]
        if np.isnan(x_vals).any():
            raise OptimizerError("Optimization vector contains NaN's.")

        if self.reduce_x is not None:
            x_full = self.x0.copy()
            x_full[self.reduce_x] = x_vals
        else:
            x_full = x_vals

        if 3 in self.reduce_x and 5 in self.reduce_x:
            # Optimising on both elevation angle and powering C_L,C_D
            # Only optimise over powering after reaching max elevation angle
            elevation_angle_traction = x_full[2]
            max_elev_border = self.bounds_real_scale[2, 1]*self.scaling_x[2] * 0.95
            if elevation_angle_traction - max_elev_border < 0:
                # Set powering random starting value to 1
                x_full[5] = 1

        bounds_adhered = (x_full - self.bounds_real_scale[:, 0]*self.scaling_x >= -1e-3).all() and \
                         (x_full - self.bounds_real_scale[:, 1]*self.scaling_x <= 1e-3).all()
        if not bounds_adhered:

            print('Within-Opt Optimization bounds violated, precision 1e-3.'
                  ' Diff to bounds:')
            lower = (x_full - self.bounds_real_scale[:, 0]*self.scaling_x >= -1e-3)
            upper = (x_full - self.bounds_real_scale[:, 1]*self.scaling_x <= 1e-3)
            print('lower:', lower, 'upper:', upper)
            # Set violated values to bounds, simulation is very sensitive to this:
            overwrite_lower = np.logical_not(lower)
            overwrite_upper = np.logical_not(upper)
            x_full[overwrite_lower] = (self.bounds_real_scale[:, 0]*self.scaling_x)[overwrite_lower]
            x_full[overwrite_upper] = (self.bounds_real_scale[:, 1]*self.scaling_x)[overwrite_upper]
            # Test
            lower = (x_full - self.bounds_real_scale[:, 0]*self.scaling_x >= -1e-3)
            upper = (x_full - self.bounds_real_scale[:, 1]*self.scaling_x <= 1e-3)
            print('Test', 'lower:', lower, 'upper:', upper)
            # raise OptimizerError("Optimization bounds violated.")

        obj, ineq_cons = self.eval_fun(x_full, *args)
        print('x_full, ineq_cons', x_full, ineq_cons)
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

    def optimize(self, *args, maxiter=80, iprint=-1):  # 0):
        """Perform optimization."""
        self.clear_result_attributes()

        bounds_adhered = (self.x0_real_scale - self.bounds_real_scale[:, 0] >= -1e-3).all() and \
                         (self.x0_real_scale - self.bounds_real_scale[:, 1] <= 1e-3).all()
        if not bounds_adhered:
            print('Real scale x0 bounds violated, precision 1e-3.'
                  ' Diff to bounds:')
            lower = (self.x0_real_scale - self.bounds_real_scale[:, 0] >= -1e-3)
            upper = (self.x0_real_scale - self.bounds_real_scale[:, 1] <= 1e-3)
            print('lower:', lower, 'upper:', upper)
            # Set violated values to bounds, simulation is very sensitive to this:
            overwrite_lower = np.logical_not(lower)
            overwrite_upper = np.logical_not(upper)
            self.x0_real_scale[overwrite_lower] = self.bounds_real_scale[:, 0][overwrite_lower]
            self.x0_real_scale[overwrite_upper] = self.bounds_real_scale[:, 1][overwrite_upper]

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

        print_details = False
        # TODO add in options
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
                'iprint': iprint,  # 1: Show final summary
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
                op_problem.addVar('x{}'.format(i_x), 'c', lower=b[0],
                                  upper=b[1], value=xi0)

            for i_c in self.reduce_ineq_cons:
                op_problem.addCon('g{}'.format(i_c), lower=0, upper=1e10)
                # force_out_setpoint_min, force_in_setpoint_max, ineq_cons_traction_max_force, ineq_cons_cw_patterns
            if self.use_parallel_processing:
                sens_mode = 'pgc'  # TODO pyOptSparse implementation?
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
            nit, nfev, njev = op_sol.userObjCalls, self.op_eval_func_calls, op_sol.userSensCalls    # TODO old: read_slsqp_output_file(print_details) from iprint = 1

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
            # TODO obsolete?
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

        fig, ax = plt.subplots()
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
        "Minimum tether\nlength [m]",
        "Kite traction powering scale [-]",
    ]
    X0_REAL_SCALE_DEFAULT = np.array([5000, 500, 0.523599, 120, 150, 1])
    SCALING_X_DEFAULT = np.array([1e-4, 1e-4, 1, 1e-3, 1e-3, 1])
    BOUNDS_REAL_SCALE_DEFAULT = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan],
        [25*np.pi/180, 60.*np.pi/180.],
        [150, 300],
        [200, 250],
        [0, 1],
    ])

    def __init__(self, cycle_settings, system_properties,
                 environment_state, reduce_x=None,
                 reduce_ineq_cons=None,
                 bounds=[None],
                 print_details=False):
        # Initiate attributes of parent class.
        if bounds is None:
            bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
        else:
            default_bounds = self.BOUNDS_REAL_SCALE_DEFAULT.copy()
            for i in range(len(bounds)):
                if bounds[i] is not None:
                    default_bounds[i] = bounds[i]
            bounds = default_bounds

        if np.any(np.isnan(bounds[0, :])):
            bounds[0, :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        if np.any(np.isnan(bounds[1, :])):
            bounds[1, :] = [system_properties.tether_force_min_limit, system_properties.tether_force_max_limit]
        if reduce_ineq_cons is None:
            reduce_ineq_cons = np.arange(8)  # 10)  # Inequality constraints
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

        self.set_max_traction_power = 20000.  # TODO make optional in config
        self.random_x0 = False  # Randomize starting x0 (over bounds gaussian selection)
        self.smear_x0 = False  # Randomize starting x0 (gaussian smearing)

        # TODO remove? reorganise?
        self.optimization_rounds = {'total_opts': [],
                                    'successful_opts': [],
                                    'opt_and_sim_successful': [],
                                    'unstable_results': [],
                                    'second_wind_speed_test': [],
                                    'optimizer_error_starting_vals': [],
                                    'optimizer_error_wind_speed': []}
        self.print_details = print_details

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
        if res['generator']['eff']['cycle'] is not None:
            obj = obj * res['generator']['eff']['cycle']  # <0
        else:
            # Favor feasible and large efficiencies if cycle eff not calculated
            obj_effs = 1
            if res['generator']['eff']['in'] not in [0, None]:
                obj_effs = obj_effs / res['generator']['eff']['in']
            if res['generator']['eff']['out'] not in [0, None]:
                obj_effs = obj_effs / res['generator']['eff']['out']
            obj = 1/copy.deepcopy(obj_effs)  # >0

        # When speed limits are active during the optimization (see determine_new_steady_state method of Phase
        # class in qsm.py), the setpoint reel-out/reel-in forces are overruled. For special cases, the respective
        # optimization variables won't affect the simulation. The lower constraints avoid random steps between
        # iterations and drives the variables towards an extremum value of the force during the respective phase.
        if res['min_tether_force']['out'] == np.inf:
            res['min_tether_force']['out'] = 0.
        if res['min_tether_force']['in'] == np.inf:
            res['min_tether_force']['in'] = 0.
        force_out_setpoint_max = (res['max_tether_force']['out'] - x_real_scale[0])*1e-3 # + 1e-6 # was *e-2
        force_out_setpoint_min = (-res['min_tether_force']['out'] + x_real_scale[0])*1e-3 # + 1e-6 # was *e-2
        force_in_setpoint_max = (res['max_tether_force']['in'] - x_real_scale[1])*1e-3 # + 1e-6 # was *e-2
        force_in_setpoint_min = (-res['min_tether_force']['in'] + x_real_scale[1])*1e-3 # + 1e-6 # was *e-2

        # The maximum reel-out tether force can be exceeded when the tether force control is overruled by the maximum
        # reel-out speed limit and the imposed reel-out speed yields a tether force exceeding its set point. This
        # scenario is prevented by the lower constraint.
        force_max_limit = self.system_properties.tether_force_max_limit
        max_force_violation_traction = res['max_tether_force']['out'] - force_max_limit
        ineq_cons_traction_max_force = -max_force_violation_traction*1e-3 - self.precision  # same req. as in QSM: absolute difference maximal 1e-3 , no relative error: /force_max_limit

        # Constraint on the reeling speed control versus force control
        # print(res['speed_viol_diff'])

        # Constraint on lambda, the reeling factor in reel-in and reel-out phase which are to be strictly positive
        min_reeling_factor_in = res['min_reeling_factor']['in'] - self.precision
        min_reeling_factor_out = res['min_reeling_factor']['out'] - self.precision

        # Constraint on the number of cross-wind patterns. It is assumed that a realistic reel-out trajectory should
        # include at least one crosswind pattern.
        if res["n_crosswind_patterns"] is not None:
            ineq_cons_cw_patterns = res["n_crosswind_patterns"] - 1
        else:
            ineq_cons_cw_patterns = 0.  # Constraint set to 0 does not affect the optimization.

        # Generator efficiency constraint: all efficiencies greater than 0
        # Rather: Constraints for load and frequency violation:
        max_load = res['generator']['load_bounds'][1]
        max_freq = res['generator']['freq_bounds'][1]
        load_violation = 0
        for phase, load in res['generator']['load'].items():
            if load is not None:
                violation = max_load - load - self.precision
                load_violation = np.min((load_violation, violation))
        freq_violation = 0
        for phase, freq in res['generator']['freq'].items():
            if freq is not None:
                violation = max_freq - freq - self.precision
                freq_violation = np.min((freq_violation, violation))

        max_duration = 360  # 6 min
        # # TODO set as option in settings file, ...?
        duration_violation = max_duration - res['duration']['cycle']

        # Constraint: Working traction phase
        if res['average_power']['out'] is None:
            trac_phase_working = -1
        else:
            trac_phase_working = 0
        # Constrain the sequence of depowering: first elevation angle
        # and when within 5% of the maximum elevation angle, allow depowering
        elevation_angle_traction = x_real_scale[2]
        max_elev_border = self.bounds_real_scale[2][1] * 0.95
        powering = x_real_scale[5]
        if elevation_angle_traction - max_elev_border < 0 and powering < 1:
            allow_depowering = -1/powering
        else:
            allow_depowering = 0
        # TODO add ineq constraint: less maximum power
        # TODO default max reel out force * max reel out speed
        # TODO set explicitly: check agains maximum traction power
        #max_power_cons = res['average_power']['out'] - self.set_max_traction_power
        #print('Average traction (out) power per cycle: ', max_power_cons)
        ineq_cons = np.array([force_out_setpoint_max,
                              force_out_setpoint_min,
                              force_in_setpoint_max,
                              force_in_setpoint_min,
                              duration_violation,
                              allow_depowering,
                              ineq_cons_cw_patterns,
                              ineq_cons_traction_max_force,
                              trac_phase_working,
                              min_reeling_factor_in,
                              min_reeling_factor_out,
                              res['speed_viol_diff']['in'],
                              res['speed_viol_diff']['out'],
                              load_violation,
                              freq_violation
                              ])
        if self.print_details:
            print('inequality constraints: max f_out, min f_out, max f_in,'
                  ' min f_in, duration violation,'
                  ' allow depowering,'
                  ' n_cwp, '
                  'max f_out_sys, reeling factor in/out,'
                  ' speed_viol_diff in/out,'
                  'load violation, freq violation - obj ',
                  ineq_cons[:7], ineq_cons[7:], obj)

        return obj, ineq_cons[7:]

    def eval_performance_indicators(self, x_real_scale, plot_result=False, relax_errors=True):
        """Method running the simulation and returning the performance indicators needed to calculate the objective and
        constraint functions."""
        # Map the optimization vector to the separate variables.
        tether_force_traction, tether_force_retraction, \
            elevation_angle_traction, tether_length_diff, \
            tether_length_min, powering_traction = x_real_scale

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
        self.system_properties.kite_powering_traction = powering_traction

        cycle.run_simulation(self.system_properties, self.environment_state, iterative_procedure_config,
                             not relax_errors)

        if plot_result:  # Plot the simulation results.
            cycle.trajectory_plot(steady_state_markers=True)
            phase_switch_points = [cycle.transition_phase.time[0], cycle.traction_phase.time[0]]
            cycle.time_plot(['straight_tether_length', 'reeling_speed', 'tether_force_ground', 'power_ground'],
                            plot_markers=phase_switch_points)

        env_state = self.environment_state
        power_wind_trac = []
        for kin in cycle.traction_phase.kinematics:
            env_state.calculate(kin.z)
            power_wind_trac.append(
                .5 * env_state.air_density * env_state.wind_speed ** 3)

        res = {
            'average_power': {
                'cycle': cycle.average_power,
                'in': cycle.retraction_phase.average_power,
                'trans': cycle.transition_phase.average_power,
                'out': cycle.traction_phase.average_power,
                },
            'generator': {
                'load_bounds': getattr(cycle, 'load_bounds', [0, np.inf]),
                'freq_bounds': getattr(cycle, 'freq_bounds', [0, np.inf]),
                'eff': {
                    'cycle': cycle.eff_winch,
                    'in': getattr(cycle.retraction_phase,
                                  'gen_eff', None),
                    'trans': getattr(cycle.transition_phase,
                                     'gen_eff', None),
                    'out': getattr(cycle.traction_phase,
                                   'gen_eff', None),
                    },
                'load': {
                    'in': getattr(cycle.retraction_phase,
                                  'gen_load', None),
                    'trans': getattr(cycle.transition_phase,
                                     'gen_load', None),
                    'out': getattr(cycle.traction_phase,
                                   'gen_load', None),
                    },
                'freq': {
                    'in': getattr(cycle.retraction_phase,
                                  'gen_freq', None),
                    'trans': getattr(cycle.transition_phase,
                                     'gen_freq', None),
                    'out': getattr(cycle.traction_phase,
                                   'gen_freq', None),
                    },
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
            'n_crosswind_patterns': getattr(cycle.traction_phase,
                                            'n_crosswind_patterns', None),
            # TODO add max height / average heigt? optimal harvesting height
            'min_height': min([cycle.traction_phase.kinematics[0].z,
                               cycle.traction_phase.kinematics[-1].z]),
            'average_traction_height': cycle.avg_traction_height,
            'wind_speed_at_avg_traction_height':
                cycle.wind_speed_at_avg_traction_height,
            'power_dens_wind_trac': power_wind_trac,
            'min_reeling_factor': {
                'in': np.min([ss.tangential_speed_factor
                              for ss in cycle.retraction_phase.steady_states]),
                'out': np.min([ss.tangential_speed_factor
                               for ss in cycle.traction_phase.steady_states]),
            },
            'max_elevation_angle':
                cycle.transition_phase.kinematics[0].elevation_angle,
            'duration': {
                'cycle': cycle.duration,
                'in': cycle.retraction_phase.duration,
                'trans': cycle.transition_phase.duration,
                'out': cycle.traction_phase.duration,
            },
            'duty_cycle': cycle.duty_cycle,
            'pumping_efficiency': cycle.pumping_efficiency,
            'kinematics': cycle.kinematics,
            'traction_kinematics': cycle.traction_phase.kinematics,
            'speed_viol_diff': {
                'in': cycle.retraction_phase.speed_viol_diff,
                'out': cycle.traction_phase.speed_viol_diff,
            },
        }
        return res

    def run_optimization(self,
                         x0,
                         second_attempt=False,
                         save_initial_value_scan_output=True,
                         n_x_test=2, test_until_n_succ=3):
        # TODO set save scan output to False by default
        # TODO log? print("x0:", x0)

        # Optimize around x0
        # perturb x0:
        print('START optimisation')
        x0_range = [x0]
        x0_range_random = []
        # Optimization variables bounds defining the search space.
        bounds = self.bounds_real_scale
        reduce_x = self.reduce_x
        print('x0:', x0)
        x0 = np.array(x0)
        def reset_x0_to_bounds(x0):
            x0 = np.array(x0)
            lower = (x0 - self.bounds_real_scale[:, 0] >= -1e-3)
            upper = (x0 - self.bounds_real_scale[:, 1] <= 1e-3)
            print('lower:', lower, 'upper:', upper)
            # Set violated values to bounds, simulation is very sensitive to this:
            overwrite_lower = np.logical_not(lower)
            overwrite_upper = np.logical_not(upper)
            x0[overwrite_lower] = (self.bounds_real_scale[:, 0])[overwrite_lower]
            x0[overwrite_upper] = (self.bounds_real_scale[:, 1])[overwrite_upper]
            return x0
        x0 = reset_x0_to_bounds(x0)

        def get_smeared_x0():
            return [np.random.normal(x0[i], x0[i]*smearing)
                    if i in reduce_x else x0[i] for i in range(len(x0))]

        def test_smeared_x0(test_x0, precision=0, return_bounds_adhered=False):
            bounds_adhered = [np.logical_and(
                test_x0[i] >= (bounds[i][0]-precision),
                test_x0[i] <= (bounds[i][1]+precision))
                for i in range(len(test_x0))]
            if return_bounds_adhered:
                return np.all(bounds_adhered), bounds_adhered
            else:
                return np.all(bounds_adhered)

        def smearing_x0():
            n_smearing = 0
            test_smearing = get_smeared_x0()
            bounds_adhered = test_smeared_x0(test_smearing)
            while not bounds_adhered:
                n_smearing += 1
                print('smearing, bounds adhered: ', bounds_adhered)
                test_smearing = get_smeared_x0()
                bounds_adhered = test_smeared_x0(test_smearing)
                if n_smearing > 10:
                    print('Resetting smeared x0 to bounds')
                    test_smearing = reset_x0_to_bounds(test_smearing)
                    break

            return test_smearing
        print('Define x0 to test...')
        for n_test in range(n_x_test):
            print(n_test)
            if self.random_x0:
                # Gaussian random selection of x0 within bounds
                # TODO or just use uniform distr, mostly at bounds anyways...?
                x0_range_random.append([truncnorm(a=bounds[i][0]/bounds[i][1],
                                                  b=1, scale=bounds[i][1]).rvs()
                                        if i in reduce_x else x0[i]
                                        for i in range(len(x0))])
            if self.smear_x0:
                print('smearing')
                # Gaussian random smearing of x0 within bounds
                smearing = 0.05  # 10% smearing of the respective values

                # Test on two smeared variations of x0
                x0_range_random.append(smearing_x0())
                print('1/2 smearing')
                smearing = 0.1
                x0_range_random.append(smearing_x0())
                print('2/2 smearing')
                # Constrain the sequence of depowering: first elevation angle
                # and when within 5% of the maximum elevation angle, allow depowering
                # elevation_angle_traction = x0[2]
                # max_elev_border = bounds[2][1] * 0.95
                # if elevation_angle_traction - max_elev_border < 0:
                #     # Set powering random starting value to 1
                #     x0_range_random[-1][5] = 1
                #     x0_range_random[-2][5] = 1
                x0_range_random[-2][0] = bounds[0][1]

        # Add custom x0 from previous results:
        vw = self.environment_state.wind_speed_ref
        if vw >= 10:
            wind_speed_steps = [10, 16, 30]
            x0_vs_wind_speeds = np.array([
                [bounds[0][1], 2000, 0.7, bounds[3][1], 200, 1],
                [bounds[0][1], 5000, bounds[2][1], bounds[3][1], 200, 0.8],
                [bounds[0][1], 8000, bounds[2][1], bounds[3][1], 200, 0.8]])
            if bounds[0][1] > 100000:
                # larger (500kW) system
                print('shape', x0_vs_wind_speeds.shape)
                x0_vs_wind_speeds[:, 1] = (8000, 15000, 25000)
            # Interpolation
            preset_x0 = [np.interp(vw, wind_speed_steps,
                                   x0_vs_wind_speeds[:, i]) for i in range(len(x0))]
            x0_range.append(preset_x0)

        x0_range += x0_range_random
        print(x0_range)
        x0_range = np.array(x0_range)
        # TODO log? print('Testing x0 range: ', x0_range)
        n_x0 = x0_range.shape[0]
        x_opts = []
        op_ress = []
        conss = []
        kpiss = []
        sim_successfuls = []
        opt_successfuls = []
        if self.print_details:
            print('optimise ...')
        for i in range(n_x0):
            if self.print_details:
                print('|     {}/{}     |'.format(i+1, n_x0))
            x0_test = x0_range[i]
            self.x0_real_scale = x0_test
            try:
                # TODO log? print("Testing the {}th starting values:
                # {}".format(i,
                #                                                    x0_test))
                x_opts.append(self.optimize())
                test_passed, bounds_adhered = test_smeared_x0(
                    self.x_opt_real_scale,
                    precision=self.precision, return_bounds_adhered=True)
                if test_passed:
                    # Safety check if variable bounds are adhered
                    op_ress.append(self.op_res)
                    opt_successfuls.append(True)
                    try:
                        cons, kpis = self.eval_point()
                        if not np.all(cons >= -self.precision):
                            x_opts = x_opts[:-1]
                            op_ress = op_ress[:-1]
                            opt_successfuls = opt_successfuls[:-1]
                            raise OptimizerError('Final constraints do not'
                                                 ' fulfill bounds: >=0 within'
                                                 ' precision.')
                        conss.append(cons)
                        kpiss.append(kpis)
                        sim_successfuls.append(True)
                        # TODO log? print('Simulation successful')
                        if sum(sim_successfuls) == test_until_n_succ:
                            x0_range = x0_range[:i+1]
                            print(self.environment_state.wind_speed_ref, i+1, 'successful. break.', kpis['average_power']['cycle'])
                            break
                    except (SteadyStateError, OperationalLimitViolation,
                            PhaseError) as e:
                        print("Error occurred while evaluating the"
                              "resulting optimal point: {}".format(e))
                        # Evaluate results with relaxed errors
                        # relaxed errors only relax OperationalLimitViolation
                        cons, kpis = self.eval_point(
                            relax_errors=True)
                        conss.append(cons)
                        if not np.all(cons >= -self.precision):
                            x_opts = x_opts[:-1]
                            op_ress = op_ress[:-1]
                            opt_successfuls = opt_successfuls[:-1]
                            raise OptimizerError('Final constraints do not'
                                                 ' fulfill bounds: >=0 within'
                                                 ' precision.')
                        kpiss.append(kpis)
                        sim_err = e
                        sim_successfuls.append(False)
                        continue
                        # TODO log?print('Simulation failed -> errors relaxed')
                else:
                    print("Optimization number "
                           "{} finished with an error: {}".format(
                               i+1, 'Optimization bounds violated'))
                    # Drop last x_opts, bonds are not adhered
                    x_opts = x_opts[:-1]
                    # TODO remove?
                    raise OptimizerError('Optimization bounds violated '
                                         'in final result')

            except (OptimizerError) as e:
                print("Optimization number "
                      "{} finished with an error: {}".format(i+1, e))
                opt_err = e
                opt_successfuls.append(False)
                continue
            except (SteadyStateError, PhaseError,
                    OperationalLimitViolation) as e:
                print("Optimization number "
                      "{} finished with a simulation error: {}".format(i+1, e))
                opt_err = e
                opt_successfuls.append(False)
                continue
            except (FloatingPointError) as e:
                print("Optimization number "
                      "{} finished due to a mathematical simulation error: {}"
                      .format(i+1, e))
                opt_err = e
                opt_successfuls.append(False)
                continue
            print(self.environment_state.wind_speed_ref, i+1, 'successful.', kpis['average_power']['cycle'])


        # TODO Include test for correct power?
        # TODO Output handling different? own function?

        self.optimization_rounds['total_opts'].append(
            len(opt_successfuls))
        self.optimization_rounds['successful_opts'].append(
            sum(opt_successfuls))
        self.optimization_rounds['opt_and_sim_successful'].append(
            sum(sim_successfuls))  # good results
        # TODO remoe second attempt parameter, useless?
        self.optimization_rounds['second_wind_speed_test'].append(
            second_attempt)

        # if save_initial_value_scan_output:
        #    # TODO log? print('Saving optimizer scan output')
        #    # TODO scan optimizer output / sim results to file
        #    # - dep on wind_speed

        #    #TODO independent of this: optvis history output?

        if sum(sim_successfuls) > 0:
            # Optimization and Simulation successful at least once:
            # append to results
            # consistency check sim results - both optimization
            # and simulation work
            x0_success = x0_range[opt_successfuls][sim_successfuls]
            # x0_failed = list(x0_range[np.logical_not(opt_successfuls)])
            # + list(x0_range[opt_successfuls][np.logical_not(sim_successfuls)])
            # print('Failed starting values: ', x0_failed)
            # print('Successful starting values: ', x0_success)

            # consistency check function values
            # corresponding eval function values from the optimizer
            flag_unstable_opt_result = False

            # print('Optimizer x point results: ', x_opts)
            # print(' Leading to a successful simulation:', sim_successfuls)
            x_opts_succ = np.array(x_opts)[sim_successfuls]
            (x_opt_mean, x_opt_std) = (np.mean(x_opts_succ, axis=0),
                                       np.std(x_opts_succ, axis=0))
            # print('  The resulting mean {} with a standard deviation of {}'
            # .format(x_opt_mean, x_opt_std))
            if (x_opt_std > np.abs(0.1*x_opt_mean)).any():
                # TODO: lower/higher, different check? -
                # make this as debug output?
                # print('  More than 1% standard deviation - unstable result')
                flag_unstable_opt_result = True

            # corresponding eval function values from the optimizer
            op_ress_succ = [op_ress[i] for i in range(len(op_ress))
                            if sim_successfuls[i]]
            f_opt = [op_res['fun'] for op_res in op_ress_succ]
            # TEST
            def get_obj(res):
                env_state = self.environment_state
                env_state.calculate(100.)
                power_wind_100m = .5 * env_state.air_density * env_state.wind_speed ** 3

                # Determine optimization objective and constraints.
                obj = -res['average_power']['cycle']/power_wind_100m/self.system_properties.kite_projected_area
                if res['generator']['eff']['cycle'] is not None:
                    obj = obj * res['generator']['eff']['cycle']  # <0
                else:
                    # Favor feasible and large efficiencies if cycle eff not calculated
                    obj_effs = 1
                    if res['generator']['eff']['in'] not in [0, None]:
                        obj_effs = obj_effs / res['generator']['eff']['in']
                    if res['generator']['eff']['out'] not in [0, None]:
                        obj_effs = obj_effs / res['generator']['eff']['out']
                    obj = 1/copy.deepcopy(obj_effs)  # >0
                return obj

            f_opt_test = [get_obj(kpiss[i]) for i in range(len(kpiss))
                          if sim_successfuls[i]]
            print('test objective:', f_opt, f_opt_test)
            # print('Successful optimizer eval function results: ', f_opt)
            (f_opt_mean, f_opt_std) = (np.mean(f_opt), np.std(f_opt))
            # print('  The resulting mean {} with a standard deviation of {}'
            # .format(f_opt_mean, f_opt_std))
            if f_opt_std > np.abs(0.1*f_opt_mean):
                # print('  More than 1% standard deviation - unstable result')
                flag_unstable_opt_result = True

            self.optimization_rounds['unstable_results'].append(
                flag_unstable_opt_result)

            # Chose best optimization result:
            # Matching index in sim_successfuls
            minimal_f_opt = np.argmin(f_opt)

            cons = [conss[i] for i in range(len(kpiss))
                    if sim_successfuls[i]][minimal_f_opt]
            kpis = [kpiss[i] for i in range(len(kpiss))
                    if sim_successfuls[i]][minimal_f_opt]
            # TODO log? print("cons:", cons)
            # TODO remove / manage? Failed simulation results are later masked
            kpis['sim_successful'] = True

            res = (
                x0_success[minimal_f_opt],
                x_opts_succ[minimal_f_opt],
                op_ress_succ[minimal_f_opt],
                cons,
                kpis)

            return res  # x0, x_opt result, op_res, cons, kpis

        # TODO fix handling of failed optimisation/simulations?
        # return something?
        # return via error?

        elif sum(opt_successfuls) > 0:
            # simulations failed (run again with loose errors) but
            # optimization worked
            # TODO log? print('All simulations failed, save flagged
            # loose error simulation output')
            # self.x0.append(x0_range[opt_successfuls][-1])
            # self.x_opts.append(x_opts[-1])
            # self.optimization_details.append(op_ress[-1])

            # TODO log? print("cons:", conss[-1])
            # self.constraints.append(conss[-1])
            # Failed simulation results are later masked
            # kpis = kpiss[-1]
            # kpis['sim_successful'] = False
            # self.performance_indicators.append(kpis)

            # TODO log? print('Output appended, raise simulation error: ')
            raise sim_err
        else:
            # optimizatons all failed
            # TODO remove?
            self.optimization_rounds['optimizer_error_starting_vals'].append(
                x0_range)
            self.optimization_rounds['optimizer_error_wind_speed'].append(
                self.environment_state.wind_speed)
            print('All optimizations for this wind speed '
                  'failed, raise optimizer error.')
            raise opt_err


def test():
    from .qsm import LogProfile, TractionPhaseHybrid
    from .kitepower_kites import sys_props_v3

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
    oc = OptimizerCycle(cycle_sim_settings, sys_props_v3, env_state, reduce_x=np.array([0, 1, 2, 3, 5]))
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
