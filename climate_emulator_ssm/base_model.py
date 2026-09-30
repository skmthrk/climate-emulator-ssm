import numpy as np
import scipy.linalg
import time
from collections import OrderedDict
import numdifftools as nd
import warnings

# Import KalmanFilter implementation
from .kalman_filter import KalmanFilter, ensure_psd

# this is important!
default_parameters_bounds = {
    'lamda_su': (0.000000001, 2),
    'lamda_lu': (1, 3),
    'gamma_su': (0.1, 50),
    'gamma_lu': (0.1, 50),
    'sigma_su': (0.1, 7),
    'sigma_lu': (0.1, 7),
    'sigma_s': (0.1, 7),
    'chi_s': (0.5, 15),
    'chi_o': (1, 500),
    'chi_d': (10, 500),
    'chi_b': (10, 1000),
    'kappa_o': (0.1, 5),
    'kappa_d': (0.1, 5),
    'kappa_b': (0.1, 2000),
}

parameters_bounds = {
    'lamda_su': (None, None),
    'lamda_lu': (None, None),
    'gamma_su': (None, None),
    'gamma_lu': (None, None),
    'sigma_su': (None, None),
    'sigma_lu': (None, None),
    'sigma_s': (None, None),
    'chi_s': (None, None),
    'chi_o': (None, None),
    'chi_d': (None, None),
    'chi_b': (None, None),
    'kappa_o': (None, None),
    'kappa_d': (None, None),
    'kappa_b': (None, None),
}

MAX_RATE = 1e12  # per year; see Model.discretize_matrices

def discretize_noise(A, V, dt=1.0):
    """Discrete process noise covariance Vd = int_0^dt exp(As) V exp(A's) ds.

    Van Loan's formula applied over the whole step needs exp(-A dt), whose
    entries grow like exp(|eigenvalue| dt) and destroy the result through
    cancellation once the system has a fast mode (|eigenvalue| of roughly 20
    per year or more). Instead, apply it over a short step h = dt / 2^k with
    ||A h|| <= 1/4 and double the interval k times with
    Vd(2h) = Vd(h) + exp(Ah) Vd(h) exp(Ah)', which only adds positive
    semidefinite terms.
    """
    m = A.shape[0]
    norm = np.abs(A).sum(axis=1).max() * dt
    k = max(0, int(np.ceil(np.log2(norm / 0.25)))) if norm > 0 else 0
    h = dt / 2 ** k
    F = np.block([[-A, V], [np.zeros((m, m)), A.T]]) * h
    expF = scipy.linalg.expm(F)
    Vd = expF[m:, m:].T @ expF[:m, m:]
    Phi = expF[m:, m:].T  # exp(A h)
    for _ in range(k):
        Vd = Vd + Phi @ Vd @ Phi.T
        Phi = Phi @ Phi
    if not np.all(np.isfinite(Vd)):
        raise FloatingPointError('non-finite noise covariance')
    return (Vd + Vd.T) / 2


class Model:
    """
    Base class for models to be estimated using the Kalman filter.
    """

    def __init__(self, parameters_bounds=parameters_bounds):
        """Initialize the model with default parameters."""

        # Placeholder for model parameters
        self.parameters = []

        # Default parameter values
        self.parameters_default = OrderedDict([])

        # Pysics-informed parameter bounds
        self.parameters_bounds = parameters_bounds

        # Pysics-informed parameter bounds
        self.default_parameters_bounds = default_parameters_bounds

        # Kalman filter instance
        self.kf = KalmanFilter()

        # Default optimization settings
        self.optimization_methods = ['BFGS', 'Powell', 'COBYQA', 'BOBYQA', 'SLSQP', 'Nelder-Mead']
        self.optimization_attempts = 2
        self.optimization_maxiter = 1000
        self.optimization_tol = 1e-5
        #self.optimization_tol = 1e-8

        # Results storage
        self.optimization_results = {}
        self.best_method = None
        self.best_parameters = None
        self.best_fvalue = float('inf')
        self.parameter_std_errors = None

    def build_matrices(self, parameters):
        """
        Build and discretize system matrices from model parameters.
        Must be implemented by subclasses.

        Parameters:
        parameters (list): Model parameters

        Returns:
        tuple: (x0, P0, Ad, Bd, Vd, C, w, W) - Discrete-time matrices
        """
        raise NotImplementedError("Subclasses must implement build_matrices()")

    def discretize_matrices(self, A, B, C, V):
        """
        Convert continuous-time system matrices to discrete-time.

        Parameters:
        A (numpy.ndarray): Continuous-time state matrix
        B (numpy.ndarray): Continuous-time input matrix
        C (numpy.ndarray): Observation matrix
        V (numpy.ndarray): Continuous-time process noise covariance

        Returns:
        tuple: (x0, P0, Ad, Bd, Vd, C, w, W) - Discrete-time matrices
        """
        m = A.shape[0]  # State dimension
        if not all(np.all(np.isfinite(M)) for M in (A, B, C, V)):
            raise FloatingPointError('non-finite system matrices')
        # Rates far beyond anything physical (here 1e12 per year) arise only when
        # an optimiser line search probes extreme parameter values; the matrix
        # exponential of such matrices can overflow internally and, with some
        # LAPACK builds, not return. Treat these evaluations as failed. (Valid
        # estimates reach about 1e7 per year when the surface heat capacity
        # tends to zero.)
        if np.abs(A).sum(axis=1).max() > MAX_RATE:
            raise FloatingPointError('system too stiff')

        # 1. Discretize A using matrix exponential
        Ad = scipy.linalg.expm(A)

        # 2. Discretize B
        if B.shape[1] > 0:  # Only if B is not empty
            D = np.block([
                [A, B],
                [np.zeros((B.shape[1], m)), np.zeros((B.shape[1], B.shape[1]))]
            ])
            expD = scipy.linalg.expm(D)

            Bd = expD[0:m, m:m+B.shape[1]]
        else:
            Bd = np.zeros((m, 0))

        # 3. Discretize V (process noise covariance)
        Vd = discretize_noise(A, V)

        # Ensure Vd is positive semidefinite
        Vd = ensure_psd(Vd)

        # 4. Calculate steady-state x0 and P0
        try:
            # Find steady-state x0
            x0 = np.zeros((m, 1))
            if B.shape[1] > 0:
                # If steady state exists, solve (I-Ad)x0 = Bd*u0
                u0 = np.zeros((B.shape[1], 1))
                I_minus_Ad = np.eye(m) - Ad
                if np.linalg.matrix_rank(I_minus_Ad) == m:
                    x0 = np.linalg.solve(I_minus_Ad, Bd @ u0)
                else:
                    # Use pseudo-inverse for singular case
                    x0 = np.linalg.lstsq(I_minus_Ad, Bd @ u0, rcond=None)[0]

            # Find steady-state P0
            try:
                # Estimate P0 using the Kronecker method.
                vecP0 = np.linalg.solve(np.eye(m*m) - np.kron(Ad, Ad), Vd.ravel(order='F'))
                P0 = vecP0.reshape(m, m, order='F')
            except:
                # Fallback: solution to discrete Lyapunov equation P = A*P*A' + Q
                P0 = scipy.linalg.solve_discrete_lyapunov(Ad, Vd)

            # Ensure P0 is positive semidefinite
            P0 = ensure_psd(P0)

        except Exception as e:
            # Fallback for numerical issues
            warnings.warn(f"Error computing steady-state: {e}. Using default values.")
            x0 = np.zeros((m, 1))
            P0 = 1e+12 * np.eye(m) # diffuse prior

        # 5. Measurement noise (typically very small)
        W = np.eye(C.shape[0]) * 1e-10
        w = np.zeros((C.shape[0], 1))

        return x0, P0, Ad, Bd, Vd, C, w, W

    def set_observation(self, Y):
        """
        Set the observation sequence.

        Parameters:
        Y (list): List of observation vectors
        """
        self.kf.Y = Y

    def set_control(self, U):
        """
        Set the control sequence.

        Parameters:
        U (list): List of control vectors
        """
        self.kf.U = U

    def generate_sample(self, n=None, seed=None):
        """
        Generate a sample path from the model.

        Parameters:
        n (int): Length of sample path. If None, uses control sequence length.
        seed (int): Random seed for reproducibility

        Returns:
        tuple: (X, Y, Disturbance, Noise) - States, observations, and noise components
        """
        if seed is not None:
            np.random.seed(seed=seed)

        # Use current parameters or defaults
        if self.parameters:
            parameters = self.parameters
        else:
            parameters = list(self.parameters_default.values())

        # Build and discretize matrices
        x0, P0, Ad, Bd, Vd, C, w, W = self.build_matrices(parameters)

        # Get control sequence
        U = self.kf.U
        if not U and n is not None:
            # Create zero controls of size n if none provided
            dim_u = Bd.shape[1] if Bd.shape[1] > 0 else 1
            U = [np.zeros((dim_u, 1)) for _ in range(n)]
        elif not U:
            raise ValueError("Either n or control sequence U must be provided")

        n = len(U)

        # Simulate trajectory
        X, Y, Disturbance, Noise = self.kf.simulate(x0, P0, Ad, Bd, Vd, C, w, W, U, add_noise=False)

        return X, Y, Disturbance, Noise

    def objfun(self, input_values, log_input=True, check_error=False):
        """
        Objective function to minimize (negative log-likelihood).

        Parameters:
        input_values (numpy.ndarray): Parameter values or log-parameter values
        log_input (bool): Whether input_values are log-transformed parameters

        Returns:
        float: Negative log-likelihood
        """
        val0 = 1e+10  # Default high value for invalid parameters
        error_rate = np.nan

        try:
            # Convert from log-space if necessary
            if log_input:
                parameters = np.exp(input_values)
            else:
                parameters = input_values

            # Validate parameters
            if np.any(~np.isfinite(parameters)) or np.any(parameters <= 0) or np.any(parameters > 1e+10):
                warnings.warn("Invalid parameter values")
                if check_error:
                    return val0, error_rate
                else:
                    return val0

            # Build and discretize matrices
            x0, P0, Ad, Bd, Vd, C, w, W = self.build_matrices(parameters)

            # Validate matrices
            matrices = [x0, P0, Ad, Bd, Vd, C, w, W]
            if any(np.any(~np.isfinite(m)) for m in matrices):
                if check_error:
                    return val0, error_rate
                else:
                    return val0

            # Ensure key matrices are psd (posirtive semi definite)
            P0 = ensure_psd(P0)
            Vd = ensure_psd(Vd)
            W = ensure_psd(W)

            # Compute log-likelihood
            #val = -self.kf.log_likelihood(x0, P0, Ad, Bd, Vd, C, w, W)
            log_L, error_count, total_steps = self.kf.log_likelihood(x0, P0, Ad, Bd, Vd, C, w, W)
            val = -log_L

            # Add penalty for high error rates
            error_rate = error_count / total_steps
            if error_rate > 0:
                # Increase penalty based on error rate
                val += 1e+10 * error_rate

            #if log_input:
            #    # L2 regularization in log space
            #    reg_strength = 1e-4
            #    reg_term = reg_strength * np.sum(input_values**2)
            #    val += + reg_term

            if not np.isfinite(val):
                val = val0

            if check_error:
                return val, error_rate
            else:
                return val

        except Exception as e:
            warnings.warn(f"Error in objfun: {e}")
            if check_error:
                return val0, error_rate
            else:
                return val0

    def differential_evolution_optimizer(self, log_parameters, bounds, max_iter=1000, disp=False):
        from scipy.optimize import differential_evolution

        # Run differential evolution
        res = differential_evolution(lambda x: self.objfun(x, log_input=True),
                                     bounds,
                                     x0=log_parameters,
                                     maxiter=max_iter,
                                     strategy='rand2bin', #'best1bin',
                                     disp=disp,
                                     updating='deferred',  # Changed from 'immediate'
                                     popsize=20,
                                     mutation=(0.5, 1.0), # differential weight factor
                                     recombination=0.9, # crossover probability
                                     tol=1e-6,
                                     )
        return res

    def dual_annealing_optimizer(self, log_parameters, bounds, max_iter=1000):
        from scipy.optimize import dual_annealing

        # Run dual annealing
        res = dual_annealing(lambda x: self.objfun(x, log_input=True),
                             bounds,
                             x0=log_parameters,
                             maxiter=max_iter,
                             initial_temp=500,  # Lower initial temperature relative to default 5230
                             restart_temp_ratio=2e-5,  # More conservative cooling
                             visit=2.0,  # More conservative visitation parameter
                             accept=-5.0,  # More conservative acceptance parameter
                             maxfun=10000,  # Limit function evaluations
                             no_local_search=True,  # Try disabling local search initially
                             )
        return res

    def basin_hopping_optimizer(self, log_parameters, bounds, max_iter=1000):
        from scipy.optimize import basinhopping

        # Set up options for the local minimizer (e.g., BFGS)
        minimizer_kwargs = {"method": "BFGS", "tol": self.optimization_tol}

        # Run basin hopping
        res = basinhopping(lambda x: self.objfun(x, log_input=True),
                              log_parameters,  # initial guess in log-space
                              minimizer_kwargs=minimizer_kwargs,
                              niter=max_iter)  # number of basin hops
        return res

    def particle_swarm_optimizer(self, log_parameters, bounds, max_iter=1000, n_particles=30):
        """
        Particle Swarm Optimization using PySwarms library.

        Parameters:
        log_parameters (numpy.ndarray): Initial parameters in log space
        bounds (tuple): (lower_bounds, upper_bounds) in log space
        max_iter (int): Maximum number of iterations
        n_particles (int): Number of particles in the swarm

        Returns:
        tuple: (best_params, best_fvalue, success, message, num_iter)
        """
        import pyswarms as ps
        from pyswarms.utils.functions import single_obj as fx

        # Define bounds for PySwarms format
        lower_bounds = np.array([bound[0] for bound in bounds])
        upper_bounds = np.array([bound[1] for bound in bounds])
        bounds = (lower_bounds, upper_bounds)

        # Define objective function wrapper for PySwarms
        def objective_function(x):
            return np.array([self.objfun(ind, log_input=True) for ind in x])

        # Define optimizer options
        options = {
            'c1': 1.5,           # Cognitive parameter
            'c2': 1.5,           # Social parameter
            'w': 0.7,            # Inertia parameter
            'k': 3,              # Number of neighbors (for local PSO variants)
            'p': 2,              # Minkowski p-norm (for local PSO variants)
        }

        # Initialize a global PSO optimizer
        init_pos = np.tile(log_parameters, (n_particles, 1))
        try:
            optimizer = ps.single.GlobalBestPSO(
                n_particles=n_particles,
                dimensions=len(log_parameters),
                options=options,
                bounds=bounds,
                init_pos=init_pos,
            )
        except Exception as e:
            print(e)

        # Run optimization
        best_cost, best_pos = optimizer.optimize(
            objective_function,
            iters=max_iter,
            verbose=True,
        )

        success = True  # PySwarms doesn't explicitly report success/failure
        message = "Completed"

        res = type('', (), {})()  # Create a dummy object
        res.x = best_pos
        res.fun = best_cost
        res.success = success
        res.message = message
        res.nit = np.nan

        return res

    def consistent_with_physical_constraints(self, parameters, verbose=False):

        val = True

        dct = {}
        keys = self.parameters_default.keys()
        for key, parameter in zip(keys, parameters):
            dct[key] = parameter

        if 'chi_s' in dct:
            if 'chi_o' in dct:
                if dct['chi_o'] <= dct['chi_s']:
                    val = False
                #if 'chi_d' in dct and dct['chi_d'] <= dct['chi_o']:
                #    val = False

        for key in keys:
            parameter = dct[key]
            lower_bound, upper_bound = self.parameters_bounds[key]
            if lower_bound is not None and lower_bound > parameter:
                val = False
            if upper_bound is not None and upper_bound < parameter:
                val = False

        if verbose and (val is False):
            for key in keys:
                print(f" - {key}: {dct[key]}")

        return val


    def estimate(self, initial_guess=None, seed=None, num_attempts=None, seek_global_minimum=False, initial_search=True):
        """
        Estimate model parameters using maximum likelihood.

        Parameters:
        initial_guess (list): Initial parameter values
        seed (int): Random seed for reproducibility
        num_attempts (int): Number of optimization attempts

        Returns:
        dict: Optimization results
        """
        # Set optimization parameters
        if num_attempts is None:
            num_attempts = self.optimization_attempts

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed=seed)

        # Use default initial guess if none provided
        if initial_guess is None:
            initial_guess = list(self.parameters_default.values())

        parameters = initial_guess
        num_parameters = len(parameters)

        # Initialize best solution tracking
        self.best_fvalue = float('inf')
        self.best_parameters = parameters.copy()
        self.optimization_results = {}

        bounds = []
        for key in self.parameters_default.keys():
            lb, ub = self.parameters_bounds[key]
            log_lb = None if lb == None else np.log(lb)
            log_ub = None if ub == None else np.log(ub)
            bounds.append((log_lb, log_ub))
        #bounds = [(np.log(self.parameters_bounds[key][0]), np.log(self.parameters_bounds[key][1])) for key in self.parameters_default.keys()]
        #bounds = [(None, None) for _ in parameters]

        #initial_search = True

        # set lower bound for f value
        fvalue1 = -500

        # Run multiple optimization attempts
        for attempt in range(num_attempts):
            print(f"\n### Attempt {attempt+1}/{num_attempts} ###\n")

            try:
                fvalue = self.objfun(np.log(self.best_parameters))
            except:
                fvalue = float('inf')
            print('fvalue:', fvalue)

            print('Perturb around current best with increasing radius...')
            # Perturb around current best with increasing radius
            parameters = self.best_parameters
            num_search = 500
            max_radius = 0.1
            for idx_perturb in range(num_search):
                radius = max_radius * idx_perturb/num_search
                log_parameters_candidate = np.log(parameters) + np.random.randn(num_parameters) * radius
                try:
                    fvalue0 = self.objfun(log_parameters_candidate)
                    parameters0 = np.exp(log_parameters_candidate)
                    if fvalue0 <= fvalue1:
                        # unreasonable value
                        continue
                    if fvalue0 < fvalue and self.consistent_with_physical_constraints(parameters0):
                        # update
                        fvalue = fvalue0
                        parameters = parameters0
                        print(fvalue)
                except Exception:
                    continue

            # further polish the starting point using Nelder-Mead
            # we do not expect this step to converge, only for polishing to give a better starting point for the optimization routine that follows
            method = 'Nelder-Mead'
            options = {'maxiter': 500, 'adaptive': True}
            print(f'Improving the initial values using {method}')
            res = scipy.optimize.minimize(
                fun=self.objfun,
                x0=np.log(parameters),
                method=method,
                tol=1e-8,
                bounds=[(np.log(self.default_parameters_bounds[key][0]), np.log(self.default_parameters_bounds[key][1])) for key in self.parameters_default.keys()],
                options=options,
            )
            print(f"{res.message}(fvalue={res.fun} in {res.nit} iterations)")
            parameters0, fvalue0 = np.exp(res.x), res.fun
            if fvalue0 <= fvalue1:
                # unreasonable value
                continue
            if fvalue0 < fvalue and self.consistent_with_physical_constraints(parameters0):
                # update
                fvalue = fvalue0
                parameters = parameters0
                print(fvalue)

            # display updated initial guess
            for idx_prm, key in enumerate(self.parameters_default.keys()):
                print(f" - {key}: {parameters[idx_prm]}")

            print(f"Initial guess fvalue: {fvalue}\n")

            # Keep the polished starting point as a candidate, so that the
            # estimate is never worse than where the search started (e.g. a
            # nested start from a smaller model) even if every method fails.
            if fvalue < self.best_fvalue and self.consistent_with_physical_constraints(parameters):
                self.optimization_results['start'] = {
                    'attempt': attempt,
                    'elapsed_time': 0.0,
                    'parameters': np.array(parameters, dtype=float),
                    'fvalue': fvalue,
                    'message': 'polished starting point',
                    'status': 'Start',
                    'std_errors': None,
                    'confidence_intvls': None,
                    'res': None,
                }
                self.best_method = 'start'
                self.best_parameters = np.array(parameters, dtype=float)
                self.best_fvalue = fvalue
                self.parameter_std_errors = None

            # Try various optimization methods
            self._run_optimization_methods(parameters, attempt, seek_global_minimum)

        # Print summary of results
        log_text = self._print_summary(initial_guess)

        return {
            'parameters': self.parameters,
            'fvalue': self.best_fvalue,
            'std_errors': self.parameter_std_errors,
            'best_method': self.best_method,
            'results': self.optimization_results,
            'log': log_text,
        }

    def _run_optimization_methods(self, parameters, attempt, seek_global_minimum=False):
        """
        Run optimization with different methods.

        Parameters:
        parameters (list): Initial parameter values
        attempt (int): Current optimization attempt
        """

        all_methods = self.optimization_methods[:]
        #bounds = [(np.log(self.parameters_bounds[key][0]), np.log(self.parameters_bounds[key][1])) for key in self.parameters_default.keys()]
        bounds = []
        for key in self.parameters_default.keys():
            lb, ub = self.parameters_bounds[key]
            log_lb = None if lb == None else np.log(lb)
            log_ub = None if ub == None else np.log(ub)
            bounds.append((log_lb, log_ub))

        # Add global optimization if this is an early attempt
        if seek_global_minimum and (attempt < 1):  # Use global methods in early attempts
#            if 'DE' not in all_methods:
#                all_methods.append('DE')
#            if 'DA' not in all_methods:
#                all_methods.append('DA')
#            if 'BH' not in all_methods:
#                all_methods.append('BH')
            # Particle swarm needs the optional package pyswarms; without it
            # the global step is skipped (as in all results in this repository).
            try:
                import pyswarms  # noqa: F401
                if 'PS' not in all_methods:
                    all_methods.append('PS')
            except ImportError:
                print('pyswarms not installed: particle-swarm step skipped')

        import scipy.optimize
        try:
            # Try to import pybobyqa if available
            import pybobyqa
            pybobyqa_available = True
        except ImportError:
            pybobyqa_available = False
            if 'BOBYQA' in all_methods:
                all_methods.remove('BOBYQA')

        for method_idx, method in enumerate(all_methods):
            print(f"Solving for MLE with {method}")
            success = False
            std_errs = None
            confidence_intvls = None

            try:

                if method in ['BOBYQA', 'DE', 'DA', 'PS', 'BH']:
                    unbounded = False
                    for bound in bounds:
                        lb, ub = bound
                        if (lb is None) or (ub is None):
                            unbounded = True
                    if unbounded:
                        bounds = [np.log(self.default_parameters_bounds[key]) for key in self.parameters_default.keys()]

                if method == 'DE':
                    # Run Differential Evolution
                    start_time = time.time()
                    res = self.differential_evolution_optimizer(
                        np.log(parameters), bounds,
                        max_iter=50, #self.optimization_maxiter,
                        disp=False,
                    )
                    elapsed_time = time.time() - start_time

                    # Process results
                    success = res.success
                    #success = True
                    fvalue = res.fun
                    message = res.message
                    num_iter = res.nit
                    res_parameters = np.exp(res.x)

                    # allow influence on the next method within the methods loop
                    #parameters = res_parameters

                    # Compute standard errors
                    if success:
                        std_errs = np.nan * np.ones(len(res.x))
                        confidence_intvls = 1.96 * std_errs

                elif method == 'DA':
                    # Run dual annealing
                    start_time = time.time()
                    res = self.dual_annealing_optimizer(
                        np.log(parameters), bounds,
                        max_iter=250, #self.optimization_maxiter,
                    )
                    elapsed_time = time.time() - start_time

                    # Process results
                    success = res.success
                    #success = True
                    fvalue = res.fun
                    message = res.message[0]
                    num_iter = res.nit
                    res_parameters = np.exp(res.x)

                    # allow influence on the next method within the methods loop
                    #parameters = res_parameters

                    # Compute standard errors
                    if success:
                        std_errs = np.nan * np.ones(len(res.x))
                        confidence_intvls = 1.96 * std_errs

                elif method == 'PS':
                    # Run Particle Swarm
                    start_time = time.time()
                    res = self.particle_swarm_optimizer(
                        np.log(parameters), bounds,
                        max_iter=250, #self.optimization_maxiter,
                    )
                    elapsed_time = time.time() - start_time

                    # Process results
                    success = res.success
                    #success = True
                    fvalue = res.fun
                    message = res.message
                    num_iter = res.nit
                    res_parameters = np.exp(res.x)

                    # Compute standard errors
                    if success:
                        std_errs = np.nan * np.ones(len(res.x))
                        confidence_intvls = 1.96 * std_errs

                elif method == 'BH':
                    # Run basin hopping
                    start_time = time.time()
                    res = self.basin_hopping_optimizer(
                        np.log(parameters), bounds, max_iter=50,
                    )
                    elapsed_time = time.time() - start_time

                    # Process results
                    success = res.success
                    #success = True
                    fvalue = res.fun
                    message = res.message[0]
                    num_iter = res.nit
                    res_parameters = np.exp(res.x)

                    # Compute standard errors
                    if success:
                        std_errs = np.nan * np.ones(len(res.x))
                        confidence_intvls = 1.96 * std_errs

                elif method == 'BOBYQA' and pybobyqa_available:
                    # Setup BOBYQA
                    lower = [bound[0] for bound in bounds]
                    upper = [bound[1] for bound in bounds]

                    # Run optimization
                    start_time = time.time()
                    res = pybobyqa.solve(
                        objfun=self.objfun,
                        x0=np.log(parameters),
                        maxfun=self.optimization_maxiter,
                        bounds=(lower, upper),
                        scaling_within_bounds=True,
                        rhoend=1e-8,
                        seek_global_minimum=False,
                        #seek_global_minimum=seek_global_minimum,
                        do_logging=False,
                    )
                    elapsed_time = time.time() - start_time

                    success = res.flag == res.EXIT_SUCCESS
                    message = res.msg
                    # Process results
                    if "maximum total number of unsuccessful restarts" in message:
                        # Consider unsuccessful if message indicates maximum restarts reached
                        success = False
                    fvalue = res.f
                    num_iter = res.nf

                    res_parameters = np.exp(res.x)

                    # Compute standard errors
                    if success:
                        std_errs = np.nan * np.ones(len(res.x))
                        confidence_intvls = 1.96 * std_errs

                else:
                    # Setup scipy.optimize methods
                    options = {'maxiter': self.optimization_maxiter}

                    if method == 'Nelder-Mead':
                        options['adaptive'] = True
                    if method == 'BFGS':
                        options['gtol'] = 1e-5  # gradient tolerance for convergence
                        options['eps'] = 1e-8  # step size
                        #options['ftol'] = 1e-4  # Function value tolerance
                        #options['maxls'] = 50     # More line search steps
                        #options['disp'] = True    # More output for debugging
                    if method == 'L-BFGS-B':
                        options['gtol'] = 1e-8  # gradient tolerance for convergence
                        options['eps'] = 1e-8  # step size
                        #options['ftol'] = 1e-4  # Function value tolerance
                        #options['maxls'] = 50     # More line search steps
                        #options['disp'] = True    # More output for debugging

                    # Run optimization
                    start_time = time.time()
                    res = scipy.optimize.minimize(
                        fun=self.objfun,
                        x0=np.log(parameters),
                        method=method,
                        bounds=bounds,
                        tol=self.optimization_tol,
                        options=options
                    )
                    elapsed_time = time.time() - start_time

                    # Process results
                    success = res.success
                    fvalue = res.fun
                    message = res.message
                    num_iter = res.nit
                    res_parameters = np.exp(res.x)

                    # Compute standard errors
                    if success:
                        std_errs = np.nan * np.ones(len(res.x))
                        if method == 'BFGS' and hasattr(res, 'hess_inv'):
                            hess_inv = res.hess_inv
                            jacobian = np.diag(res_parameters)
                            covariance = ensure_psd(jacobian @ hess_inv @ jacobian.T)
                            std_errs = np.sqrt(np.diag(covariance))
                        else:
                            pass
                            #try:
                            #    hessian = nd.Hessian(self.objfun)
                            #    hess_inv = scipy.linalg.inv(hessian(res.x))
                            #    jacobian = np.diag(res_parameters)
                            #    covariance = ensure_psd(jacobian @ hess_inv @ jacobian.T)
                            #    std_errs = np.sqrt(np.diag(covariance))
                            #except Exception as e:
                            #    print(f"Error computing standard errors: {e}")
                        confidence_intvls = 1.96 * std_errs

            except Exception as e:
                print(f"===> Error in {method}: {e}\n")
                continue

            # Report results
            status = 'Success' if success else 'Failure'
            objvalue, error_rate = self.objfun(np.log(res_parameters), log_input=True, check_error=True)
            print(f"- {status} in {elapsed_time:.3f} seconds: {message} (f(x)={fvalue}, error={error_rate}, {num_iter} iterations)")

            # if physical parameter constraints are satisfied
            if not self.consistent_with_physical_constraints(res_parameters, verbose=True):
                success = False
                print("Physical constraints violated")

            if success:
                objvalue, error_rate = self.objfun(np.log(res_parameters), log_input=True, check_error=True)
                print(f"- fvalue = {fvalue} (error_rate = {error_rate})")

                # Update best result for this method
                if (method not in self.optimization_results) or (fvalue < self.optimization_results[method]['fvalue']):
                    self.optimization_results[method] = {
                        'attempt': attempt,
                        'elapsed_time': elapsed_time,
                        'parameters': res_parameters.copy(),
                        'fvalue': fvalue,
                        'message': message,
                        'status': status,
                        'std_errors': std_errs,
                        'confidence_intvls': confidence_intvls,
                        'res': res,
                    }
                    print(f'===> Best estimate for {method} updated')
                    parameter_labels = list(self.parameters_default.keys())
                    for parameter, ce, parameter_label in zip(res_parameters, confidence_intvls, parameter_labels):
                        print(f" {parameter:.4f}, # +-{ce:.4f} ({parameter_label})")

                # Update global best
                if fvalue < self.best_fvalue:
                    print('===> Best estimate among all methods updated')
                    self.best_method = method
                    self.best_parameters = res_parameters.copy()
                    self.best_fvalue = fvalue
                    self.parameter_std_errors = std_errs

            if method_idx != len(all_methods) - 1:
                print()

    def _print_summary(self, initial_guess):
        """
        Print summary of optimization results.

        Parameters:
        initial_guess (list): Initial parameter values
        """
        print('\n=== Summary ===\n')
        print(f" Sample size n: {len(self.kf.Y)}")
        print()

        output = [] # for print and log

        # Print results for each method
        for method in self.optimization_results:
            result = self.optimization_results[method]
            fvalue = result['fvalue']
            status = result['status']
            attempt = result['attempt']
            elapsed_time = result['elapsed_time']
            message = result['message']
            parameters = result['parameters']
            confidence_intvls = result.get('confidence_intvls', None)
            objvalue, error_rate = self.objfun(np.log(parameters), log_input=True, check_error=True)

            if method == self.best_method:
                output.append(f"--- {method} (Best method)")
                self.parameters = list(parameters.copy())
            else:
                output.append(f"--- {method}")
            output.append(f" fvalue: {fvalue} (attempt {attempt+1}) error={error_rate}")
            output.append(f" status: {status} in {elapsed_time:.3f} seconds")
            output.append(f" message: {message}")
            output.append(f" estimated parameters (vs initial guess):")

            parameter_labels = list(self.parameters_default.keys())
            for i, (parameter_label, parameter, parameter0) in enumerate(zip(parameter_labels, parameters, initial_guess)):
                lb, ub = self.parameters_bounds[parameter_label]
                if lb is not None:
                    if ub is not None:
                        db = (ub-lb)/100
                        bounded = (parameter<=lb+db) or (parameter>=ub-db)
                    else:
                        bounded = (parameter<=lb*1.01)
                else:
                    if ub is not None:
                        bounded = (parameter>=ub*0.99)
                    else:
                        bounded = False
                star = '*' if bounded else ''
                if confidence_intvls is not None:
                    ce = confidence_intvls[i]
                    output.append(f"  {parameter:.4f} +-{ce:.4f} ({parameter0:.4f}) {parameter_label} with bounds = ({lb}, {ub}) {star}")
                else:
                    output.append(f"  {parameter:.4f} ({parameter0:.4f}) {parameter_label} with bounds = ({lb}, {ub})")
            output.append('')

        log_text = '\n'.join(output)
        print(log_text)

        return log_text
