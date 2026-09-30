import numpy as np
from collections import OrderedDict
from .base_model import Model

# constant
phi_co2 = 5.35
S = 1366 # solar constant (W m-2)
sigma = 5.67 * 1.0e-8 # Stefan-Boltzmann constant (kg s-3 K-4)
r = 6.371 * 1.0e+6 # Earth's radius (m)
Ts_bar = 288.00 # pre-industrial surface temperature (K)
alpha_bar = 0.30 # pre-industrial albedo (fraction)
beta_bar = 0.386 # pre-industrial greenhouse effect
C_s = 2.22 * 1.0e+8 # C_s: heat capacity of mixed layer (W m-2 K-1) when ds = 70
delta_t = 1/(86400 * 365.25) # delta_t: second in years (yr)
#print((((1-alpha_bar)/(1-beta_bar)) * (S/(4*sigma)))**(1/4)) # should give the value of Ts_bar
lamda_lu_base = (1 - beta_bar) * 4 * sigma * (Ts_bar)**3

# CMIP6 models whose outgoing shortwave radiation falls with warming: the
# shortwave feedback is estimated as a positive magnitude and enters the
# system with the opposite sign (net feedback lamda_lu + lamda_su).
SIGN_FLIP_MODELS = ('GISS-E2-2-G', 'GISS-E2-2-H')

class RsutFit(Model):
    """
    Model for shortwave upward flux (RSUT).

    Model structure:
    du/dt = -gamma_su * u - gamma_su * lamda_su * Ts + sigma_su * dW/dt

    where:
    - u is the RSUT anomaly
    - Ts is the surface air temperature anomaly
    - gamma_su is the radiative adjustment rate
    - lamda_su is the shortwave feedback parameter
    - sigma_su is the process noise standard deviation
    - dW/dt is white noise
    """

    def __init__(self):
        """Initialize RSUT model with default parameters."""
        super().__init__()

        # Initial parameter guesses
        self.parameters_default = OrderedDict([
            ('gamma_su', 1.0),     # Feedback strength (yr^-1)
            ('lamda_su', 1.0),     # Shortwave feedback parameter (W/m^2/K)
            ('sigma_su', 0.5),     # Process noise std (W/m^2/yr^0.5)
        ])

    def build_matrices(self, parameters):
        """
        Build continuous-time system matrices from model parameters.

        Parameters:
        parameters (list): [gamma_su, lamda_su, sigma_su]

        Returns:
        tuple: Discrete-time system matrices
        """
        gamma_su, lamda_su, sigma_su = parameters

        # State matrix: A = -gamma_su
        A = np.array([[-gamma_su]])
        m = A.shape[0]  # State dimension

        # Input matrix: B = -lamda_su * gamma_su
        B = np.zeros((m, 1))
        B[0][0] = -lamda_su * gamma_su  # For surface temperature

        # Observation matrix: C = 1 (directly observe RSUT)
        C = np.array([[1.0]])

        # Process noise covariance: V = sigma_su^2
        V = np.zeros((m, m))
        V[0][0] = sigma_su**2

        return self.discretize_matrices(A, B, C, V)


class RlutFit(Model):
    """
    Model for longwave upward flux (RLUT).

    Model structure:
    dv/dt = -gamma_lu * v + gamma_lu * lamda_lu * Ts - gamma_lu * F + sigma_lu * dW/dt

    where:
    - v is the RLUT anomaly
    - Ts is the surface air temperature anomaly
    - F is the CO2 forcing
    - gamma_lu is the radiative adjustment rate
    - lamda_lu is the longwave feedback parameter
    - sigma_lu is the process noise standard deviation
    - dW/dt is white noise
    """

    def __init__(self):
        """Initialize RLUT model with default parameters."""
        super().__init__()

        self.phi_co2 = phi_co2

        # Initial parameter guesses
        self.parameters_default = OrderedDict([
            ('gamma_lu', 1.0),     # Feedback strength (yr^-1)
            ('lamda_lu', 1.0),     # Longwave feedback parameter (W/m^2/K)
            ('sigma_lu', 0.5),     # Process noise std (W/m^2/yr^0.5)
        ])

    def build_matrices(self, parameters):
        """
        Build continuous-time system matrices from model parameters.

        Parameters:
        parameters (list): [gamma_lu, lamda_lu, sigma_lu]

        Returns:
        tuple: Discrete-time system matrices
        """
        gamma_lu, lamda_lu, sigma_lu = parameters

        # State matrix: A = -gamma_lu
        A = np.array([[-gamma_lu]])
        m = A.shape[0]  # State dimension

        # Input matrix: B has two inputs (Ts and CO2 forcing)
        B = np.zeros((m, 2))
        B[0][0] = gamma_lu * lamda_lu  # For surface temperature
        B[0][1] = -gamma_lu            # For CO2 forcing

        # Observation matrix: C = 1 (directly observe RLUT)
        C = np.array([[1.0]])

        # Process noise covariance: V = sigma_lu^2
        V = np.zeros((m, m))
        V[0][0] = sigma_lu**2

        return self.discretize_matrices(A, B, C, V)


class OneLayerNoFeedbackModel(Model):
    """
    One-layer energy balance model without feedback for climate.
    """

    def __init__(self, model_id=None):
        """Initialize one-layer model without additional feedback."""
        super().__init__()

        self.phi_co2 = phi_co2
        self.model_id = model_id

        # Initial parameter guesses
        self.parameters_default = OrderedDict([
            ('gamma_su', 6),
            ('gamma_lu', 1),
            ('sigma_su', 0.787817618782046),
            ('sigma_lu', 1.1734558525459313),
            ('sigma_s', 0.8952604668204938),
            ('chi_s', 6.965017170607446),
        ])

    def build_matrices(self, parameters):
        """
        Build continuous-time system matrices from model parameters.

        Parameters:
        parameters (list): Model parameters in order of parameters_default

        Returns:
        tuple: Discrete-time system matrices
        """
        # Extract parameters
        gamma_su, gamma_lu, sigma_su, sigma_lu, sigma_s, chi_s  = parameters

        # State matrix A
        A = np.array([
            [-gamma_su, 0, 0], # du/dt
            [0, -gamma_lu, gamma_lu*lamda_lu_base], # dv/dt
            [-1/chi_s, -1/chi_s, 0], # dTs/dt
        ])

        # Input matrix B (for CO2 forcing)
        B = np.zeros((3, 1))
        B[1][0] = -gamma_lu  # CO2 forcing affects RLUT

        # Observation matrix C (observe u, v, Ts)
        C = np.array([
            [1, 0, 0],  # Rsut
            [0, 1, 0],  # Rlut
            [0, 0, 1]   # Ts
        ])

        # Process noise covariance V
        V = np.zeros((3, 3))
        V[0][0] = sigma_su**2            # RSUT noise
        V[1][1] = sigma_lu**2            # RLUT noise
        V[2][2] = (sigma_s/chi_s)**2     # Surface temp noise

        return self.discretize_matrices(A, B, C, V)


class OneLayerModel(Model):
    """
    One-layer energy balance model for climate.
    """

    def __init__(self, model_id=None):
        """Initialize one-layer model with default parameters."""
        super().__init__()

        self.phi_co2 = phi_co2
        self.model_id = model_id

        # Initial parameter guesses
        self.parameters_default = OrderedDict([
            ('lamda_su', 1),
            ('lamda_lu', 2),
            ('gamma_su', 6),
            ('gamma_lu', 1),
            ('sigma_su', 0.787817618782046),
            ('sigma_lu', 1.1734558525459313),
            ('sigma_s', 0.8952604668204938),
            ('chi_s', 6.965017170607446),
        ])

    def build_matrices(self, parameters):
        """
        Build continuous-time system matrices from model parameters.

        Parameters:
        parameters (list): Model parameters in order of parameters_default

        Returns:
        tuple: Discrete-time system matrices
        """
        # Extract parameters
        lamda_su, lamda_lu, gamma_su, gamma_lu, sigma_su, sigma_lu, sigma_s, chi_s  = parameters

        if self.model_id in SIGN_FLIP_MODELS:
            lamda_su = -lamda_su

        # State matrix A
        A = np.array([
            [-gamma_su, 0, -gamma_su*lamda_su], # du/dt
            [0, -gamma_lu, gamma_lu*lamda_lu], # dv/dt
            [-1/chi_s, -1/chi_s, 0], # dTs/dt
        ])

        # Input matrix B (for CO2 forcing)
        B = np.zeros((3, 1))
        B[1][0] = -gamma_lu  # CO2 forcing affects RLUT

        # Observation matrix C (observe u, v, Ts)
        C = np.array([
            [1, 0, 0],  # Rsut
            [0, 1, 0],  # Rlut
            [0, 0, 1]   # Ts
        ])

        # Process noise covariance V
        V = np.zeros((3, 3))
        V[0][0] = sigma_su**2            # RSUT noise
        V[1][1] = sigma_lu**2            # RLUT noise
        V[2][2] = (sigma_s/chi_s)**2     # Surface temp noise

        return self.discretize_matrices(A, B, C, V)


class TwoLayerModel(Model):
    """
    Two-layer energy balance model for climate.

    State variables:
    - u: RSUT anomaly
    - v: RLUT anomaly
    - Ts: Surface temperature anomaly
    - To: Upper-ocean temperature anomaly

    Model structure:
    du/dt = -gamma_su * u - gamma_su * lamda_su * Ts + sigma_su * dW_u/dt
    dv/dt = -gamma_lu * v + gamma_lu * lamda_lu * Ts - gamma_lu * F + sigma_lu * dW_v/dt
    dTs/dt = (-u - v)/chi_s - kappa_o/chi_s * (Ts - To) + sigma_s/chi_s * dW_s/dt
    dTo/dt = kappa_o/chi_o * (Ts - To)

    where:
    - F is the CO2 forcing
    - gamma_su and gamma_lu are separate radiative adjustment rates
    - lamda_su, lamda_lu are feedback parameters
    - chi_s, chi_o are heat capacities
    - kappa_o is heat exchange coefficient
    - sigma_* are noise standard deviations
    """

    def __init__(self, model_id=None):
        """Initialize two-layer model with default parameters."""
        super().__init__()

        self.phi_co2 = phi_co2
        self.model_id = model_id

        # Initial parameter guesses
        self.parameters_default = OrderedDict([
            ('lamda_su', 0.8630161821275864),
            ('lamda_lu', 1.9686805503284466),
            ('gamma_su', 6),
            ('gamma_lu', 1),
            ('sigma_su', 0.787817618782046),
            ('sigma_lu', 1.1734558525459313),
            ('sigma_s', 0.8952604668204938),
            ('chi_s', 6.965017170607446),
            ('chi_o', 75.29990423424559),
            ('kappa_o', 0.7670063285095702),
        ])

    def build_matrices(self, parameters):
        """
        Build continuous-time system matrices from model parameters.

        Parameters:
        parameters (list): Model parameters in order of parameters_default

        Returns:
        tuple: Discrete-time system matrices
        """
        # Extract parameters
        lamda_su, lamda_lu, gamma_su, gamma_lu, sigma_su, sigma_lu, sigma_s, \
        chi_s, chi_o, kappa_o  = parameters

        if self.model_id in SIGN_FLIP_MODELS:
            lamda_su = -lamda_su

        # State matrix A
        A = np.array([
            # u        v        Ts                To
            [-gamma_su, 0,      -gamma_su*lamda_su, 0                ], # du/dt
            [0,        -gamma_lu, gamma_lu*lamda_lu, 0                ], # dv/dt
            [-1/chi_s, -1/chi_s, -kappa_o/chi_s,    kappa_o/chi_s    ], # dTs/dt
            [0,        0,        kappa_o/chi_o,     -kappa_o/chi_o   ]  # dTo/dt
        ])

        # Input matrix B (for CO2 forcing)
        B = np.zeros((4, 1))
        B[1][0] = -gamma_lu  # CO2 forcing affects RLUT

        # Observation matrix C (observe u, v, Ts)
        C = np.array([
            [1, 0, 0, 0],  # RSUT
            [0, 1, 0, 0],  # RLUT
            [0, 0, 1, 0]   # Ts
        ])

        # Process noise covariance V
        V = np.zeros((4, 4))
        V[0][0] = sigma_su**2            # RSUT noise
        V[1][1] = sigma_lu**2            # RLUT noise
        V[2][2] = (sigma_s/chi_s)**2     # Surface temp noise

        return self.discretize_matrices(A, B, C, V)


class ThreeLayerModel(Model):
    """
    Three-layer energy balance model for climate.

    State variables:
    - u: RSUT anomaly
    - v: RLUT anomaly
    - Ts: Surface temperature anomaly
    - To: Upper ocean temperature anomaly
    - Td: Deep ocean temperature anomaly

    Model structure extends the two-layer model with an additional deep ocean layer.
    """

    def __init__(self, model_id=None):
        """Initialize three-layer model with default parameters."""
        super().__init__()

        self.phi_co2 = phi_co2
        self.model_id = model_id

        # Initial parameter guesses
        self.parameters_default = OrderedDict([
            ('lamda_su', 0.8630161821275864),
            ('lamda_lu', 1.9686805503284466),
            ('gamma_su', 6),
            ('gamma_lu', 1),
            ('sigma_su', 0.787817618782046),
            ('sigma_lu', 1.1734558525459313),
            ('sigma_s', 0.6913121281438991),
            ('chi_s', 4.889552636694985),
            ('chi_o', 21.35482590040838),
            ('chi_d', 138.92043312675452),
            ('kappa_o', 1.6627021160021545),
            ('kappa_d', 0.8530824983287542),
        ])

    def build_matrices(self, parameters):
        """
        Build continuous-time system matrices from model parameters.

        Parameters:
        parameters (list): Model parameters in order of parameters_default

        Returns:
        tuple: Discrete-time system matrices
        """
        # Extract parameters
        lamda_su, lamda_lu, gamma_su, gamma_lu, sigma_su, sigma_lu, sigma_s, \
        chi_s, chi_o, chi_d, kappa_o, kappa_d = parameters

        if self.model_id in SIGN_FLIP_MODELS:
            lamda_su = -lamda_su

        # State matrix A
        A = np.array([
            # u        v        Ts                To                        Td
            [-gamma_su, 0,      -gamma_su*lamda_su, 0,                     0                 ], # du/dt
            [0,        -gamma_lu, gamma_lu*lamda_lu, 0,                     0                 ], # dv/dt
            [-1/chi_s, -1/chi_s, -kappa_o/chi_s,    kappa_o/chi_s,         0                 ], # dTs/dt
            [0,        0,        kappa_o/chi_o,     -kappa_o/chi_o-kappa_d/chi_o, kappa_d/chi_o     ], # dTo/dt
            [0,        0,        0,                 kappa_d/chi_d,          -kappa_d/chi_d    ]  # dTd/dt
        ])

        # Input matrix B (for CO2 forcing)
        B = np.zeros((5, 1))
        B[1][0] = -gamma_lu  # CO2 forcing affects RLUT

        # Observation matrix C (observe u, v, Ts)
        C = np.array([
            [1, 0, 0, 0, 0],  # RSUT
            [0, 1, 0, 0, 0],  # RLUT
            [0, 0, 1, 0, 0]   # Ts
        ])

        # Process noise covariance V
        V = np.zeros((5, 5))
        V[0][0] = sigma_su**2            # RSUT noise
        V[1][1] = sigma_lu**2            # RLUT noise
        V[2][2] = (sigma_s/chi_s)**2     # Surface temp noise

        return self.discretize_matrices(A, B, C, V)


class FourLayerModel(Model):
    """
    Four-layer energy balance model for climate.
    Model structure extends the three-layer model with a bottom ocean layer.
    """

    def __init__(self, model_id=None):
        """Initialize four-layer model with default parameters."""
        super().__init__()

        self.phi_co2 = phi_co2
        self.model_id = model_id

        # Initial parameter guesses
        self.parameters_default = OrderedDict([
            ('lamda_su', 0.8630161821275864),
            ('lamda_lu', 1.9686805503284466),
            ('gamma_su', 6),
            ('gamma_lu', 1),
            ('sigma_su', 0.787817618782046),
            ('sigma_lu', 1.1734558525459313),
            ('sigma_s', 0.6913121281438991),
            ('chi_s', 4.889552636694985),
            ('chi_o', 21.35482590040838),
            ('chi_d', 138.92043312675452),
            ('chi_b', 138.92043312675452),
            ('kappa_o', 1.6627021160021545),
            ('kappa_d', 0.8530824983287542),
            ('kappa_b', 0.8530824983287542),
        ])

    def build_matrices(self, parameters):
        """
        Build continuous-time system matrices from model parameters.

        Parameters:
        parameters (list): Model parameters in order of parameters_default

        Returns:
        tuple: Discrete-time system matrices
        """
        # Extract parameters
        lamda_su, lamda_lu, gamma_su, gamma_lu, sigma_su, sigma_lu, sigma_s, \
        chi_s, chi_o, chi_d, chi_b, kappa_o, kappa_d, kappa_b = parameters

        if self.model_id in SIGN_FLIP_MODELS:
            lamda_su = -lamda_su

        # State matrix A
        A = np.array([
            # u        v        Ts                To                        Td                Tk
            [-gamma_su, 0,      -gamma_su*lamda_su, 0,                     0                 ,0], # du/dt
            [0,        -gamma_lu, gamma_lu*lamda_lu, 0,                     0                 ,0], # dv/dt
            [-1/chi_s, -1/chi_s, -kappa_o/chi_s,    kappa_o/chi_s,         0                 ,0], # dTs/dt
            [0,        0,        kappa_o/chi_o,     -kappa_o/chi_o - kappa_d/chi_o, kappa_d/chi_o,     0], # dTo/dt
            [0,        0,        0,                 kappa_d/chi_d,  -kappa_d/chi_d - kappa_b/chi_d, kappa_b/chi_d],  # dTd/dt
            [0,        0,        0,                 0,  kappa_b/chi_b, -kappa_b/chi_b],  # dTk/dt
        ])
        m = A.shape[0]  # State dimension

        # Input matrix B (for CO2 forcing)
        B = np.zeros((m, 1))
        B[1][0] = -gamma_lu  # CO2 forcing affects RLUT

        # Observation matrix C (observe u, v, Ts)
        C = np.zeros((3, m))
        C[0][0] = 1
        C[1][1] = 1
        C[2][2] = 1

        # Process noise covariance V
        V = np.zeros((m, m))
        V[0][0] = sigma_su**2            # RSUT noise
        V[1][1] = sigma_lu**2            # RLUT noise
        V[2][2] = (sigma_s/chi_s)**2     # Surface temp noise

        return self.discretize_matrices(A, B, C, V)


def estimate_ar1_parameters_from_ols(observations, y_variable, x_variable):
    """
    Estimate physical parameters from AR1 regression coefficients.

    Parameters:
    observations (dict): Dictionary of time series
    y_variable (str): Dependent variable name ('rsut' or 'rlut')
    x_variable (str): Independent variable name ('tas')

    Returns:
    dict: Estimated physical parameters
    """
    from .ols_tools import ols_estimate_ar1_params

    # Extract data
    y, x = observations[y_variable], observations[x_variable]

    # Estimate AR1 model
    result = ols_estimate_ar1_params(y, x)

    # Extract coefficients
    beta_0 = result['beta_0']
    beta_1 = result['beta_1']
    beta_2 = result['beta_2']
    sigma = result['sigma']

    # Convert to physical parameters
    if y_variable == 'rsut':
        # For RSUT model:
        # du/dt = -gamma_su * u - gamma_su * lamda_su * Ts + sigma_su * dW/dt
        # Discretized: u_t = beta_0 + beta_1 * u_{t-1} + beta_2 * Ts_t + noise
        gamma_su = -np.log(beta_1) if beta_1 > 0 else -np.log(0.01)
        lamda_su = -beta_2 / (1 - beta_1)
        sigma_su = sigma * np.sqrt((2 * gamma_su) / (1 - np.exp(-2 * gamma_su)))

        return {
            'gamma_su': gamma_su,
            'lamda_su': lamda_su,
            'sigma_su': sigma_su,
        }

    elif y_variable == 'rlut':
        # For RLUT model:
        # dv/dt = -gamma_lu * v + gamma_lu * lamda_lu * Ts - gamma_lu * F + sigma_lu * dW/dt
        # Discretized: v_t = beta_0 + beta_1 * v_{t-1} + beta_2 * Ts_t + noise
        gamma_lu = -np.log(beta_1) if beta_1 > 0 else -np.log(0.01)
        lamda_lu = beta_2 / (1 - beta_1)
        sigma_lu = sigma * np.sqrt((2 * gamma_lu) / (1 - np.exp(-2 * gamma_lu)))

        return {
            'gamma_lu': gamma_lu,
            'lamda_lu': lamda_lu,
            'sigma_lu': sigma_lu,
        }

    else:
        raise ValueError(f"Unsupported variable: {y_variable}")


def estimate_equilibrium_climate_params(observations, estimated_parameters, phi_co2=phi_co2):
    """
    Estimate equilibrium climate parameters from observations and estimated radiative parameters
    based on Geoffroy's method.

    Parameters:
    observations (dict): Dictionary of time series
    estimated_parameters (dict): Dictionary of estimated radiative parameters
    phi_co2 (float): CO2 forcing parameter (default: 5.35 W/m²)

    Returns:
    dict: Estimated equilibrium climate parameters
    """

    # Calculate net feedback parameter
    lamda_su = estimated_parameters['lamda_su']
    lamda_lu = estimated_parameters['lamda_lu']
    lamda_nd = lamda_lu - lamda_su

    # Equilibrium temperature for quadrupled CO2
    Tbar = phi_co2 * np.log(4) / lamda_nd

    # Estimate time constants using temperature response
    t_start = 30  # Starting year for slow component estimation
    t_end = 10    # End year for fast component estimation

    tas = observations['tas']
    years = np.arange(len(tas))

    # Estimate slow time constant (tau_2) and its weight (theta_2)
    y = 1 - tas[t_start:] / Tbar
    x = years[t_start:]

    ## remove negative y values
    x = x[y > 0]
    y = y[y > 0]

    from .ols_tools import OLS
    model = OLS(np.log(y), x, add_constant=True)
    tau_2 = -1 / model.params[1]
    theta_2 = np.exp(model.params[0])

    # Estimate fast time constant (tau_1)
    tau_1_estimates = []
    for idx, Ts in enumerate(tas[:t_end]):
        t = idx + 1
        term1 = np.log(1 - theta_2)
        term2 = np.log(1 - Ts/Tbar - theta_2 * np.exp(-t/tau_2))
        if not np.isnan(term1 - term2) and not np.isinf(term1 - term2):
            tau_1_est = t / (term1 - term2)
            if tau_1_est > 0:  # Only keep positive estimates
                tau_1_estimates.append(tau_1_est)

    tau_1 = np.median(tau_1_estimates) if tau_1_estimates else 1.0

    # Calculate heat capacity and coupling parameters
    c = (1 - theta_2) / tau_1 + theta_2 / tau_2
    b = 1 / (c * tau_1 * tau_2)
    a = 1 / tau_1 + 1 / tau_2 - b - c

    # Convert to physical parameters
    chi_s = (1 / c) * lamda_nd
    kappa_o = a * chi_s
    chi_o = kappa_o / b

    return {
        'chi_s': chi_s,
        'chi_o': chi_o,
        'kappa_o': kappa_o
    }
