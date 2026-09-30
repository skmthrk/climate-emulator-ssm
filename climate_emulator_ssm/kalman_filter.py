import numpy as np
import scipy.linalg
from scipy.stats import multivariate_normal
import warnings

# Suppress common warnings that may occur during matrix operations
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def ensure_psd(matrix, epsilon=1e-10):
    """
    Ensure a matrix is positive semi-definite while preserving its structure.

    Parameters:
    matrix (numpy.ndarray): Input matrix
    epsilon (float): Minimum eigenvalue threshold

    Returns:
    numpy.ndarray: Positive semi-definite matrix
    """
    # LAPACK routines can hang on non-finite input (observed with np.linalg.svd
    # on matrices containing inf); let the caller treat the evaluation as failed.
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError('non-finite matrix')

    # Force symmetry first
    matrix = (matrix + matrix.T) / 2.0

    # Compute eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(matrix)

        # Check if already PSD within tolerance
        if np.min(eigvals) >= epsilon:
            return matrix

        # Adjust eigenvalues while preserving relative scales
        min_eig = np.min(eigvals)
        scale_factor = 1.0

        # If we have very large eigenvalues, scale the adjustment
        if np.max(np.abs(eigvals)) > 1e4 * epsilon:
            scale_factor = np.max(np.abs(eigvals)) / 1e4

        # Apply the adjustment and floor small eigenvalues
        eigvals = np.maximum(eigvals, epsilon)

        # Reconstruct matrix
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    except np.linalg.LinAlgError as e1:
        # Fallback method for computational issues

        warnings.warn(f"Standard PSD enforcement failed: {e1}, trying SVD")
        try:
            # Try SVD approach
            U, s, Vh = np.linalg.svd(matrix)
            s[s < epsilon] = epsilon
            return U @ np.diag(s) @ Vh
        except Exception as e2:
            warnings.warn(f"SVD PSD enforcement failed: {e2}, using diagonal regularization")
            # Last resort: add small value to diagonal
            return matrix + epsilon * np.eye(matrix.shape[0])

def spd_inverse(matrix):
    """Inverse of a symmetric positive definite matrix via Cholesky."""
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError('non-finite matrix')
    factor = scipy.linalg.cho_factor((matrix + matrix.T) / 2, lower=True)
    return scipy.linalg.cho_solve(factor, np.eye(matrix.shape[0]))


class KalmanFilter:
    """
    An implementation of the Kalman filter for linear state-space models.

    The model is defined as:
    State equation:     x_{t+1} = A x_t + B u_t + nu_t,  where nu_t ~ N(0, V)
    Measurement equation: y_t = C x_t + w + omega_t,  where omega_t ~ N(0, W)

    Initial state: x_0 ~ N(x0, P0)
    """

    def __init__(self):
        """Initialize the Kalman filter with default parameters."""

        # Set diffuse prior as default initial state
        self.x0 = np.zeros((1, 1))
        self.P0 = 1e+12 * np.eye(1)

        # Placeholder for observations and controls
        self.Y = []  # List of observation vectors
        self.U = []  # List of control vectors

        # Set numerical methods for matrix operations
        # Inverse of the (symmetric positive definite) forecast covariance.
        # A pseudo-inverse would silently drop nearly singular directions.
        self.inv = spd_inverse
        self.det = scipy.linalg.det

        # For numerical stability
        self._epsilon = 1e-10

        # For logging/debugging
        self.log_enabled = False
        self.log = []

        # Initialize model parameters
        self._reset_model(self.x0, self.P0)

    def enable_logging(self, enabled=True):
        """Enable or disable logging of filter state."""
        self.log_enabled = enabled
        if enabled:
            self.log = []

    def _reset_model(self, x0, P0):
        """
        Reset the model with new initial state parameters.

        Parameters:
        x0 (numpy.ndarray): Initial state mean
        P0 (numpy.ndarray): Initial state covariance
        """
        # Initial state distribution x ~ N(x0, P0)
        self.x = x0
        self.P = P0

        # Get dimensions
        dim_x = x0.shape[0]
        dim_y = self.Y[0].shape[0] if self.Y else 1
        dim_u = self.U[0].shape[0] if self.U else 1

        # State transition: x' = Ax + v + nu where v = Bu and nu ~ N(0,V)
        self.A = np.eye(dim_x)
        self.B = np.zeros((dim_x, dim_u))
        self.u = np.zeros((dim_u, 1))
        self.v = self.B @ self.u
        self.V = np.eye(dim_x)

        # Measurement: y' = Cx' + w + omega, omega ~ N(0,W)
        self.C = np.zeros((dim_y, dim_x))
        self.w = np.zeros((dim_y, 1))
        self.W = np.eye(dim_y)

        # Prior distribution: x'|y ~ N(x_prior, P_prior)
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # Forecast: y'|y ~ N(y, Q)
        self.y = np.zeros((dim_y, 1))
        self.Q = np.eye(dim_y)
        self.Qinv = self.inv(self.Q)

        # Kalman gain and prediction error
        self.K = np.zeros((dim_x, dim_y))
        self.q = np.zeros((dim_y, 1))

        # Clear log if enabled
        if self.log_enabled:
            self.log = []

    def predict(self, A, B, u, V, C, w, W):
        """
        Calculate the prior distribution x'|y and forecast y'|y.

        Parameters:
        A (numpy.ndarray): State transition matrix
        B (numpy.ndarray): Control input matrix
        u (numpy.ndarray): Control input vector
        V (numpy.ndarray): Process noise covariance
        C (numpy.ndarray): Observation matrix
        w (numpy.ndarray): Observation offset
        W (numpy.ndarray): Observation noise covariance
        """
        # Store current parameters
        self.A, self.B, self.u, self.V = A, B, u, V
        self.C, self.w, self.W = C, w, W

        # 1. Compute state prior: x_{t}|Y_{t-1} ~ N(x_{t|t-1}, P_{t|t-1})

        # Prior mean: x_{t|t-1} = A_{t}x_{t-1|t-1} + B_{t}u_{t}
        self.x_prior = A @ self.x + B @ u

        # Prior covariance: P_{t|t-1} = A_{t}P_{t-1|t-1}A_{t}^T + V_{t}
        self.P_prior = A @ self.P @ A.T + V
        self.P_prior = ensure_psd(self.P_prior, self._epsilon)

        # 2. Compute forecast: y_{t}|Y_{t-1} ~ N(y_{t|t-1}, Q_{t|t-1})

        # Forecast mean: y_{t|t-1} = C_{t}x_{t|t-1} + w_{t}
        self.y = C @ self.x_prior + w

        # Forecast covariance: Q_{t|t-1} = C_{t}P_{t|t-1}C_{t}^T + W_{t}
        self.Q = C @ self.P_prior @ C.T + W
        self.Q = ensure_psd(self.Q, self._epsilon)

        # Compute inverse for efficiency
        self.Qinv = self.inv(self.Q)

        # 3. Compute Kalman gain: K_{t} = P_{t|t-1}C_{t}^T Q_{t|t-1}^{-1}
        self.K = self.P_prior @ C.T @ self.Qinv

        # Log state if enabled
        if self.log_enabled:
            self.log.append({
                'step': 'predict',
                'x_prior': self.x_prior.copy(),
                'P_prior': self.P_prior.copy(),
                'y_forecast': self.y.copy(),
                'Q': self.Q.copy(),
                'K': self.K.copy()
            })

    def update(self, y):
        """
        Calculate the posterior distribution x'|y' after observing y'.

        Parameters:
        y (numpy.ndarray): Observed measurement vector
        """
        # Forecast error (innovation)
        self.q = y - self.y

        # Posterior mean: x_{t|t} = x_{t|t-1} + K_{t}q_{t}
        self.x = self.x_prior + self.K @ self.q

        # Posterior covariance: P_{t|t} = P_{t|t-1} - K_{t}Q_{t|t-1}K_{t}^T
        # Direct calculation
        #self.P = self.P_prior - self.K @ self.Q @ self.K.T
        ## Or alternatively, use Joseph form for numerical stability
        I = np.eye(self.P_prior.shape[0])
        self.P = (I - self.K @ self.C) @ self.P_prior @ (I - self.K @ self.C).T + self.K @ self.W @ self.K.T

        # Ensure P remains positive definite
        self.P = ensure_psd(self.P, self._epsilon)

        # Log state if enabled
        if self.log_enabled:
            self.log.append({
                'step': 'update',
                'y_observed': y.copy(),
                'q': self.q.copy(),
                'x_posterior': self.x.copy(),
                'P_posterior': self.P.copy()
            })

    def Pt(self):
        """
        Calculate the probability density Prob(y_{t}|Y_{t-1})

        Returns:
        float: Probability density value
        """
        # Compute multivariate normal probability density
        try:
            # Compute determinant and check if valid
            det_Q = abs(self.det(self.Q))
            if det_Q < self._epsilon:
                det_Q = self._epsilon  # Prevent division by zero

            # Calculate the quadratic form: q^T Q^{-1} q
            quadratic = (self.q.T @ self.Qinv @ self.q)[0, 0]

            # Calculate the density
            dim = self.q.shape[0]
            val = np.exp(-0.5 * quadratic) / ((2*np.pi)**(dim/2) * np.sqrt(det_Q))
            return float(val)
        except Exception as e:
            warnings.warn(f"Error in probability density computation: {e}")
            return self._epsilon  # Return small positive number on error

    def log_Pt(self):
        """
        Calculate ln(Prob(y_{t}|Y_{t-1})) - the log likelihood contribution

        The Gaussian log-density -0.5 (k ln 2 pi + ln det Q + q' Q^{-1} q) is
        evaluated with a Cholesky factorisation of the forecast covariance Q.
        (An earlier version fell back to a pseudo-inverse density when the
        direct density underflowed, which ignores the residuals along nearly
        singular directions of Q and can give arbitrarily high likelihoods.)

        Returns:
        tuple: (log probability value, error flag)
        """
        val0 = -50  # Default value for invalid cases
        error_flag = True

        # Check for invalid states
        if np.any(~np.isfinite(self.q)) or np.any(~np.isfinite(self.Q)):
            return val0, error_flag

        try:
            L = np.linalg.cholesky((self.Q + self.Q.T) / 2)
        except np.linalg.LinAlgError:
            return val0, error_flag
        z = scipy.linalg.solve_triangular(L, self.q.flatten(), lower=True)
        val = -0.5 * (len(z) * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + z @ z)
        if not np.isfinite(val):
            return val0, error_flag
        return float(val), False

    def log_likelihood(self, x0, P0, A, B, V, C, w, W):
        """
        Calculate the log-likelihood for the entire sequence of observations.

        Parameters:
        x0 (numpy.ndarray): Initial state mean
        P0 (numpy.ndarray): Initial state covariance
        A (numpy.ndarray): State transition matrix
        B (numpy.ndarray): Control input matrix
        V (numpy.ndarray): Process noise covariance
        C (numpy.ndarray): Observation matrix
        w (numpy.ndarray): Observation offset
        W (numpy.ndarray): Observation noise covariance

        Returns:
        tuple: (log-likelihood value, number of errors)
        """
        # Reset model with initial state
        self._reset_model(x0, P0)

        # Get observation and control sequences
        Y = self.Y  # List of observation vectors
        U = self.U  # List of control vectors

        # Accumulate log-likelihood
        log_likelihood = 0.0
        error_count = 0
        total_steps = len(Y)

        for period_idx, (y, u) in enumerate(zip(Y, U)):
            # Predict next state
            self.predict(A, B, u, V, C, w, W)

            # Update with observation
            self.update(y)

            # Add log probability
            log_prob, is_error = self.log_Pt()
            log_likelihood += log_prob
            error_count += 1*is_error

        return log_likelihood, error_count, total_steps

    def simulate(self, x0, P0, A, B, V, C, w, W, U, add_noise=True):
        """
        Simulate a sequence of states and observations given the model parameters.

        Parameters:
        x0 (numpy.ndarray): Initial state mean
        P0 (numpy.ndarray): Initial state covariance
        A (numpy.ndarray): State transition matrix
        B (numpy.ndarray): Control input matrix
        V (numpy.ndarray): Process noise covariance
        C (numpy.ndarray): Observation matrix
        w (numpy.ndarray): Observation offset
        W (numpy.ndarray): Observation noise covariance
        U (list): List of control inputs
        add_noise (bool): Whether to add process and measurement noise

        Returns:
        tuple: (X, Y) where X is list of states and Y is list of observations
        """
        # Initialize
        n = len(U)
        dim_x = x0.shape[0]
        dim_y = C.shape[0]

        # Draw initial state
        if add_noise and np.any(P0 > 0):
            try:
                L = np.linalg.cholesky(ensure_psd(P0))
                x = x0 + L @ np.random.randn(dim_x, 1)
            except np.linalg.LinAlgError:
                x = x0
        else:
            x = x0.copy()

        # Generate sample paths
        X, Y = [], []
        for u in U:
            # State transition
            x = A @ x + B @ u
            X.append(x.copy())

            # Measurement
            y = C @ x + w
            Y.append(y.copy())
        X, Y = np.array(X), np.array(Y)

        # Prepare disturbance and noise
        try:
            L_V = np.linalg.cholesky(ensure_psd(V))
        except np.linalg.LinAlgError:
            L_V = np.zeros_like(V)
        try:
            L_W = np.linalg.cholesky(ensure_psd(W))
        except np.linalg.LinAlgError:
            L_W = np.zeros_like(W)

        # generate state disturbance and measurement noise
        Disturbance, Noise = [], []
        x = np.zeros((len(x0),1))
        for u in U:
            # Process noise
            x = A @ x + L_V @ np.random.randn(dim_x, 1)
            Disturbance.append(x.copy())

            # Measurement noise
            y = C @ x + L_W @ np.random.randn(dim_y, 1)
            Noise.append(y.copy())
        Disturbance, Noise = np.array(Disturbance), np.array(Noise)

        # X + Disturbance = state sample path
        # Y + Noise = measurement sample path

        return X, Y, Disturbance, Noise
