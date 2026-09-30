import numpy as np

class OLS:
    """
    A custom Ordinary Least Squares implementation with improved numerical stability.
    """
    def __init__(self, y, X, add_constant=False):
        """
        Initialize the OLS model.

        Parameters:
        y (array-like): Dependent variable
        X (array-like): Independent variables
        add_constant (bool): Whether to add a constant column to X
        """
        # Convert inputs to numpy arrays if they aren't already
        self.y = np.asarray(y)
        self.X = np.asarray(X)

        # Add constant if requested (intercept term)
        if add_constant:
            self.X = np.column_stack((np.ones(len(self.X)), self.X))

        self.n, self.k = self.X.shape  # n observations, k parameters

        # Validate dimensions
        if len(self.y) != self.n:
            raise ValueError("y and X must have the same number of observations")

        # Fit the model
        self._fit()

    def _fit(self):
        """Fit the OLS model using numerically stable methods."""
        # Calculate coefficients: beta_hat = (X'X)^(-1)X'y
        # Using more stable SVD method for pseudo-inverse when possible
        try:
            # Try SVD method first (more stable)
            U, s, Vh = np.linalg.svd(self.X, full_matrices=False)
            s_inv = np.diag(1.0 / s)
            self.params = Vh.T @ s_inv @ U.T @ self.y
        except np.linalg.LinAlgError:
            # Fall back to traditional method
            XtX = self.X.T @ self.X
            XtX_inv = np.linalg.inv(XtX)
            Xty = self.X.T @ self.y
            self.params = XtX_inv @ Xty

        # Calculate fitted values: y_hat = X * beta_hat
        self.fitted_values = self.X @ self.params

        # Calculate residuals: e = y - y_hat
        self.residuals = self.y - self.fitted_values

        # Calculate residual sum of squares
        self.rss = np.sum(self.residuals**2)

        # Estimate of error variance: sigma**2 = RSS / (n - k)
        self.sigma_squared = self.rss / (self.n - self.k)

        # Calculate R-squared
        y_mean = np.mean(self.y)
        total_sum_squares = np.sum((self.y - y_mean)**2)
        self.r_squared = 1 - (self.rss / total_sum_squares)

        # Calculate adjusted R-squared
        self.adj_r_squared = 1 - ((1 - self.r_squared) * (self.n - 1) / (self.n - self.k))

        # Standard errors of the coefficients
        try:
            if self.n > self.k:  # Only if we have degrees of freedom
                XtX_inv = np.linalg.inv(self.X.T @ self.X)
                self.std_errors = np.sqrt(np.diag(XtX_inv) * self.sigma_squared)
            else:
                self.std_errors = np.full(self.k, np.nan)
        except np.linalg.LinAlgError:
            self.std_errors = np.full(self.k, np.nan)

    def predict(self, X_new):
        """
        Predict using the OLS model.

        Parameters:
        X_new (array-like): New independent variables

        Returns:
        array: Predicted values
        """
        X_new = np.asarray(X_new)

        # Check if we need to add a constant
        if self.X.shape[1] > X_new.shape[1] and np.allclose(self.X[:, 0], 1):
            X_new = np.column_stack((np.ones(len(X_new)), X_new))

        return X_new @ self.params

    def summary(self):
        """
        Print a summary of the OLS regression results.
        """
        print("OLS Regression Results")
        print("======================")
        print(f"Number of observations: {self.n}")
        print(f"Number of parameters: {self.k}")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Adjusted R-squared: {self.adj_r_squared:.4f}")
        print(f"Residual standard error: {np.sqrt(self.sigma_squared):.4f}")

        print("\nCoefficients:")
        print("-------------")
        for i in range(self.k):
            param_name = f"Intercept" if i == 0 and np.allclose(self.X[:, 0], 1) else f"X{i}"
            print(f"{param_name}: {self.params[i]:.4f} (SE: {self.std_errors[i]:.4f})")


def ols_estimate_ar1_params(y, x):
    """
    Estimate parameters for the AR1 model with an exogenous variable:
    y_{t} = beta_0 + beta_1 * y_{t-1} + beta_2 * x_{t} + epsilon_{t}

    Parameters:
    y (array-like): Time series y
    x (array-like): Covariate x

    Returns:
    dict: Dictionary with estimated parameters
    """
    # Convert inputs to numpy arrays
    y = np.asarray(y)
    x = np.asarray(x)

    # Create lagged variable Y_{t-1}
    y_lag = y[:-1]

    # Get corresponding Y_{t} and x_{t}
    y_t = y[1:]
    x_t = x[1:]

    # Create design matrix [Y_{t-1}, x_{t}]
    X = np.column_stack((y_lag, x_t))

    # Fit OLS model with intercept
    model = OLS(y_t, X, add_constant=True)

    beta_0 = model.params[0]  # Intercept
    beta_1 = model.params[1]  # AR coefficient (y_{t-1})
    beta_2 = model.params[2]  # Exogenous variable coefficient (x_{t})

    # Calculate estimated sigma from the residuals
    sigma_hat = np.sqrt(model.sigma_squared)

    # Convert AR(1) parameters to physical parameters if needed
    # For an AR(1) process: y_t = beta_0 + beta_1*y_{t-1} + beta_2*x_t + ε_t
    # Equivalent to: Δy_t = beta_0 + (beta_1-1)*y_{t-1} + beta_2*x_t + ε_t
    # Speed of mean reversion = -(beta_1-1) = 1-beta_1
    mean_reversion_speed = 1 - beta_1

    # Long-run multiplier (effect of x on y in equilibrium)
    long_run_multiplier = beta_2 / mean_reversion_speed if abs(mean_reversion_speed) > 1e-10 else np.nan

    # Long-run mean (if no exogenous variable)
    long_run_mean = beta_0 / mean_reversion_speed if abs(mean_reversion_speed) > 1e-10 else np.nan

    #model.summary()

    return {
        'beta_0': beta_0,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'sigma': sigma_hat,
        'mean_reversion_speed': mean_reversion_speed,
        'long_run_multiplier': long_run_multiplier,
        'long_run_mean': long_run_mean,
        'residuals': model.residuals,
        'fitted_values': model.fitted_values,
        'r_squared': model.r_squared,
        'model': model
    }


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    beta_true = np.array([1.5, -0.5, 2.0])  # intercept, beta1, beta2
    y = beta_true[0] + X[:, 0] * beta_true[1] + X[:, 1] * beta_true[2] + np.random.randn(n) * 0.5

    # Fit the model
    model = OLS(y, X, add_constant=True)
    print("True parameters:", beta_true)
    print("Estimated parameters:", model.params)
    model.summary()

    # AR(1) example
    ar_y = np.zeros(n)
    ar_x = np.random.randn(n)
    ar_y[0] = ar_x[0]
    for t in range(1, n):
        ar_y[t] = 0.2 + 0.7 * ar_y[t-1] + 0.5 * ar_x[t] + np.random.randn() * 0.3

    ar_result = ols_estimate_ar1_params(ar_y, ar_x)
    print("\nAR(1) Model Results:")
    print(f"beta_0: {ar_result['beta_0']:.4f}")
    print(f"beta_1: {ar_result['beta_1']:.4f}")
    print(f"beta_2: {ar_result['beta_2']:.4f}")
    print(f"sigma: {ar_result['sigma']:.4f}")
    print(f"Mean reversion speed: {ar_result['mean_reversion_speed']:.4f}")
    print(f"Long-run multiplier: {ar_result['long_run_multiplier']:.4f}")
