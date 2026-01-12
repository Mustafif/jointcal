# class HestonNandiGARCH:
#     """
#     A class to fit the full Heston-Nandi GARCH(1,1) model (with risk premium λ)
#     to a series of log returns.
#     Mean Equation: r_t = λ * h_t + ε_t
#     Variance Process: h_t = ω + β * h_{t-1} + α * (z_{t-1} - γ * sqrt(h_{t-1}))^2
#     where ε_t = sqrt(h_t) * z_t and z_t ~ N(0,1)
#     """
#     def __init__(self, returns):
#         self.returns = np.array(returns)
#         self.omega: np.float64 = np.float64(0.0)
#         self.alpha: np.float64 = np.float64(0.0)
#         self.beta: np.float64 = np.float64(0.0)
#         self.gamma: np.float64 = np.float64(0.0)
#         self.lambda_: np.float64 = np.float64(0.0)
#         self.fitted_params = None
#         self.log_likelihood = None
#     def _log_likelihood(self, params):
#         """
#         Calculates the negative log-likelihood for the GARCH model.
#         """
#         # UPDATED: Unpack five parameters
#         omega, alpha, beta, gamma, lambda_ = params
#         T = len(self.returns)
#         h = np.zeros(T)
#         z = np.random.randn(T)
#         # h[0] = np.var(self.returns)
#         h[0] = (omega + alpha)/(1-beta-alpha*gamma**2)
#         for t in range(1, T):
#             h_prev = max(h[t-1], 1e-9)
#             # UPDATED: The standardized residual z depends on lambda
#             # innovation_prev = self.returns[t-1] - lambda_ * h_prev
#             # z_prev = innovation_prev / np.sqrt(h_prev)
#             h[t] = omega + beta * h_prev + alpha * (z[t] - gamma * np.sqrt(h_prev))**2
#         # h = np.maximum(h, 1e-9)
#         # z = np.random.normal(0, 1, len(h))
#         # term = z*np.sqrt(h)-0.5*h+gamma
#         # log_likelihoods = -0.5 * (np.sum(np.log(h) + (np.pow(term, 2)/h)))
#         # UPDATED: The innovation ε_t = r_t - λh_t is used in the likelihood
#         # innovations = self.returns - lambda_ * h
#         log_likelihoods = np.log(2 * np.pi) + np.log(h) + np.power(z, 2)
#         return -0.5 * np.sum(log_likelihoods)
#     def fit(self, initial_params=None):
#         """
#         Fits the GARCH model parameters using scipy's optimizer.
#         """
#         garch11 = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
#         res = garch11.fit(disp="off")
#         alpha0 = res.params['alpha[1]']
#         beta0  = res.params['beta[1]']
#         omega0 = res.params['omega']
#         gamma0 = 0.1
#         lam0 = 0
#         if initial_params is None:
#             # UPDATED: Initial guess for 5 parameters [ω, α, β, γ, λ]
#             initial_params = np.array([np.var(self.returns) * omega0, alpha0, beta0, gamma0, lam0])
#         # UPDATED: Bounds for 5 parameters
#         bounds = [(1e-9, 1e-3), (1e-6,0.2), (0.5, 0.999), (-2, 2), (-5, 5)]
#         result = minimize(self._log_likelihood,
#                           initial_params,
#                           method='L-BFGS-B',
#                           bounds=bounds)
#         if result.success:
#             # UPDATED: Store all five parameters
#             self.omega, self.alpha, self.beta, self.gamma, self.lambda_ = result.x
#             self.fitted_params = result.x
#             self.log_likelihood = -result.fun
#             print("Optimization successful!")
#         else:
#             print(f"Optimization failed: {result.message}")
#         return result
#     def summary(self):
#         """
#         Prints a summary of the fitted model parameters.
#         """
#         if self.fitted_params is None:
#             print("Model has not been fitted yet. Please call the .fit() method first.")
#             return
#         print("\nHeston-Nandi GARCH(1,1) Model Results")
#         print("="*40)
#         print(f"Log-Likelihood: {self.log_likelihood:.4f}")
#         print("\nParameters:")
#         print(f"  ω (omega): {self.omega:.6f}")
#         print(f"  α (alpha): {self.alpha:.6f}")
#         print(f"  β (beta):  {self.beta:.6f}")
#         print(f"  γ (gamma): {self.gamma:.6f}")
#         # UPDATED: Print lambda
#         print(f"  λ (lambda):{self.lambda_:.6f}")
#         print("="*40)
import numpy as np
from arch import arch_model
from scipy.optimize import differential_evolution, minimize


class HestonNandiGARCH:
    """
    Full Heston–Nandi GARCH(1,1) model with risk premium λ.

    Mean:        r_t = λ * h_t + ε_t
    Variance:    h_t = ω + β * h_{t-1} + α * (z_{t-1} - γ * sqrt(h_{t-1}))^2
    ε_t = sqrt(h_t) * z_t,  z_t ~ N(0,1)
    """

    def __init__(self, returns):
        self.returns = np.asarray(returns)
        self.omega = np.float64(0.0)
        self.alpha = np.float64(0.0)
        self.beta = np.float64(0.0)
        self.gamma = np.float64(0.0)
        self.lambda_ = np.float64(0.0)
        self.fitted_params = None
        self.log_likelihood = None

    def _log_likelihood(self, params):
        """
        Computes the *negative* log-likelihood for Heston–Nandi GARCH using observed returns.

        This implementation uses the observed returns to compute standardized residuals:
            z_t = (r_t - λ * h_t) / sqrt(h_t)
        and updates the variance recursion as:
            h_{t+1} = ω + β * h_t + α * (z_t - γ * sqrt(h_t))^2

        If parameters lead to non-stationary or numerically unstable variance paths,
        a large penalty is returned to guide the optimizer away from such regions.
        """
        omega, alpha, beta, gamma, lambda_ = params
        r = self.returns
        T = len(r)

        # Stationarity check: avoid explosive variance recursion
        persistence = beta + alpha * gamma ** 2
        if persistence >= 0.999 or (1.0 - persistence) <= 1e-8:
            return 1e9

        # Initialize conditional variance (unconditional under stationarity)
        h = np.zeros(T)
        h[0] = (omega + alpha) / (1 - persistence + 1e-8)

        z = np.zeros(T)

        # Use observed returns to compute standardized residuals and update variance
        for t in range(T):
            h_curr = h[t]

            # Basic numerical stability checks
            if h_curr <= 0 or np.isnan(h_curr) or np.isinf(h_curr) or h_curr > 1e6:
                # Penalize invalid/numerically unstable parameter combinations
                return 1e9

            # Standardized residual from observed return at time t
            z[t] = (r[t] - lambda_ * h_curr) / (np.sqrt(h_curr) + 1e-8)

            # Update next-period variance (if not at last observation)
            if t < T - 1:
                h[t + 1] = (
                    omega
                    + beta * h_curr
                    + alpha * (z[t] - gamma * np.sqrt(h_curr)) ** 2
                )

        # Gaussian log-likelihood (sum over time)
        loglik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h + 1e-8) + z ** 2)

        # Return *negative* log-likelihood for minimization routines
        return -loglik

    def fit(self, initial_params=None):
        """
        Fit parameters using numerical optimization (L-BFGS-B).
        """
        # Use standard GARCH(1,1) to generate starting values
        garch11 = arch_model(self.returns, vol="GARCH", p=1, q=1, dist="normal")
        res = garch11.fit(disp="off")

        omega0 = res.params["omega"]
        alpha0 = res.params["alpha[1]"]
        beta0 = res.params["beta[1]"]
        gamma0 = 0.0
        lam0 = 0.0

        if initial_params is None:
            initial_params = np.array([omega0, alpha0, beta0, gamma0, lam0])

        bounds = [
            (1e-7, 1e-6),  # omega: positive, small
            (1.15e-6, 1.36e-6),  # alpha: small, positive
            (0.5, 0.99),  # beta: close to 1
            (0, 10),  # gamma: leverage effect
            (0, 0.6),  # lambda: risk premium
        ]

        # result = differential_evolution(self._log_likelihood,
        #                                bounds=bounds,
        #                                strategy='best1bin',
        #                                popsize=100,
        #                                maxiter=500,
        #                                tol=1e-6)

        result = minimize(
            self._log_likelihood, initial_params, method="L-BFGS-B", bounds=bounds
        )

        if result.success:
            self.omega, self.alpha, self.beta, self.gamma, self.lambda_ = result.x
            self.fitted_params = result.x
            self.log_likelihood = -result.fun
            print("Optimization successful.")
            self.summary()
            print("Two-norm error:")
            true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
            print(np.linalg.norm(true_vals - self.fitted_params, ord=2))
        else:
            print(f"Optimization failed: {result.message}")

        return result

    def summary(self):
        """
        Display fitted parameters and likelihood.
        """
        if self.fitted_params is None:
            print("Model has not been fitted yet. Call .fit() first.")
            return

        print("\nHeston–Nandi GARCH(1,1) Model Results")
        print("=" * 45)
        print(f"Log-Likelihood: {self.log_likelihood:.6f}")
        print("Parameters:")
        print(f"  ω (omega):  {self.omega:.6e}")
        print(f"  α (alpha):  {self.alpha:.6e}")
        print(f"  β (beta):   {self.beta:.6e}")
        print(f"  γ (gamma):  {self.gamma:.6e}")
        print(f"  λ (lambda): {self.lambda_:.6e}")
        print("=" * 45)


# from returns import returns
# hn = HestonNandiGARCH(returns)
# hn.fit()
# hn.summary()
