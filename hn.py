import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats.distributions import alpha
from arch import arch_model

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
from scipy.optimize import minimize
from arch import arch_model

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
        Computes the *negative* log-likelihood for Heston–Nandi GARCH.
        """
        omega, alpha, beta, gamma, lambda_ = params
        r = self.returns
        T = len(r)

        # Initialize conditional variance
        h = np.zeros(T)
        h[0] = np.var(r) if np.var(r) > 0 else 1e-6

        # Compute variance recursion
        for t in range(1, T):
            z_prev = (r[t-1] - lambda_ * h[t-1]) / np.sqrt(h[t-1])  # standardized residual
            h[t] = omega + beta * h[t-1] + alpha * (z_prev - gamma * np.sqrt(h[t-1]))**2
            h[t] = max(h[t], 1e-9)  # numerical safeguard

        # Compute standardized residuals
        z = (r - lambda_ * h) / np.sqrt(h)

        # Gaussian log-likelihood
        loglik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + z**2)

        # Return *negative* log-likelihood for minimization
        return -loglik

    def fit(self, initial_params=None):
        """
        Fit parameters using numerical optimization (L-BFGS-B).
        """
        # Use standard GARCH(1,1) to generate starting values
        garch11 = arch_model(self.returns, vol='GARCH', p=1, q=1, dist='normal')
        res = garch11.fit(disp="off")

        omega0 = res.params['omega']
        alpha0 = res.params['alpha[1]']
        beta0 = res.params['beta[1]']
        gamma0 = 0.0
        lam0 = 0.0

        if initial_params is None:
            initial_params = np.array([omega0, alpha0, beta0, gamma0, lam0])

        bounds = [
            (1e-9, 1e-3),   # ω
            (1e-6, 0.2),    # α
            (0.5, 0.999),   # β
            (-2.0, 2.0),    # γ
            (-5.0, 5.0)     # λ
        ]

        result = minimize(self._log_likelihood,
                          initial_params,
                          method='L-BFGS-B',
                          bounds=bounds)

        if result.success:
            self.omega, self.alpha, self.beta, self.gamma, self.lambda_ = result.x
            self.fitted_params = result.x
            self.log_likelihood = -result.fun
            print("Optimization successful.")
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
