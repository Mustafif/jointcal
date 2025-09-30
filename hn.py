import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats.distributions import alpha
from arch import arch_model

class HestonNandiGARCH:
    """
    A class to fit the full Heston-Nandi GARCH(1,1) model (with risk premium λ)
    to a series of log returns.

    Mean Equation: r_t = λ * h_t + ε_t
    Variance Process: h_t = ω + β * h_{t-1} + α * (z_{t-1} - γ * sqrt(h_{t-1}))^2
    where ε_t = sqrt(h_t) * z_t and z_t ~ N(0,1)
    """
    def __init__(self, returns):
        self.returns = np.array(returns)
        self.omega = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.lambda_ = None # Using lambda_ to avoid conflict with Python keyword
        self.fitted_params = None
        self.log_likelihood = None

    def _log_likelihood(self, params):
        """
        Calculates the negative log-likelihood for the GARCH model.
        """
        # UPDATED: Unpack five parameters
        omega, alpha, beta, gamma, lambda_ = params

        T = len(self.returns)
        h = np.zeros(T)
        z = np.random.randn(T)
        # h[0] = np.var(self.returns)
        h[0] = (omega + alpha)/(1-beta-alpha*gamma**2)

        for t in range(1, T):
            h_prev = max(h[t-1], 1e-9)

            # UPDATED: The standardized residual z depends on lambda
            # innovation_prev = self.returns[t-1] - lambda_ * h_prev
            # z_prev = innovation_prev / np.sqrt(h_prev)


            h[t] = omega + beta * h_prev + alpha * (z[t] - gamma * np.sqrt(h_prev))**2

        # h = np.maximum(h, 1e-9)
        # z = np.random.normal(0, 1, len(h))
        # term = z*np.sqrt(h)-0.5*h+gamma

        # log_likelihoods = -0.5 * (np.sum(np.log(h) + (np.pow(term, 2)/h)))

        # UPDATED: The innovation ε_t = r_t - λh_t is used in the likelihood
        # innovations = self.returns - lambda_ * h
        log_likelihoods = -0.5 * (np.log(2 * np.pi) + np.log(h) + np.power(z, 2))

        return -np.sum(log_likelihoods)

    def fit(self, initial_params=None):
        """
        Fits the GARCH model parameters using scipy's optimizer.
        """
        garch11 = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
        res = garch11.fit(disp="off")

        alpha0 = res.params['alpha[1]']
        beta0  = res.params['beta[1]']
        omega0 = res.params['omega']
        gamma0 = 0.1
        lam0 = 0
        if initial_params is None:
            # UPDATED: Initial guess for 5 parameters [ω, α, β, γ, λ]
            initial_params = np.array([np.var(self.returns) * omega0, alpha0, beta0, gamma0, lam0])

        # UPDATED: Bounds for 5 parameters
        bounds = [(1e-9, None), (0, 1), (0, 1), (None, None), (None, None)]

        result = minimize(self._log_likelihood,
                          initial_params,
                          method='L-BFGS-B',
                          bounds=bounds)

        if result.success:
            # UPDATED: Store all five parameters
            self.omega, self.alpha, self.beta, self.gamma, self.lambda_ = result.x
            self.fitted_params = result.x
            self.log_likelihood = -result.fun
            print("Optimization successful!")
        else:
            print(f"Optimization failed: {result.message}")

        return result

    def summary(self):
        """
        Prints a summary of the fitted model parameters.
        """
        if self.fitted_params is None:
            print("Model has not been fitted yet. Please call the .fit() method first.")
            return

        print("\nHeston-Nandi GARCH(1,1) Model Results")
        print("="*40)
        print(f"Log-Likelihood: {self.log_likelihood:.4f}")
        print("\nParameters:")
        print(f"  ω (omega): {self.omega:.6f}")
        print(f"  α (alpha): {self.alpha:.6f}")
        print(f"  β (beta):  {self.beta:.6f}")
        print(f"  γ (gamma): {self.gamma:.6f}")
        # UPDATED: Print lambda
        print(f"  λ (lambda):{self.lambda_:.6f}")
        print("="*40)

from returns import returns
hn = HestonNandiGARCH(returns)
hn.fit()
hn.summary()
