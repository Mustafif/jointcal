"""
Simplified Differential Evolution calibration for Heston-Nandi GARCH parameters.

This module provides a clean implementation of parameter calibration using SciPy's
differential_evolution optimizer with proper handling of:
- Initial guess from HN GARCH fitting
- Parameter projection to ensure stationarity
- Convergence tracking and validation
"""

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize

from cal_loss import Calibration_Loss, ll_option_torch, ll_returns_torch
from hn import HestonNandiGARCH

# Device selection
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Parameter bounds: [omega, alpha, beta, gamma, lambda]
DEFAULT_BOUNDS = [
    (1e-10, 1e-5),   # omega: positive, small
    (1e-12, 1e-5),   # alpha: very small to small
    (0.2, 0.999),    # beta: close to 1
    (0.0, 30.0),     # gamma: leverage effect
    (0.0, 2.0),      # lambda: risk premium
]


def project_parameters(params, persistence_limit=0.999):
    """
    Project parameters to valid domain ensuring stationarity.

    The stationarity condition for HN-GARCH is: beta + alpha * gamma^2 < 1
    We use a stricter limit (default 0.999) to avoid numerical instability.

    Args:
        params: array-like of [omega, alpha, beta, gamma, lambda]
        persistence_limit: maximum allowed persistence (default 0.999)

    Returns:
        Projected parameters as numpy array
    """
    params = np.asarray(params, dtype=np.float64)

    omega = np.clip(params[0], 1e-12, None)
    alpha = np.clip(params[1], 1e-14, 1.0)
    beta = np.clip(params[2], 0.0, persistence_limit - 1e-6)
    gamma = np.clip(params[3], 0.0, None)
    lambda_param = np.clip(params[4], 0.0, None)

    # Enforce stationarity: beta + alpha * gamma^2 < persistence_limit
    # If violated, reduce gamma to satisfy constraint
    max_gamma_sq = (persistence_limit - beta) / (alpha + 1e-14)
    if max_gamma_sq > 0:
        max_gamma = np.sqrt(max_gamma_sq) * 0.999  # slight margin
        gamma = min(gamma, max_gamma)
    else:
        gamma = 0.0

    # Double-check persistence
    persistence = beta + alpha * gamma ** 2
    if persistence >= persistence_limit:
        # Reduce beta as fallback
        beta = max(0.0, persistence_limit - alpha * gamma ** 2 - 1e-6)

    return np.array([omega, alpha, beta, gamma, lambda_param], dtype=np.float64)


class DECalibrator:
    """
    Differential Evolution calibrator for Heston-Nandi GARCH parameters.

    This class encapsulates the calibration process with proper state management,
    avoiding issues with mutable closures and ensuring consistent evaluation.
    """

    def __init__(
        self,
        model,
        dataset,
        bounds=None,
        true_params=None,
        returns_weight=0.5,
        sigma_eps=0.01,
        # Regularization controls
        reg_weight: float = 0.0,
        reg_center="hn",
        gamma_reg_weight: float = 0.0,
        gamma_reg_center="hn",
        lambda_reg_weight: float = 0.0,
        lambda_reg_center="hn",
    ):
        """
        Initialize the calibrator.

        Args:
            model: Trained neural network for option pricing
            dataset: CalibrationDataset with returns and option data
            bounds: Parameter bounds (default: DEFAULT_BOUNDS)
            true_params: True parameters for error reporting (optional)
            returns_weight: Weight for returns likelihood (default 0.5)
            sigma_eps: Noise parameter for option likelihood (default 0.01)
            reg_weight: L2 regularization weight for full parameter vector (default 0.0)
            reg_center: Center for full-vector regularization ('hn', 'true', or array-like)
            gamma_reg_weight: Regularization weight for gamma (default 0.0)
            gamma_reg_center: Center for gamma regularization ('hn', 'true', or float)
            lambda_reg_weight: Regularization weight for lambda (default 0.0)
            lambda_reg_center: Center for lambda regularization ('hn', 'true', or float)
        """
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.bounds = bounds if bounds is not None else DEFAULT_BOUNDS
        self.true_params = np.asarray(true_params) if true_params is not None else None
        self.returns_weight = returns_weight
        self.sigma_eps = sigma_eps

        # Precompute tensors on device
        self.X = dataset.X.to(device)
        self.sigma_obs = dataset.sigma.to(device)
        self.returns = dataset.returns.to(device)
        self.N = len(self.returns)
        self.M = len(dataset)

        # Extract mean risk-free rate
        self.r_daily = self.X[:, 2].mean()

        # State tracking
        self.eval_count = 0
        self.best_loss = float('inf')
        self.best_params = None
        self.history = []
        self.iteration = 0

        # Initial guess from HN GARCH
        self.initial_guess = self._fit_initial_guess()
        self.initial_loss = None

        # ---- Regularization configuration (resolve centers now that initial is available) ----
        self.reg_weight = float(reg_weight)
        self.reg_center = reg_center
        self.reg_center_np = None
        if self.reg_weight > 0.0:
            if isinstance(reg_center, (list, tuple, np.ndarray)):
                self.reg_center_np = np.asarray(reg_center, dtype=float)
            elif isinstance(reg_center, str) and reg_center.lower() == "true" and self.true_params is not None:
                self.reg_center_np = np.asarray(self.true_params, dtype=float)
            else:
                # default to HN initial guess
                self.reg_center_np = np.asarray(self.initial_guess, dtype=float)

        self.gamma_reg_weight = float(gamma_reg_weight)
        self.gamma_reg_center = gamma_reg_center
        self.gamma_center_val = None
        if self.gamma_reg_weight > 0.0:
            if isinstance(gamma_reg_center, (int, float)):
                self.gamma_center_val = float(gamma_reg_center)
            elif isinstance(gamma_reg_center, str) and gamma_reg_center.lower() == "true" and self.true_params is not None:
                self.gamma_center_val = float(self.true_params[3])
            else:
                # default to HN initial guess gamma
                self.gamma_center_val = float(self.initial_guess[3])

        self.lambda_reg_weight = float(lambda_reg_weight)
        self.lambda_reg_center = lambda_reg_center
        self.lambda_center_val = None
        if self.lambda_reg_weight > 0.0:
            if isinstance(lambda_reg_center, (int, float)):
                self.lambda_center_val = float(lambda_reg_center)
            elif isinstance(lambda_reg_center, str) and lambda_reg_center.lower() == "true" and self.true_params is not None:
                self.lambda_center_val = float(self.true_params[4])
            else:
                # default to HN initial guess lambda
                self.lambda_center_val = float(self.initial_guess[4])

    def _fit_initial_guess(self):
        """Fit HN GARCH model to get initial parameter estimates."""
        hn_model = HestonNandiGARCH(self.returns.cpu().numpy())
        hn_model.fit()
        initial = np.asarray(hn_model.fitted_params, dtype=np.float64)

        # Clip to bounds
        lower = np.array([b[0] for b in self.bounds])
        upper = np.array([b[1] for b in self.bounds])
        initial = np.clip(initial, lower, upper)

        return initial

    def objective(self, params):
        """
        Compute calibration loss for given parameters.

        Args:
            params: numpy array [omega, alpha, beta, gamma, lambda]

        Returns:
            float: Loss value (lower is better)
        """
        self.eval_count += 1

        # Project parameters to valid domain
        params_proj = project_parameters(params)

        # Clip to bounds
        lower = np.array([b[0] for b in self.bounds])
        upper = np.array([b[1] for b in self.bounds])
        params_proj = np.clip(params_proj, lower, upper)

        # Check stationarity
        omega, alpha, beta, gamma, lambda_ = params_proj
        persistence = beta + alpha * gamma ** 2

        if persistence >= 0.999:
            return 1e9  # Penalty for non-stationary

        # Check unconditional variance
        denom = 1.0 - persistence
        if denom <= 1e-8:
            return 1e9

        h_unconditional = (omega + alpha) / denom
        if h_unconditional > 1.0 or h_unconditional < 0:
            return 1e9  # Unreasonable unconditional variance

        # Convert to torch and compute loss
        params_tensor = torch.tensor(params_proj, device=device, dtype=torch.float32)

        try:
            with torch.no_grad():
                loss = Calibration_Loss(
                    params_tensor,
                    self.returns,
                    self.sigma_obs,
                    self.model,
                    self.X,
                    self.N,
                    self.M
                )
                loss_val = float(loss.cpu().item())
        except Exception as e:
            # On exception return a large penalty so DE moves away
            return 1e9

        # ---- Regularization terms (if configured) ----
        reg_val = 0.0
        try:
            if getattr(self, "reg_weight", 0.0) and getattr(self, "reg_center_np", None) is not None:
                l2_c = np.linalg.norm(params_proj - self.reg_center_np)
                reg_term = float(self.reg_weight * (l2_c ** 2))
                reg_val += reg_term
                loss_val = float(loss_val + reg_term)
        except Exception:
            pass

        try:
            if getattr(self, "gamma_reg_weight", 0.0) and getattr(self, "gamma_center_val", None) is not None:
                gamma_term = float(self.gamma_reg_weight * ((gamma - self.gamma_center_val) ** 2))
                reg_val += gamma_term
                loss_val = float(loss_val + gamma_term)
        except Exception:
            pass

        try:
            if getattr(self, "lambda_reg_weight", 0.0) and getattr(self, "lambda_center_val", None) is not None:
                lambda_term = float(self.lambda_reg_weight * ((lambda_ - self.lambda_center_val) ** 2))
                reg_val += lambda_term
                loss_val = float(loss_val + lambda_term)
        except Exception:
            pass

        # Cache the last regularization contribution for diagnostics
        self._last_reg = reg_val

        # Debugging: alert if gamma/lambda are drifting far from HN initial guess (only informational)
        try:
            if getattr(self, "initial_guess", None) is not None and reg_val == 0.0:
                gamma_init = float(self.initial_guess[3])
                lambda_init = float(self.initial_guess[4])
                if gamma_init > 0 and gamma > max(10.0, 3.0 * gamma_init):
                    print(f"WARNING: gamma drifting high: {gamma:.2f} (init {gamma_init:.2f})")
                if lambda_init >= 0 and lambda_ > max(2.0, 5.0 * (lambda_init + 1e-8)):
                    print(f"WARNING: lambda drifting high: {lambda_:.2f} (init {lambda_init:.2f})")
        except Exception:
            pass

        # Check for numerical issues after regularization
        if np.isnan(loss_val) or np.isinf(loss_val) or loss_val > 1e8:
            return 1e9

        # Track best (includes regularization)
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_params = params_proj.copy()

        return loss_val

    def callback(self, xk, convergence=None):
        """
        Callback for tracking optimization progress.

        Note: We use cached best_loss instead of re-evaluating to avoid
        inconsistency and wasted computation.
        """
        self.iteration += 1

        # Grab latest regularizer contribution (if any)
        reg_val = float(getattr(self, "_last_reg", 0.0))

        # Record history (include reg for diagnostics)
        self.history.append({
            'iteration': self.iteration,
            'loss': self.best_loss,
            'params': self.best_params.copy() if self.best_params is not None else None,
            'reg': reg_val
        })

        # Build status string
        status_parts = [f"Iter {self.iteration:4d}", f"Loss: {self.best_loss:.6f}"]

        if reg_val and reg_val > 0.0:
            status_parts.append(f"Reg: {reg_val:.6f}")

        if self.true_params is not None and self.best_params is not None:
            l2_error = np.linalg.norm(self.best_params - self.true_params)
            status_parts.append(f"L2 Error: {l2_error:.6f}")

        if self.best_params is not None:
            p = self.best_params
            status_parts.append(
                f"Params: ω={p[0]:.2e}, α={p[1]:.2e}, β={p[2]:.4f}, γ={p[3]:.2f}, λ={p[4]:.2f}"
            )

        print(" | ".join(status_parts))

    def create_initial_population(self, popsize):
        """
        Create initial population with HN guess seeded in.

        Args:
            popsize: Population size multiplier

        Returns:
            numpy array of shape (n_pop, n_params)
        """
        n_params = len(self.bounds)
        n_pop = max(4, popsize * n_params)

        lower = np.array([b[0] for b in self.bounds])
        upper = np.array([b[1] for b in self.bounds])

        # Random initialization
        init_pop = np.random.rand(n_pop, n_params) * (upper - lower) + lower

        # Insert initial guess as first member
        init_pop[0] = self.initial_guess.copy()

        # Add perturbations around initial guess (indices 1-4)
        n_perturb = min(4, n_pop - 1)
        for k in range(n_perturb):
            noise = 1.0 + np.random.randn(n_params) * 0.05
            perturbed = self.initial_guess * noise

            # Handle zeros
            zero_mask = self.initial_guess == 0
            if np.any(zero_mask):
                perturbed[zero_mask] = lower[zero_mask] + (upper[zero_mask] - lower[zero_mask]) * 0.01 * np.random.rand()

            init_pop[1 + k] = np.clip(perturbed, lower, upper)

        return init_pop

    def calibrate(
        self,
        popsize=10,
        maxiter=100,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-6,
        polish=False,
        local_refine=True,
        refine_maxiter=200,
        debug_init_pop=False,
    ):
        """
        Run differential evolution calibration.

        Args:
            popsize: Population size multiplier
            maxiter: Maximum iterations
            strategy: DE strategy
            mutation: Mutation constant or range
            recombination: Crossover probability
            tol: Convergence tolerance
            polish: Use scipy's built-in polishing
            local_refine: Run L-BFGS-B refinement after DE
            refine_maxiter: Max iterations for local refinement
            debug_init_pop: If True, evaluate and print initial population statistics
        """
        print(f"Starting DE calibration: {self.M} options, {self.N} returns")
        print(f"Population: {popsize}x, Max iterations: {maxiter}")
        print(f"Strategy: {strategy}, Mutation: {mutation}, Recombination: {recombination}")
        print(f"Initial guess: {self.initial_guess}")

        # Evaluate initial guess
        self.initial_loss = self.objective(self.initial_guess)
        print(f"Initial guess loss: {self.initial_loss:.6f}")

        if self.true_params is not None:
            init_error = np.linalg.norm(self.initial_guess - self.true_params)
            print(f"Initial guess L2 error: {init_error:.6f}")

        # Create initial population
        init_pop = self.create_initial_population(popsize)

        # Optional debug: evaluate initial population losses (with diagnostics)
        if debug_init_pop:
            print("DEBUG: Evaluating initial population losses (this may be slow)...")
            try:
                init_losses = []
                for i in range(init_pop.shape[0]):
                    try:
                        loss_i = self.objective(init_pop[i])
                    except Exception as err:
                        print(f"DEBUG: Member {i} evaluation raised: {err}")
                        loss_i = float("nan")
                    init_losses.append(loss_i)
                init_losses = np.array(init_losses, dtype=float)

                print(f"DEBUG: Initial population size: {init_pop.shape[0]}")
                print(
                    f"DEBUG: Init Loss stats - min: {np.nanmin(init_losses):.6f}, "
                    f"median: {np.nanmedian(init_losses):.6f}, "
                    f"max: {np.nanmax(init_losses):.6f}, "
                    f"mean: {np.nanmean(init_losses):.6f}"
                )
                huge_count = int(np.sum(init_losses > 1e6))
                print(f"DEBUG: Number of huge losses (>1e6): {huge_count}")

                # Provide diagnostics for the best initial member (mean option LO and mean return LR)
                try:
                    min_idx = int(np.nanargmin(init_losses))
                    best_loss = init_losses[min_idx]
                    print(f"DEBUG: Best init member index: {min_idx}, loss: {best_loss:.6f}")

                    # Use a limited sample to compute option diagnostics to avoid OOM
                    sample_size = min(500, self.M)
                    sample_idx = np.linspace(0, self.M - 1, num=sample_size, dtype=int)
                    X_sample = self.X[sample_idx].clone()
                    sigma_sample = self.sigma_obs[sample_idx]

                    cand = project_parameters(init_pop[min_idx])
                    p_tensor = torch.tensor(cand, device=device, dtype=torch.float32)

                    # Inject parameters into the sample features (same convention as Calibration_Loss)
                    x_s = X_sample.clone()
                    x_s[:, 5] = p_tensor[1]
                    x_s[:, 6] = p_tensor[2]
                    x_s[:, 7] = p_tensor[0]
                    x_s[:, 8] = p_tensor[3]
                    x_s[:, 9] = p_tensor[4]
                    eps = 1e-8
                    x_s[:, 19] = torch.log(p_tensor[3] + eps)
                    x_s[:, 20] = torch.log(p_tensor[0] + eps)
                    x_s[:, 21] = torch.log(p_tensor[4] + eps)
                    x_s[:, 22] = torch.log(p_tensor[1] + eps)
                    x_s[:, 23] = torch.log(p_tensor[2] + eps)

                    with torch.no_grad():
                        sigma_cand = self.model(x_s).squeeze()
                    lo = ll_option_torch(sigma_sample, sigma_cand, sigma_eps=self.sigma_eps)
                    mean_lo = float((lo / sigma_cand.shape[0]).cpu().item())

                    try:
                        lr = ll_returns_torch(self.returns, p_tensor, self.r_daily)
                        mean_lr = float((lr / self.N).cpu().item())
                    except Exception:
                        mean_lr = None

                    print(f"DEBUG: Best member diagnostics - mean_lo: {mean_lo:.6f}, mean_lr: {mean_lr}")

                    # If true parameters are available, show their diagnostics for comparison
                    if self.true_params is not None:
                        l2_best = np.linalg.norm(init_pop[min_idx] - self.true_params)
                        print(f"DEBUG: L2 error of best init member to true params: {l2_best:.6f}")

                        true_proj = project_parameters(self.true_params)
                        p_t = torch.tensor(true_proj, device=device, dtype=torch.float32)

                        x_t = X_sample.clone()
                        x_t[:, 5] = p_t[1]
                        x_t[:, 6] = p_t[2]
                        x_t[:, 7] = p_t[0]
                        x_t[:, 8] = p_t[3]
                        x_t[:, 9] = p_t[4]
                        x_t[:, 19] = torch.log(p_t[3] + eps)
                        x_t[:, 20] = torch.log(p_t[0] + eps)
                        x_t[:, 21] = torch.log(p_t[4] + eps)
                        x_t[:, 22] = torch.log(p_t[1] + eps)
                        x_t[:, 23] = torch.log(p_t[2] + eps)

                        with torch.no_grad():
                            sigma_true = self.model(x_t).squeeze()
                        lo_true = ll_option_torch(sigma_sample, sigma_true, sigma_eps=self.sigma_eps)
                        mean_lo_true = float((lo_true / sigma_true.shape[0]).cpu().item())

                        try:
                            lr_true = ll_returns_torch(self.returns, p_t, self.r_daily)
                            mean_lr_true = float((lr_true / self.N).cpu().item())
                        except Exception:
                            mean_lr_true = None

                        print(f"DEBUG: True params diagnostics - mean_lo: {mean_lo_true:.6f}, mean_lr: {mean_lr_true}")
                except ValueError:
                    print("DEBUG: All initial population losses are NaN, cannot identify best member.")
            except Exception as e:
                print(f"DEBUG: Failed to evaluate initial population: {e}")

        # Run DE
        print("\nStarting optimization...")
        result = differential_evolution(
            func=self.objective,
            bounds=self.bounds,
            strategy=strategy,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            callback=self.callback,
            polish=polish,
            init=init_pop,
            disp=False,
        )

        # Get best result
        best_params = project_parameters(result.x)
        best_loss = result.fun

        print(f"\n✅ DE complete: success={result.success}, iterations={result.nit}")
        print(f"Final DE loss: {best_loss:.6f}")

        # Optional local refinement
        if local_refine:
            print("\n🔧 Running L-BFGS-B refinement...")
            best_params, best_loss = self._local_refine(
                best_params, maxiter=refine_maxiter
            )

        # Final validation
        self._print_results(best_params, best_loss)

        return best_params, self.history, self.initial_guess

    def _local_refine(self, start_params, maxiter=200):
        """
        Local refinement using L-BFGS-B starting from given parameters.

        Also tries starting from initial guess and returns the best result.
        """
        candidates = [
            ('DE result', start_params),
            ('HN initial', self.initial_guess),
        ]

        results = []

        for name, x0 in candidates:
            try:
                res = minimize(
                    self.objective,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-9}
                )
                params_refined = project_parameters(res.x)
                loss_refined = self.objective(params_refined)
                results.append((name, params_refined, loss_refined))
                print(f"  {name}: loss={loss_refined:.6f}")
            except Exception as e:
                print(f"  {name}: refinement failed - {e}")

        # Select best
        if results:
            best = min(results, key=lambda x: x[2])
            print(f"  Best refinement: {best[0]} with loss={best[2]:.6f}")
            return best[1], best[2]

        return start_params, self.objective(start_params)

    def _print_results(self, params, loss):
        """Print final calibration results."""
        print("\n" + "=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)

        omega, alpha, beta, gamma, lambda_ = params
        persistence = beta + alpha * gamma ** 2

        print(f"Final Loss: {loss:.6f}")
        print(f"Initial Loss: {self.initial_loss:.6f}")
        print(f"Improvement: {self.initial_loss - loss:.6f}")
        print()
        print("Parameters:")
        print(f"  ω (omega):  {omega:.6e}")
        print(f"  α (alpha):  {alpha:.6e}")
        print(f"  β (beta):   {beta:.6f}")
        print(f"  γ (gamma):  {gamma:.6f}")
        print(f"  λ (lambda): {lambda_:.6f}")
        print()
        print(f"Persistence (β + αγ²): {persistence:.6f}")
        print(f"Stationary: {'✅' if persistence < 1.0 else '❌'}")

        if self.true_params is not None:
            l2_error = np.linalg.norm(params - self.true_params)
            init_error = np.linalg.norm(self.initial_guess - self.true_params)
            print()
            print(f"L2 Error (final): {l2_error:.6f}")
            print(f"L2 Error (initial): {init_error:.6f}")
            print(f"Error improvement: {init_error - l2_error:.6f}")

        print("=" * 60)


def calibrate_scipy_de(
    model,
    dataset,
    popsize=10,
    maxiter=100,
    strategy='best1bin',
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=False,
    atol=1e-6,
    true_vals=None,
    local_refine=True,
    refine_maxiter=200,
    # Unused legacy parameters (kept for API compatibility)
    debug_init_pop=False,
    reg_weight=0.0,
    reg_center="hn",
    gamma_reg_weight=0.0,
    gamma_reg_center="hn",
    lambda_reg_weight=0.0,
    lambda_reg_center="hn",
    refine_w_ret=0.5,
    refine_tol=1e-9,
    local_de=False,
    local_de_radius=0.2,
    local_de_popsize=15,
    local_de_maxiter=50,
    local_de_strategy="rand1bin",
    sigma_reg_weight=0.0,
):
    """
    Calibrate GARCH parameters using SciPy's Differential Evolution.

    This is a simplified wrapper around DECalibrator for backward compatibility.

    Args:
        model: Trained neural network model
        dataset: Calibration dataset
        popsize: DE population size multiplier
        maxiter: Maximum iterations
        strategy: DE strategy
        mutation: Mutation constant or range
        recombination: Crossover probability
        polish: Use scipy's built-in polishing
        atol: Convergence tolerance
        true_vals: True parameters for error reporting
        local_refine: Run L-BFGS-B refinement after DE
        refine_maxiter: Max iterations for local refinement

    Returns:
        tuple: (best_params, history, initial_guess)
    """
    calibrator = DECalibrator(
        model=model,
        dataset=dataset,
        true_params=true_vals,
        reg_weight=reg_weight,
        reg_center=reg_center,
        gamma_reg_weight=gamma_reg_weight,
        gamma_reg_center=gamma_reg_center,
        lambda_reg_weight=lambda_reg_weight,
        lambda_reg_center=lambda_reg_center,
    )

    return calibrator.calibrate(
        popsize=popsize,
        maxiter=maxiter,
        strategy=strategy,
        mutation=mutation,
        recombination=recombination,
        tol=atol,
        polish=polish,
        local_refine=local_refine,
        refine_maxiter=refine_maxiter,
        debug_init_pop=debug_init_pop,
    )


if __name__ == "__main__":
    # Simple test
    import json

    from dataset2 import cal_dataset

    MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"

    print("Loading model...")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print("✅ Model loaded")

    print("Loading dataset...")
    dataset = cal_dataset(
        "joint_dataset/scalable_hn_dataset_250x60.csv",
        "joint_dataset/assetprices.csv"
    )
    print(f"✅ Dataset loaded: {len(dataset)} options")

    # Load true parameters
    with open("true_params/1.json") as f:
        true_params_dict = json.load(f)

    true_vals = np.array([
        true_params_dict["omega"],
        true_params_dict["alpha"],
        true_params_dict["beta"],
        true_params_dict["gamma"],
        true_params_dict["lambda"],
    ])

    print(f"True params: {true_vals}")

    # Run calibration
    params, history, initial_guess = calibrate_scipy_de(
        model=model,
        dataset=dataset,
        popsize=10,
        maxiter=50,
        true_vals=true_vals,
        local_refine=True,
    )
