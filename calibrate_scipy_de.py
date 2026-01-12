import json

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize

from cal_loss import Calibration_Loss, ll_option_torch, ll_returns_torch
from hn import HestonNandiGARCH

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)


def project_parameters(params):
    """Project parameters to valid domain and enforce stationarity (β + α γ^2 < persistence_cut).

    This function clamps individual parameter ranges and then enforces the
    Heston–Nandi stationarity constraint by limiting `gamma` (and if needed `beta`)
    so that beta + alpha * gamma^2 < persistence_cut (default 0.98). The stricter
    cut prevents DE from exploring severely persistent regions that cause numerical
    instability. A small `eps_alpha` is added to the denominator to avoid division by zero.
    """
    eps_alpha = 1e-12
    persistence_cut = 0.999

    if isinstance(params, torch.Tensor):
        omega = torch.clamp(params[0], min=1e-9)
        alpha = torch.clamp(params[1], min=0.0, max=1.0)
        # keep beta strictly below 1 for numeric stability
        beta = torch.clamp(params[2], min=0.0, max=1.0 - 1e-12)
        gamma = torch.clamp(params[3], min=0.0)
        lambda_param = torch.clamp(params[4], min=0.0)

        # Compute safe gamma upper bound: gamma_max = sqrt(max((1 - beta)/alpha, 0))
        # Add eps_alpha to avoid division by zero when alpha is zero.
        gamma_max = torch.sqrt(torch.clamp((persistence_cut - beta) / (alpha + eps_alpha), min=0.0))
        # Slightly reduce to ensure strict inequality
        gamma = torch.minimum(gamma, gamma_max * 0.99999999)

        # Recompute persistence; if still too large, reduce beta conservatively
        persistence = beta + alpha * gamma * gamma
        if (persistence >= persistence_cut).item():
            beta = torch.clamp(persistence_cut - alpha * gamma * gamma, min=0.0, max=1.0 - 1e-12)

        return torch.stack([omega, alpha, beta, gamma, lambda_param])
    else:
        # Handle numpy arrays
        omega = np.clip(params[0], a_min=1e-9, a_max=None)
        alpha = np.clip(params[1], a_min=0.0, a_max=1.0)
        beta = np.clip(params[2], a_min=0.0, a_max=1.0 - 1e-12)
        gamma = np.clip(params[3], a_min=0.0, a_max=None)
        lambda_param = np.clip(params[4], a_min=0.0, a_max=None)

        eps_alpha = 1e-12
        gamma_max = np.sqrt(max((persistence_cut - beta) / (alpha + eps_alpha), 0.0))
        gamma = min(gamma, gamma_max * 0.99999999)

        persistence = beta + alpha * gamma * gamma
        if persistence >= persistence_cut:
            beta = float(max(min(persistence_cut - alpha * gamma * gamma, 1.0 - 1e-12), 0.0))

        beta = np.clip(beta, 0.0, 1.0 - 1e-12)
        return np.array([omega, alpha, beta, gamma, lambda_param])


def calibrate_scipy_de(
    model,
    dataset,
    popsize,
    maxiter,
    strategy,
    mutation,
    recombination,
    polish=False,
    atol=1e-6,
    true_vals=None,
    debug_init_pop=False,
    # General L2 regularization for full parameter vector (0 disables)
    reg_weight: float = 0.0,
    reg_center = "hn",
    # Gamma-specific regularization (penalize deviation from center)
    gamma_reg_weight: float = 0.0,
    gamma_reg_center = "hn",
    # Lambda-specific regularization (penalize deviation from center)
    lambda_reg_weight: float = 0.0,
    lambda_reg_center = "hn",
    local_refine=False,
    refine_w_ret=0.5,
    refine_maxiter=200,
    refine_tol=1e-9,
    # Optional local DE search options (runs AFTER global DE)
    local_de=False,
    local_de_radius=0.2,      # fraction of global range to search around HN initial guess
    local_de_popsize=15,      # population multiplier for local DE (smaller by default)
    local_de_maxiter=50,      # iterations for the local DE run
    local_de_strategy="rand1bin",
    sigma_reg_weight: float = 0.0,
):
    """
    Calibrate GARCH parameters using SciPy's Differential Evolution

    Args:
        model: Trained neural network model
        dataset: Calibration dataset
        popsize: DE population size (multiplier for number of parameters)
        maxiter: Maximum iterations
        strategy: DE strategy ('best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', etc.)
        mutation: Mutation constant or range (min, max)
        recombination: Crossover probability [0, 1]
        seed: Random seed for reproducibility
        polish: Whether to use L-BFGS-B to polish the best result
        atol: Absolute tolerance for convergence

    Returns:
        Calibrated parameters as numpy array
        History of best fitness values
    """

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False  # freeze network

    # Precompute tensors
    X_all = dataset.X.to(device)  # M x num_features
    sigma_all = dataset.sigma.to(device)  # M
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"Starting SciPy DE calibration: {M} options, {N} returns")
    print(f"Population size: {popsize}, Max iterations: {maxiter}")
    print(f"Strategy: {strategy}, Mutation: {mutation}, Recombination: {recombination}")

    # Initialize parameters using HN GARCH
    hn_model = HestonNandiGARCH(all_returns.cpu().numpy())
    _ = hn_model.fit()
    initial_guess = hn_model.fitted_params

    print(f"Initial HN GARCH params: {initial_guess}")

    # Setup optional L2 regularization on parameters (centered on HN or true params)
    reg_center_np = None
    gamma_center_val = None
    lambda_center_val = None
    try:
        if reg_weight is not None and float(reg_weight) > 0.0:
            if isinstance(reg_center, (list, tuple, np.ndarray)):
                reg_center_np = np.asarray(reg_center, dtype=float)
            elif isinstance(reg_center, str) and reg_center.lower() == "true" and true_vals is not None:
                reg_center_np = np.asarray(true_vals, dtype=float)
            else:
                # default 'hn' or fallback
                reg_center_np = np.asarray(initial_guess, dtype=float)
            print(f"Regularization enabled: weight={reg_weight}, center={reg_center}")
    except Exception:
        reg_center_np = None
    # Gamma-specific center resolution
    try:
        if gamma_reg_weight is not None and float(gamma_reg_weight) > 0.0:
            if isinstance(gamma_reg_center, (int, float)):
                gamma_center_val = float(gamma_reg_center)
            elif isinstance(gamma_reg_center, str) and gamma_reg_center.lower() == "true" and true_vals is not None:
                gamma_center_val = float(true_vals[3])
            else:
                # default to HN initial guess gamma (projected)
                gamma_center_val = float(initial_proj[3])
            print(f"Gamma regularization enabled: weight={gamma_reg_weight}, center={gamma_reg_center}, center_val={gamma_center_val}")
    except Exception:
        gamma_center_val = None
    # Lambda-specific center resolution
    try:
        if lambda_reg_weight is not None and float(lambda_reg_weight) > 0.0:
            if isinstance(lambda_reg_center, (int, float)):
                lambda_center_val = float(lambda_reg_center)
            elif isinstance(lambda_reg_center, str) and lambda_reg_center.lower() == "true" and true_vals is not None:
                lambda_center_val = float(true_vals[4])
            else:
                # default to HN initial guess lambda (projected)
                lambda_center_val = float(initial_proj[4])
            print(f"Lambda regularization enabled: weight={lambda_reg_weight}, center={lambda_reg_center}, center_val={lambda_center_val}")
    except Exception:
        lambda_center_val = None
    # Define parameter bounds for SciPy DE
    # [omega, alpha, beta, gamma, lambda]
    # Bounds tightened to avoid exploring extreme regions that cause numerical instability,
    # while still providing reasonable room around the expected true parameters.
    bounds = [
        (1e-10, 1e-5),  # omega: positive, small
        (1e-12, 1e-5),  # alpha: very small to small (allow near-zero alpha)
        (0.2, 0.999),   # beta: close to 1
        (0, 30),        # gamma: leverage effect (tighter)
        (0, 2.0),       # lambda: risk premium (tighter)
    ]

    print(f"Parameter bounds: {bounds}")
    # Precompute and store a clipped projection of the HN initial guess so it is
    # always available for regularization or local searches even if later
    # population insertion fails for some reason.
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    initial_proj = np.clip(initial_guess, lower, upper).astype(float).reshape(-1)

    # Build an explicit initial population so that the HN initial guess is present
    n_params = len(bounds)
    npop = int(popsize * n_params)
    if npop < 4:
        npop = 4
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    # Random initialization within bounds
    init_pop = np.random.rand(npop, n_params) * (upper - lower) + lower

    # Try to project and insert the HN initial guess as the first individual
    try:
        # Ensure a numeric 1D numpy array of the expected length
        initial_proj = np.clip(initial_guess, lower, upper)
        initial_proj = np.asarray(initial_proj, dtype=float).reshape((n_params,))

        # Insert the exact projected initial guess as the first individual
        init_pop[0, :] = initial_proj.copy()

        # Add a few small local perturbations around the initial guess so the population
        # contains nearby candidates and DE can explore the local neighborhood effectively.
        # The perturbations are relative (~5% noise) with a floor for near-zero params,
        # and are clipped to the global bounds.
        n_local = min(4, npop - 1)
        for k in range(n_local):
            # relative Gaussian perturbation (~5%)
            noise = np.random.randn(n_params) * 0.05
            perturb = initial_proj * (1.0 + noise)

            # For parameters that are exactly zero, generate per-index small absolute perturbations
            zero_idx = np.where(initial_proj == 0.0)[0]
            if zero_idx.size > 0:
                small_abs = (upper - lower) * 0.01 * np.random.rand(zero_idx.size)
                # assign per-index to avoid boolean-indexing shape pitfalls
                for j, idx in enumerate(zero_idx):
                    perturb[idx] = lower[idx] + small_abs[j]

            # clip to bounds and insert (reserve index 0 for the exact initial guess)
            init_pop[1 + k, :] = np.clip(perturb, lower, upper)

        print(f"Inserted initial_guess into initial population (index 0) and {n_local} local perturbations.")
    except Exception as e:
        print(f"Warning: could not insert initial_guess into initial population: {e}")
        # Emit debugging info to diagnose shape/type problems
        try:
            print("  initial_guess type:", type(initial_guess), "shape:", getattr(initial_guess, "shape", None))
            if "initial_proj" in locals():
                print("  initial_proj type:", type(initial_proj), "shape:", getattr(initial_proj, "shape", None))
            else:
                print("  initial_proj: <not defined>")
            print("  init_pop shape:", getattr(init_pop, "shape", None))
            print("  lower shape:", getattr(lower, "shape", None), "upper shape:", getattr(upper, "shape", None))
            print("  n_params:", n_params)
        except Exception:
            # best-effort logging; don't crash
            pass

    # Precompute a small sample and CPU model for diagnostics (avoids GPU OOM in objective)
    sample_size = min(500, M)
    sample_idx = np.linspace(0, M - 1, num=sample_size, dtype=int)
    X_sample_cpu = X_all.cpu()[sample_idx]
    sigma_sample_np = sigma_all.cpu().numpy()[sample_idx]
    try:
        model_cpu = torch.load(MODEL_PATH, map_location="cpu")
        model_cpu.eval()
    except Exception:
        model_cpu = None

    # Precompute sigma for the HN-projected initial parameters on the sample, for sigma-surface regularization
    sigma_init_sample = None
    if model_cpu is not None:
        try:
            p_cpu_init = torch.tensor(initial_proj, dtype=torch.float32)
            omega_p_i, alpha_p_i, beta_p_i, gamma_p_i, lambda_p_i = p_cpu_init
            x_init_s = X_sample_cpu.clone()
            x_init_s[:, 5] = alpha_p_i
            x_init_s[:, 6] = beta_p_i
            x_init_s[:, 7] = omega_p_i
            x_init_s[:, 8] = gamma_p_i
            x_init_s[:, 9] = lambda_p_i
            eps = 1e-8
            x_init_s[:, 19] = torch.log(gamma_p_i + eps)
            x_init_s[:, 20] = torch.log(omega_p_i + eps)
            x_init_s[:, 21] = torch.log(lambda_p_i + eps)
            x_init_s[:, 22] = torch.log(alpha_p_i + eps)
            x_init_s[:, 23] = torch.log(beta_p_i + eps)
            with torch.no_grad():
                sigma_init_sample = model_cpu(x_init_s).squeeze().cpu().numpy()
        except Exception:
            sigma_init_sample = None

    # Track convergence history
    convergence_history = []
    iteration_count = [0]  # Use list to make it mutable in callback
    # Cache holder for the latest regularization term so the callback can display it.
    # It is a list so we can mutate its single element from `objective_function`.
    _LAST_REG_CACHE = [0.0]
    explosion_count = [0]  # counts how many times the objective returned a penalty due to explosion/invalid params
    clamp_count = [0]  # counts how many times parameters were clamped to bounds

    def objective_function(params):
        """
        Objective function for SciPy DE optimization

        Args:
            params: numpy array of parameters [omega, alpha, beta, gamma, lambda]

        Returns:
            float: Loss value
        """
        # Convert to torch tensor and project parameters
        params_tensor = torch.tensor(params, device=device, dtype=torch.float32)
        params_proj = project_parameters(params_tensor)

        # Clamp projected parameters into the global bounds to avoid gamma/lambda excursions.
        # Keep a counter of clamp events and print a concise debug line when clipping occurs.
        try:
            params_np = params_proj.cpu().numpy()
            params_clipped = np.clip(params_np, lower, upper)
            if not np.allclose(params_np, params_clipped):
                clamp_count[0] += 1
                clipped_idx = np.where(~np.isclose(params_np, params_clipped))[0]
                names = ["omega", "alpha", "beta", "gamma", "lambda"]
                clipped_info = ", ".join(
                    f"{names[i]}:{params_np[i]:.3e}->{params_clipped[i]:.3e}" for i in clipped_idx
                )
                print(f"DEBUG: Clamped parameters to bounds: {clipped_info}")
            params_proj = torch.tensor(params_clipped, device=device, dtype=torch.float32)
        except Exception:
            # If clamping fails for any reason, fall back to the projected tensor
            pass

        # QUICK GUARD: avoid exploring parameter regions that violate stationarity
        # or imply an unreasonable unconditional variance which leads to recursion
        # explosions in the returns likelihood.
        try:
            omega_p = float(params_proj[0].cpu().item())
            alpha_p = float(params_proj[1].cpu().item())
            beta_p = float(params_proj[2].cpu().item())
            gamma_p = float(params_proj[3].cpu().item())

            # Use the projected alpha/beta values to compute persistence (bugfix)
            persistence = beta_p + alpha_p * (gamma_p ** 2)
            persistence_cut = 0.999
            if persistence >= persistence_cut:
                # print the persistence warning only once to avoid spam
                if explosion_count[0] == 0:
                    print(f"WARNING: persistence ({persistence:.6f}) >= {persistence_cut} — returning penalty")
                explosion_count[0] += 1
                return 1e9

            # approximate unconditional variance (using projected params)
            denom = 1.0 - persistence
            if denom <= 1e-8:
                h_unc = float('inf')
            else:
                h_unc = (omega_p + alpha_p) / denom

            if h_unc > 1.0:
                # print the unconditional-variance warning only once to reduce spam
                if explosion_count[0] == 0:
                    print(f"WARNING: unconditional variance ({h_unc:.6e}) > 1.0 — returning penalty")
                explosion_count[0] += 1
                return 1e9
        except Exception:
            # If any numeric extraction fails, continue — the later checks will catch issues
            pass

        # Compute calibration loss (this runs the full forward pass)
        with torch.no_grad():
            loss = Calibration_Loss(
                params_proj, all_returns, sigma_all, model, X_all, N, M
            )
            loss_val = loss.cpu().item()

            # Compute return log-likelihood for diagnostics (full)
            try:
                r_daily = X_all[:, 2].mean()
                lr = ll_returns_torch(all_returns, params_proj, r_daily)
                mean_lr = (lr / N).item()
            except Exception:
                lr = None
                mean_lr = None

            # Compute option diagnostics on small sample using CPU model (to avoid GPU spikes)
            mean_lo_sample = None
            lo_sample = None
            try:
                if model_cpu is not None:
                    # move parameters to CPU for sample prediction
                    p_cpu = params_proj.cpu()
                    omega_p, alpha_p, beta_p, gamma_p, lambda_p = p_cpu

                    x_s = X_sample_cpu.clone()
                    x_s[:, 5] = alpha_p
                    x_s[:, 6] = beta_p
                    x_s[:, 7] = omega_p
                    x_s[:, 8] = gamma_p
                    x_s[:, 9] = lambda_p

                    eps = 1e-8
                    x_s[:, 19] = torch.log(gamma_p + eps)
                    x_s[:, 20] = torch.log(omega_p + eps)
                    x_s[:, 21] = torch.log(lambda_p + eps)
                    x_s[:, 22] = torch.log(alpha_p + eps)
                    x_s[:, 23] = torch.log(beta_p + eps)

                    sigma_cand = model_cpu(x_s).squeeze().cpu()
                    lo_sample = ll_option_torch(torch.tensor(sigma_sample_np), sigma_cand, sigma_eps=0.01)
                    mean_lo_sample = (lo_sample / sigma_cand.shape[0]).item()
            except Exception:
                mean_lo_sample = None
                lo_sample = None

        # Treat extremely poor returns likelihoods as numerical explosions.
        # If the returns log-likelihood is extremely negative, this indicates
        # the parameter set leads to catastrophic mismatch (or recursion explosion),
        # so return a very large penalty immediately to steer DE away.
        try:
            if lr is not None:
                try:
                    lr_val = lr.cpu().item() if isinstance(lr, torch.Tensor) else float(lr)
                except Exception:
                    lr_val = float(lr)
                if lr_val <= -1e8:
                    if explosion_count[0] < 5:
                        print("====== OBJECTIVE EXPLOSION: EXTREMELY NEGATIVE RETURNS LL ======")
                        print(f"  Candidate params: {params}")
                        print(f"  lr (total): {lr_val:.6e} <= -1e8 -> Returning penalty")
                    explosion_count[0] += 1
                    return 1e9
        except Exception:
            # If anything goes wrong here, fall through to standard checks below
            pass

        # Add optional L2 regularization: full-vector reg + gamma/lambda-specific terms.
        reg_val = 0.0
        try:
            if reg_weight is not None and float(reg_weight) > 0.0 and reg_center_np is not None:
                cand_vec = params_proj.cpu().numpy()
                l2_c = np.linalg.norm(cand_vec - reg_center_np)
                reg_term = float(reg_weight * (l2_c ** 2))
                reg_val += reg_term
                loss_val = float(loss_val + reg_term)
        except Exception:
            # ignore regularizer failures (keep original loss)
            pass

        try:
            if gamma_reg_weight is not None and float(gamma_reg_weight) > 0.0 and gamma_center_val is not None:
                gamma_val = float(params_proj[3].cpu().item())
                gamma_term = float(gamma_reg_weight * ((gamma_val - gamma_center_val) ** 2))
                reg_val += gamma_term
                loss_val = float(loss_val + gamma_term)
        except Exception:
            pass

        try:
            if lambda_reg_weight is not None and float(lambda_reg_weight) > 0.0 and lambda_center_val is not None:
                lambda_val = float(params_proj[4].cpu().item())
                lambda_term = float(lambda_reg_weight * ((lambda_val - lambda_center_val) ** 2))
                reg_val += lambda_term
                loss_val = float(loss_val + lambda_term)
        except Exception:
            pass

        # Sigma-surface regularization: penalize deviation of candidate-predicted implied vol
        # surface (on the CPU sample) from the initial-projected sigma surface, if configured.
        try:
            if sigma_reg_weight is not None and float(sigma_reg_weight) > 0.0 and (sigma_init_sample is not None) and (model_cpu is not None):
                # compute candidate sigma on the same CPU sample
                p_cpu = params_proj.cpu()
                omega_p_s, alpha_p_s, beta_p_s, gamma_p_s, lambda_p_s = p_cpu
                x_s2 = X_sample_cpu.clone()
                x_s2[:, 5] = alpha_p_s
                x_s2[:, 6] = beta_p_s
                x_s2[:, 7] = omega_p_s
                x_s2[:, 8] = gamma_p_s
                x_s2[:, 9] = lambda_p_s
                eps = 1e-8
                x_s2[:, 19] = torch.log(gamma_p_s + eps)
                x_s2[:, 20] = torch.log(omega_p_s + eps)
                x_s2[:, 21] = torch.log(lambda_p_s + eps)
                x_s2[:, 22] = torch.log(alpha_p_s + eps)
                x_s2[:, 23] = torch.log(beta_p_s + eps)
                with torch.no_grad():
                    sigma_cand = model_cpu(x_s2).squeeze().cpu().numpy()
                sigma_l2 = float(np.linalg.norm(sigma_cand - sigma_init_sample))
                sigma_reg_term = float(sigma_reg_weight * (sigma_l2 ** 2))
                reg_val += sigma_reg_term
                loss_val = float(loss_val + sigma_reg_term)
        except Exception:
            pass

        # Cache the regularization contribution for the callback to display
        _LAST_REG_CACHE[0] = reg_val

        return loss_val
        # Check for numerical explosion or huge loss and print diagnostics
        exploded = False
        try:
            if np.isnan(loss_val) or np.isinf(loss_val):
                exploded = True
            elif loss_val > 1e6:
                exploded = True
            else:
                # Optionally compare to initial_loss (if available in outer scope)
                try:
                    if 'initial_loss' in globals() and initial_loss is not None and loss_val > (1000.0 * initial_loss):
                        exploded = True
                except Exception:
                    pass
        except Exception:
            exploded = True

        if exploded:
            if explosion_count[0] < 5:
                print("====== OBJECTIVE EXPLOSION DIAGNOSTICS ======")
                print(f"  Candidate params: {params}")
                print(f"  loss: {loss_val:.6e}")
                if lr is not None:
                    try:
                        lr_val = lr.cpu().item()
                        print(f"  LR (total): {lr_val:.6e}, mean LR: {mean_lr}")
                    except Exception:
                        print(f"  LR: {lr}")
                if lo_sample is not None:
                    try:
                        lo_val = lo_sample.cpu().item()
                        print(f"  LO_sample (total): {lo_val:.6e}, mean LO_sample: {mean_lo_sample}")
                    except Exception:
                        print(f"  LO_sample: {lo_sample}")
                # show regularizer in diagnostic explosion messages too
                if reg_val is not None and reg_val > 0:
                    print(f"  Reg contribution: {reg_val:.6e}")
                print("  Returning large penalty to guide DE away from this region.")
            explosion_count[0] += 1
            return 1e9

        return loss_val

    def callback(xk, convergence=None):
        """Callback function to track convergence"""
        iteration_count[0] += 1
        current_loss = objective_function(xk)
        convergence_history.append(current_loss)

        error_str = ""
        if true_vals is not None:
            l2_error = np.linalg.norm(xk - true_vals, ord=2)
            error_str = f" | Error: {l2_error:.6f}"

        # Display the last regularization contribution (if any) to help diagnose biasing effects
        reg_val_hook = _LAST_REG_CACHE[0] if len(_LAST_REG_CACHE) > 0 else 0.0
        reg_str = f" | Reg: {reg_val_hook:.6f}" if reg_val_hook and reg_val_hook > 0 else ""
        print(
            f"Iter {iteration_count[0]:4d} | Loss: {current_loss:.6f}{error_str}{reg_str} | "
            f"Params: ω={xk[0]:.2e}, α={xk[1]:.2e}, β={xk[2]:.4f}, γ={xk[3]:.2f}, λ={xk[4]:.2f}"
        )

    print("Starting SciPy DE optimization...")

    # Check initial guess loss
    initial_loss = objective_function(initial_guess)
    print(f"Initial Guess Loss: {initial_loss:.6f}")

    # Optional debugging: evaluate initial population losses (may be slow)
    if debug_init_pop:
        print("DEBUG: Evaluating initial population losses (this may be slow)...")
        try:
            init_losses = []
            for i in range(init_pop.shape[0]):
                try:
                    loss_i = objective_function(init_pop[i])
                except Exception as err:
                    # If a particular candidate blows up, record NaN and continue
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

            try:
                min_idx = int(np.nanargmin(init_losses))
                best_loss = init_losses[min_idx]
                print(f"DEBUG: Best init member index: {min_idx}, loss: {best_loss:.6f}")
                if true_vals is not None:
                    l2_best = np.linalg.norm(init_pop[min_idx] - true_vals)
                    print(f"DEBUG: L2 error of best init member to true params: {l2_best:.6f}")
            except ValueError:
                print("DEBUG: All initial population losses are NaN, cannot identify best member.")
        except Exception as e:
            print(f"DEBUG: Failed to evaluate initial population: {e}")

    # Run SciPy differential evolution
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=atol,
        mutation=mutation,
        recombination=recombination,
        callback=callback,
        polish=polish,
        init=init_pop,
        # x0=initial_guess,  # initial guess inserted into `init_pop`
        disp=False,  # We handle progress in callback
    )

    # Extract results
    best_params = result.x
    best_fitness = result.fun

    # Final projection
    best_params = project_parameters(best_params)

    print(f"\n✅ SciPy DE Calibration complete")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final loss: {best_fitness:.6f}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Explosions encountered (penalty returns): {explosion_count[0]}")
    print(f"Parameter clamp events: {clamp_count[0]}")

    # Optional local refinement (L-BFGS-B) to enforce returns fit.
    # This runs two refinements: one from the HN initial guess and one from the DE best.
    if local_refine:
        print("\n🔧 Starting local L-BFGS-B refinement (two starts: HN initial and DE best)")
        # Objective that uses ll_returns_torch and ll_option_torch and a specified returns weight.
        def refine_objective_np(x_np, w_ret):
            # Convert and clamp/project to valid params
            params_t = torch.tensor(x_np, device=device, dtype=torch.float32)
            params_proj = project_parameters(params_t)

            with torch.no_grad():
                # Inject parameters into the full X_all and compute model sigmas
                x_in = X_all.clone()
                omega_p, alpha_p, beta_p, gamma_p, lambda_p = params_proj
                eps = 1e-8
                x_in[:, 5] = alpha_p
                x_in[:, 6] = beta_p
                x_in[:, 7] = omega_p
                x_in[:, 8] = gamma_p
                x_in[:, 9] = lambda_p

                x_in[:, 19] = torch.log(gamma_p + eps)
                x_in[:, 20] = torch.log(omega_p + eps)
                x_in[:, 21] = torch.log(lambda_p + eps)
                x_in[:, 22] = torch.log(alpha_p + eps)
                x_in[:, 23] = torch.log(beta_p + eps)

                sigma_model = model(x_in).squeeze()
                lo = ll_option_torch(sigma_all, sigma_model, sigma_eps=0.01)
                lr = ll_returns_torch(all_returns, params_proj, X_all[:, 2].mean())

                mean_lr = lr / N
                mean_lo = lo / M

                joint = w_ret * mean_lr + mean_lo
                # Return negative joint (minimization)
                return float(-joint.cpu().item())

        # Prepare starts: initial_guess (HN) and DE best (projected best_params)
        starts = []
        try:
            hn_start = np.asarray(initial_guess, dtype=float)
            starts.append(("hn", hn_start))
        except Exception:
            pass
        try:
            de_start = np.asarray(best_params, dtype=float)
            starts.append(("de", de_start))
        except Exception:
            pass

        polished = []
        for name, x0 in starts:
            print(f"  -> Refining start '{name}' with L-BFGS-B")
            res = minimize(lambda x: refine_objective_np(x, refine_w_ret), x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": refine_maxiter, "ftol": refine_tol})
            try:
                # project the found parameters for safety
                polished_params_proj = project_parameters(torch.tensor(res.x, device=device, dtype=torch.float32))
            except Exception:
                polished_params_proj = res.x
            polished_loss_refine = float(res.fun)
            # For reporting, compute the original calibration loss (Calibration_Loss) for the polished candidate
            try:
                with torch.no_grad():
                    res_params_t = torch.tensor(res.x, device=device, dtype=torch.float32)
                    de_loss_val = Calibration_Loss(res_params_t, all_returns, sigma_all, model, X_all, N, M).cpu().item()
            except Exception:
                de_loss_val = float("nan")

            print(f"    start={name}, polished_refine_loss={polished_loss_refine:.6f}, polished_de_loss={de_loss_val:.6f}")
            polished.append((name, polished_params_proj, polished_loss_refine, de_loss_val, res.x))

        # Evaluate whether any polished candidate improves the refine objective relative to the DE best (measured by refine objective)
        base_refine_loss = refine_objective_np(np.asarray(best_params, dtype=float), refine_w_ret)
        best_polished = min(polished, key=lambda x: x[2])
        if best_polished[2] < base_refine_loss:
            print(f"  ✅ Polishing improved the refine objective: {base_refine_loss:.6f} -> {best_polished[2]:.6f}")
            # Adopt polished params (projected) as final
            best_params = best_polished[1]
            # For reporting, use polished DE loss as the 'final' fitness if available (else recompute)
            try:
                best_fitness = best_polished[3] if not np.isnan(best_polished[3]) else float(refine_objective_np(np.asarray(best_params, dtype=float), refine_w_ret))
            except Exception:
                best_fitness = float(refine_objective_np(np.asarray(best_params, dtype=float), refine_w_ret))
        else:
            print("  ℹ️ Polishing did not improve the refine objective; keeping DE result.")

    return best_params, convergence_history, initial_guess


# def main():
#     """Main function for SciPy DE calibration"""
#     try:
#         print("🧬 SciPy Differential Evolution GARCH Parameter Calibration")
#         print("=" * 65)

#         print("Loading model...")
#         model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
#         print("Model loaded successfully.")

#         print("Loading dataset...")
#         dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
#                               "joint_dataset/assetprices.csv")
#         print(f"Dataset: {len(dataset)} options, returns length {len(dataset.returns)}")

#         # Run SciPy DE calibration with different strategies
#         strategies_to_test = ['best1bin', 'best2bin', 'rand1bin']

#         for strategy in strategies_to_test:
#             print(f"\n{'='*50}")
#             print(f"Testing strategy: {strategy}")
#             print(f"{'='*50}")

#             calibrated_params, convergence_history = calibrate_scipy_de(
#                 model=model,
#                 dataset=dataset,
#                 popsize=15,               # Population size multiplier (15 * 5 params = 75)
#                 maxiter=500,              # Maximum iterations
#                 strategy=strategy,        # DE strategy
#                 mutation=(0.5, 1.0),     # Mutation range
#                 recombination=0.7,       # Crossover probability
#                 seed=42,                 # For reproducibility
#                 polish=True,             # Use L-BFGS-B polish for final refinement
#                 atol=1e-6                # Absolute tolerance
#             )

#             # Compare with true values
#             true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
#             pred_vals = calibrated_params

#             # Calculate error metrics
#             l2_error = np.linalg.norm(pred_vals - true_vals, ord=2)
#             relative_errors = np.abs((pred_vals - true_vals) / true_vals) * 100

#             print(f"\nCalibration Results for {strategy}:")
#             print(f"L2 Error: {l2_error:.6f}")

#             # Parameter comparison table
#             param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
#             print(f"\n{'Parameter':<10} {'Calibrated':<12} {'True':<12} {'Error':<12} {'Rel Error %':<12}")
#             print("-" * 60)
#             for i, name in enumerate(param_names):
#                 print(f"{name:<10} {calibrated_params[i]:<12.6f} {true_vals[i]:<12.6f} "
#                       f"{abs(calibrated_params[i] - true_vals[i]):<12.6f} {relative_errors[i]:<12.2f}")

#             # Save results
#             results = {
#                 'method': 'scipy_differential_evolution',
#                 'strategy': strategy,
#                 'calibrated_parameters': dict(zip(param_names, calibrated_params.tolist())),
#                 'true_parameters': dict(zip(param_names, true_vals.tolist())),
#                 'convergence_history': convergence_history,
#                 'final_loss': convergence_history[-1] if convergence_history else None,
#                 'errors': {
#                     'l2_error': float(l2_error),
#                     'relative_errors': dict(zip(param_names, relative_errors.tolist())),
#                     'absolute_errors': dict(zip(param_names, np.abs(calibrated_params - true_vals).tolist()))
#                 }
#             }

#             filename = f'calibrated_params_scipy_de_{strategy}.json'
#             with open(filename, 'w') as f:
#                 json.dump(results, f, indent=2)

#             print(f"\n📁 Results saved to {filename}")

#             # Parameter validation
#             alpha, beta = calibrated_params[1], calibrated_params[2]
#             persistence = alpha + beta
#             print(f"\nParameter Validation:")
#             print(f"Persistence (α+β): {persistence:.6f}")

#             if persistence < 1.0:
#                 print("✅ Stationarity condition satisfied")
#                 unconditional_var = calibrated_params[0] / (1 - persistence)
#                 print(f"Theoretical unconditional variance: {unconditional_var:.8f}")
#             else:
#                 print("⚠️  Stationarity condition violated")

#             if calibrated_params[0] > 0:
#                 print("✅ Omega is positive")
#             else:
#                 print("⚠️  Omega is not positive")

#         # Compare with empirical variance
#         empirical_var = dataset.returns.var().item()
#         print(f"\nEmpirical returns variance: {empirical_var:.8f}")

#         print(f"\n🎉 SciPy DE calibration complete for all strategies!")

#         return calibrated_params, convergence_history

#     except Exception as e:
#         print(f"❌ SciPy DE Calibration failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None, None


# if __name__ == "__main__":
#     main()
