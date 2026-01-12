#!/usr/bin/env python3
"""
inspect_params.py

Diagnostic script to inspect candidate GARCH parameter sets and identify
why some parameter vectors lead to numerical/explosive behaviour during
Differential Evolution. The script runs on CPU and performs:

- Parameter projection (same rules as the calibrator).
- A GARCH variance recursion (h_t) using the same update as in the loss.
  It tracks min/max/mean of the variance path, counts clamping events,
  detects NaN/Inf/overflow and reports any 'explosion'.
- Computes returns log-likelihood (LL) based on the simulated h_t and z_t.
- Computes option MSE and option log-likelihood using the stored model.
- Compares predicted implied vol surface (sigma_model) to the true/initial
  sigma predictions (L2).
- Optionally evaluates the initial DE population and reports how many
  initial members already cause extremely large losses.

Usage examples:
    python jointcal/inspect_params.py -c 2.77e-06,4.47e-06,0.4179,198.87,9.63
    python jointcal/inspect_params.py --check-init-pop --popsize 5
    python jointcal/inspect_params.py --candidate-file my_candidates.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Project imports (assumes script run from repo root)
from cal_loss import Calibration_Loss, ll_option_torch, ll_returns_torch
from dataset2 import cal_dataset
from hn import HestonNandiGARCH

# We rely on the same projection logic as the calibrator:
try:
    from calibrate_scipy_de import project_parameters
except Exception:
    # If calibrator import fails for some reason, we implement a fallback projection.
    def project_parameters(params):
        # params: torch.Tensor or numpy array
        if isinstance(params, torch.Tensor):
            omega = torch.clamp(params[0], min=1e-9)
            alpha = torch.clamp(params[1], min=0.0, max=1.0 - 1e-12)
            beta = torch.clamp(params[2], min=0.0, max=1.0 - 1e-12)
            gamma = torch.clamp(params[3], min=0.0)
            lambda_param = torch.clamp(params[4], min=0.0)
            return torch.stack([omega, alpha, beta, gamma, lambda_param])
        else:
            omega = float(max(params[0], 1e-9))
            alpha = float(np.clip(params[1], 0.0, 1.0 - 1e-12))
            beta = float(np.clip(params[2], 0.0, 1.0 - 1e-12))
            gamma = float(max(params[3], 0.0))
            lambda_param = float(max(params[4], 0.0))
            return np.array([omega, alpha, beta, gamma, lambda_param])


def safe_torch_load_cpu(path: str, permissive: bool = True) -> torch.nn.Module:
    """Load a model onto CPU, trying a few safe fallbacks."""
    try:
        # Try full load first (may raise on new torch if weights_only default changed).
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions may not accept weights_only kwarg
        return torch.load(path, map_location="cpu")
    except Exception as e:
        if permissive:
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                raise e
        raise e


def parse_candidate_string(s: str) -> np.ndarray:
    """Parse comma-separated 5 floats into a numpy array."""
    arr = np.fromstring(s, sep=",")
    if arr.size != 5:
        raise ValueError(f"Expected 5 values, got {arr.size}: {s}")
    return arr.astype(float)


def simulate_h_path(
    params: np.ndarray,
    returns: np.ndarray,
    r_daily: float = 0.0,
    eps: float = 1e-12,
    max_h_threshold: float = 1e9,
) -> Dict[str, Any]:
    """
    Simulate h_t and z_t iteratively using Heston-Nandi GARCH recursion.

    Returns a dict with:
      - h: numpy array of length T (variances used to standardize returns)
      - z: numpy array of length T (standardized residuals)
      - stats: min,max,mean,last,counts for clamps and anomalies
      - exploded: bool (True if early termination due to explosion)
    """
    omega, alpha, beta, gamma, lambda_ = [float(x) for x in params]
    T = returns.shape[0]
    h = np.zeros(T, dtype=float)
    z = np.zeros(T, dtype=float)

    # Initial unconditional variance if stationary
    persistence = beta + alpha * (gamma**2)
    if persistence < 0.999:
        h_curr = float((omega + alpha) / (1.0 - persistence + 1e-8))
    else:
        # fallback to empirical sample variance
        h_curr = float(np.var(returns))

    if not np.isfinite(h_curr) or h_curr <= 0 or h_curr < 1e-12:
        # ensure positive reasonable starting variance
        h_curr = 1e-6

    exploded = False
    clamp_count_small = 0
    clamp_count_large = 0
    nan_count = 0

    for t in range(T):
        # Pre-checks
        if not np.isfinite(h_curr) or h_curr <= 0 or h_curr > max_h_threshold:
            exploded = True
            break

        h[t] = h_curr
        sqrt_h = math.sqrt(h_curr) if h_curr > 0 else 1e-8

        # Compute standardized residual
        numerator = returns[t] - r_daily - lambda_ * h_curr
        denom = sqrt_h + 1e-12
        z_t = numerator / denom
        z[t] = z_t

        # Update variance for next step using corrected formula:
        # h_{t+1} = omega + beta * h_t + alpha * (z_t - gamma * sqrt(h_t))^2
        next_h = omega + beta * h_curr + alpha * (z_t - gamma * sqrt_h) ** 2

        # Stability checks and clamping
        if not np.isfinite(next_h):
            nan_count += 1
            exploded = True
            break
        if next_h < 1e-12:
            clamp_count_small += 1
            next_h = 1e-12
        if next_h > max_h_threshold:
            clamp_count_large += 1
            # keep it but note explosion; next iteration will detect it
        h_curr = float(next_h)

    stats = {
        "h_min": float(np.min(h[: t + 1]) if t >= 0 else np.nan),
        "h_max": float(np.max(h[: t + 1]) if t >= 0 else np.nan),
        "h_mean": float(np.mean(h[: t + 1]) if t >= 0 else np.nan),
        "h_last": float(h[t] if t >= 0 else np.nan),
        "clamp_small": int(clamp_count_small),
        "clamp_large": int(clamp_count_large),
        "nan_count": int(nan_count),
        "steps_simulated": int(t + 1),
        "exploded": bool(exploded),
    }

    return {"h": h[: t + 1].copy(), "z": z[: t + 1].copy(), "stats": stats}


def compute_returns_ll_from_sim(h: np.ndarray, z: np.ndarray) -> float:
    """
    Compute returns log-likelihood given vectors h and z:
      LL = -0.5 * sum( log(2*pi) + log(h_t) + z_t^2 )
    """
    eps = 1e-12
    T = h.shape[0]
    if T == 0:
        return float(-1e12)
    # Ensure non-negative
    h_safe = np.clip(h, 1e-12, None)
    val = -0.5 * np.sum(np.log(2.0 * math.pi) + np.log(h_safe + eps) + (z ** 2))
    return float(val)


def inject_params_into_X(X: torch.Tensor, params: np.ndarray) -> torch.Tensor:
    """
    Clone X and inject the given params into the feature indices expected
    by the NN model: alpha (5), beta (6), omega (7), gamma (8), lambda (9)
    and the engineered log features 19-23.
    """
    x_in = X.clone()
    eps = 1e-8
    omega, alpha, beta, gamma, lambda_ = [float(x) for x in params]
    # Base
    x_in[:, 5] = float(alpha)
    x_in[:, 6] = float(beta)
    x_in[:, 7] = float(omega)
    x_in[:, 8] = float(gamma)
    x_in[:, 9] = float(lambda_)
    # Logs
    x_in[:, 19] = float(math.log(gamma + eps))
    x_in[:, 20] = float(math.log(omega + eps))
    x_in[:, 21] = float(math.log(lambda_ + eps))
    x_in[:, 22] = float(math.log(alpha + eps))
    x_in[:, 23] = float(math.log(beta + eps))
    return x_in


def evaluate_candidate(
    name: str,
    params: np.ndarray,
    model: torch.nn.Module,
    dataset,
    sigma_eps: float = 0.01,
    verbose: bool = False,
    true_params: Optional[np.ndarray] = None,
    initial_params: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate and report diagnostics for a single parameter vector.
    Returns a dict of metrics.
    """
    device = torch.device("cpu")
    # Ensure CPU tensors
    X = dataset.X.to(device)
    sigma_obs = dataset.sigma.to(device)
    returns = dataset.returns.cpu().numpy().flatten()
    N = len(returns)
    M = len(dataset)
    # r_daily same as in Calibration_Loss: mean of column index 2
    r_daily = float(X[:, 2].mean().item())

    # Raw vs Projected params (project using calibrator rules)
    try:
        params_proj = project_parameters(params)  # may accept numpy or torch
        if isinstance(params_proj, torch.Tensor):
            params_proj = params_proj.cpu().numpy().astype(float)
    except Exception:
        # fallback simple projection
        params_proj = np.asarray(params, dtype=float)
        params_proj = np.clip(
            params_proj, [1e-10, 1e-12, 0.0, 0.0, 0.0], [1e-5, 1e-5, 0.999, 100.0, 5.0]
        )

    # 1) Simulate h-path with raw and projected params to compare
    sim_raw = simulate_h_path(params, returns, r_daily=r_daily)
    sim_proj = simulate_h_path(params_proj, returns, r_daily=r_daily)

    # 2) Compute LL from simulation
    lr_raw = compute_returns_ll_from_sim(sim_raw["h"], sim_raw["z"])
    lr_proj = compute_returns_ll_from_sim(sim_proj["h"], sim_proj["z"])

    # 3) Compare with ll_returns_torch (projected params path)
    try:
        params_torch = torch.tensor(params_proj, device=device, dtype=torch.float32)
        lr_torch = ll_returns_torch(dataset.returns.to(device), params_torch, r_daily)
        lr_torch_val = float(lr_torch.cpu().item())
    except Exception:
        lr_torch_val = None

    # 4) Predict sigma_model (options) using injected features
    with torch.no_grad():
        x_in = inject_params_into_X(X, params_proj)
        sigma_pred = model(x_in).squeeze().cpu()
        opt_mse = float(torch.nn.MSELoss()(sigma_obs, sigma_pred).item())
        opt_lo = float(ll_option_torch(sigma_obs, sigma_pred, sigma_eps=sigma_eps).cpu().item())
        opt_lo_mean = opt_lo / float(M)

    # 5) Precompute true/initial sigma predictions if available (for L2 comparisons)
    sigma_true = None
    sigma_init = None
    if true_params is not None:
        with torch.no_grad():
            sigma_true = model(inject_params_into_X(X, true_params)).squeeze().cpu().numpy()
    if initial_params is not None:
        with torch.no_grad():
            sigma_init = model(inject_params_into_X(X, initial_params)).squeeze().cpu().numpy()

    sigma_l2_to_true = None
    sigma_l2_to_init = None
    if sigma_true is not None:
        sigma_l2_to_true = float(np.linalg.norm(sigma_pred.numpy() - sigma_true))
    if sigma_init is not None:
        sigma_l2_to_init = float(np.linalg.norm(sigma_pred.numpy() - sigma_init))

    l2_params_to_true = float(np.linalg.norm(params_proj - true_params)) if (true_params is not None) else None
    l2_params_to_init = float(np.linalg.norm(params_proj - initial_params)) if (initial_params is not None) else None

    # 6) Compute joint objective used by calibrator for projected params (reconstruct)
    #    w_ret = 0.05, w_opt = 1.0 and they use mean LL normalization
    mean_lr_proj = lr_proj / float(N) if N > 0 else float(lr_proj)
    mean_lo = opt_lo_mean
    joint_ll = 0.05 * mean_lr_proj + 1.0 * mean_lo
    total_loss = -joint_ll

    # Prepare result dict
    result = {
        "name": name,
        "params_raw": params.tolist(),
        "params_proj": params_proj.tolist(),
        "l2_to_true": l2_params_to_true,
        "l2_to_init": l2_params_to_init,
        "returns_ll_raw_total": lr_raw,
        "returns_ll_proj_total": lr_proj,
        "returns_ll_torch_proj": lr_torch_val,
        "mean_returns_ll_proj": mean_lr_proj,
        "option_mse": opt_mse,
        "option_ll_total": opt_lo,
        "mean_option_ll": mean_lo,
        "sigma_l2_to_true": sigma_l2_to_true,
        "sigma_l2_to_initial": sigma_l2_to_init,
        "joint_ll": joint_ll,
        "total_loss": total_loss,
        "sim_raw_stats": sim_raw["stats"],
        "sim_proj_stats": sim_proj["stats"],
    }

    # Verbose print
    def pfmt(k, v):
        if v is None:
            return f"{k}: N/A"
        if isinstance(v, float):
            return f"{k}: {v:.6g}"
        return f"{k}: {v}"

    print("\n" + "=" * 80)
    print(f"Candidate: {name}")
    print("Raw params:", np.array2string(np.asarray(params, dtype=float), precision=6, floatmode="maxprec"))
    print("Projected params:", np.array2string(np.asarray(params_proj, dtype=float), precision=6, floatmode="maxprec"))
    if l2_params_to_true is not None:
        print(f"L2 distance to TRUE params: {l2_params_to_true:.6f}")
    if l2_params_to_init is not None:
        print(f"L2 distance to INITIAL params: {l2_params_to_init:.6f}")
    print()
    print(pfmt("Total Loss (-joint)", total_loss))
    print(pfmt("Joint LL (w_ret=0.05, w_opt=1)", joint_ll))
    print(pfmt("Option MSE", opt_mse))
    print(pfmt("Option LL total", opt_lo))
    print(pfmt("Option LL mean", mean_lo))
    print()
    print("Returns LL (sim raw total):", lr_raw)
    print("Returns LL (sim proj total):", lr_proj)
    print("Returns LL (torch proj):", lr_torch_val)
    print(pfmt("Mean Returns LL (proj)", mean_lr_proj))
    print()
    print("Simulation stats (RAW params):", sim_raw["stats"])
    print("Simulation stats (PROJECTED params):", sim_proj["stats"])
    print()
    print(pfmt("Sigma L2 to TRUE", sigma_l2_to_true))
    print(pfmt("Sigma L2 to INITIAL", sigma_l2_to_init))
    print("=" * 80 + "\n")

    return result


def build_init_population(bounds: List[Tuple[float, float]], popsize: int, initial_guess: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Build a DE initial population (small helper), inserting the initial_guess at index 0
    and adding a few local perturbations near the initial guess to aid exploration.
    """
    n_params = len(bounds)
    npop = int(popsize * n_params)
    if npop < 4:
        npop = 4
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    rng = np.random.default_rng(seed)

    pop = rng.random((npop, n_params)) * (upper - lower) + lower
    initial_proj = np.clip(initial_guess, lower, upper)
    pop[0, :] = initial_proj

    # Local perturbations (few)
    n_local = min(4, npop - 1)
    for i in range(1, 1 + n_local):
        noise = rng.normal(0, 0.05, size=n_params)
        pert = initial_proj * (1.0 + noise)
        # small absolute noise for near-zero params
        zero_mask = initial_proj == 0.0
        if zero_mask.any():
            pert[zero_mask] = lower[zero_mask] + rng.random(zero_mask.sum()) * 1e-6
        pop[i, :] = np.clip(pert, lower, upper)
    return pop


def evaluate_initial_pop(
    pop: np.ndarray,
    model: torch.nn.Module,
    dataset,
    limit_print: int = 10,
) -> Dict[str, Any]:
    """Evaluate calibration loss on each initial population member to detect 'bad' members."""
    device = torch.device("cpu")
    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    returns_t = dataset.returns.to(device)
    losses = []
    bad_indices = []
    for i in range(pop.shape[0]):
        p = pop[i, :]
        p_t = torch.tensor(p, device=device, dtype=torch.float32)
        p_proj = project_parameters(p_t)
        try:
            with torch.no_grad():
                loss_t = float(Calibration_Loss(p_proj, returns_t, sigma_all, model, X_all, len(returns_t), len(sigma_all)).cpu().item())
        except Exception as e:
            loss_t = float("nan")
        losses.append(loss_t)
        if loss_t > 1e6 or np.isnan(loss_t):
            bad_indices.append(i)
    losses = np.array(losses, dtype=float)
    stats = {
        "min": float(np.nanmin(losses)),
        "median": float(np.nanmedian(losses)),
        "max": float(np.nanmax(losses)),
        "mean": float(np.nanmean(losses)),
        "n_bad": int(len(bad_indices)),
        "bad_indices_sample": bad_indices[:limit_print],
    }
    return {"losses": losses, "stats": stats}


def main():
    p = argparse.ArgumentParser(description="Inspect GARCH parameter candidates for numerical instability")
    p.add_argument("-c", "--candidate", action="append", help="Candidate parameter set as comma-separated 5 floats. Can be repeated.", default=[])
    p.add_argument("--candidate-file", type=str, help="JSON file of candidate parameter lists (list of lists).")
    p.add_argument("--check-init-pop", action="store_true", help="Recreate and evaluate an initial DE population to detect bad members.")
    p.add_argument("--popsize", type=int, default=5, help="Population multiplier to use when constructing init population.")
    p.add_argument("--seed", type=int, default=0, help="Seed for deterministic initial-pop construction.")
    p.add_argument("--sigma-eps", type=float, default=0.01, help="Assumed sigma noise level for option likelihood.")
    p.add_argument("--verbose", action="store_true", help="Verbose printing.")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"Running on device: {device}")

    # Load loader config (first entry)
    loader_path = Path("loaders/loader.json")
    if not loader_path.exists():
        print("Error: loaders/loader.json not found in repo. Exiting.")
        sys.exit(1)

    with open(loader_path, "r") as fh:
        loader = json.load(fh)
    cfg = loader[0]
    dataset_path = cfg["dataset"]
    asset_path = cfg["asset"]
    params_path = cfg["params"]

    print("Loading dataset...")
    dataset = cal_dataset(dataset_path, asset_path)
    print(f"Loaded dataset: {len(dataset)} options, {len(dataset.returns)} returns")

    # Load true params if available
    true_params = None
    try:
        with open(params_path, "r") as fh:
            true_dict = json.load(fh)
        true_params = np.array([true_dict["omega"], true_dict["alpha"], true_dict["beta"], true_dict["gamma"], true_dict["lambda"]], dtype=float)
        print("True params:", true_params)
    except Exception:
        print("Warning: could not load true params file:", params_path)

    # Obtain initial guess (try dataset.target then HN fit)
    initial_params = None
    if hasattr(dataset, "target"):
        try:
            t = dataset.target
            if isinstance(t, torch.Tensor):
                initial_params = t.cpu().numpy().astype(float)
            else:
                initial_params = np.asarray(t, dtype=float)
            print("Initial params from dataset.target:", initial_params)
        except Exception:
            initial_params = None

    if initial_params is None:
        print("Fitting Heston–Nandi to returns for initial guess...")
        hn = HestonNandiGARCH(dataset.returns.cpu().numpy())
        hn.fit()
        initial_params = np.array(hn.fitted_params, dtype=float)
        print("Initial guess (HN fit):", initial_params)

    # Load model on CPU
    print("Loading model (CPU)...")
    model = safe_torch_load_cpu("saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt")
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

    # Build candidate list
    candidates: List[np.ndarray] = []
    # from CLI -c flags
    for s in args.candidate:
        candidates.append(parse_candidate_string(s))

    # from file
    if args.candidate_file:
        with open(args.candidate_file, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("candidate-file must contain a list of candidate lists")
        for entry in data:
            arr = np.asarray(entry, dtype=float)
            if arr.size != 5:
                raise ValueError("Each candidate must contain 5 numbers")
            candidates.append(arr)

    # defaults if none provided
    if not candidates:
        print("No candidates supplied; using example candidates from logs.")
        candidates.append(np.array([2.77e-06, 4.47e-06, 0.4179, 198.87, 9.63], dtype=float))
        candidates.append(np.array([8.14e-04, 2.53e-04, 0.5719, 201.46, 36.81], dtype=float))

    # Evaluate each candidate
    results = []
    for i, cand in enumerate(candidates):
        name = f"candidate_{i+1}"
        try:
            res = evaluate_candidate(
                name=name,
                params=cand,
                model=model,
                dataset=dataset,
                sigma_eps=args.sigma_eps,
                verbose=args.verbose,
                true_params=true_params,
                initial_params=initial_params,
            )
            results.append(res)
        except Exception as e:
            print(f"Error evaluating candidate {name}: {e}")
            results.append({"name": name, "error": str(e)})

    # Optionally check initial population
    if args.check_init_pop:
        print("\n=== Checking initial DE population ===")
        # Reuse bounds similar to calibrate_scipy_de defaults (consistent with calibrator)
        bounds = [
            (1e-10, 1e-5),
            (1e-12, 1e-5),
            (0.2, 0.999),
            (0, 100),
            (0, 5.0),
        ]
        pop = build_init_population(bounds, args.popsize, initial_params, seed=args.seed)
        pop_eval = evaluate_initial_pop(pop, model, dataset)
        print("Initial population diagnostic stats:", pop_eval["stats"])

        # Print some of the worst members (if any)
        losses = pop_eval["losses"]
        sorted_idx = np.argsort(losses)
        print("\nTop 6 best initial members (index, loss):")
        for idx in sorted_idx[:6]:
            print(f"  idx={idx} loss={losses[idx]:.6e}")

        # and worst
        print("\nTop 6 worst initial members (index, loss):")
        for idx in sorted_idx[-6:][::-1]:
            print(f"  idx={idx} loss={losses[idx]:.6e}")

    print("\nAll done. Use the printed diagnostics to locate parameter regions that cause:")
    print(" - extremely negative returns log-likelihood (often due to tiny h or mismatch in r),")
    print(" - huge variance explosion (gamma/alpha interactions causing persistence>=1),")
    print(" - or NN extrapolation producing low MSE but nonsensical parameters (check Sigma L2).")
    # Optionally save results (disabled by default) - the user can redirect output or modify script.

    return 0


if __name__ == "__main__":
    sys.exit(main())
