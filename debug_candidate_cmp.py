#!/usr/bin/env python3
"""
debug_candidate_cmp.py

Compare candidate DE parameter vectors vs true and initial parameters.
Prints:
 - Returns LL (total and mean)
 - Options LL (total and mean)
 - Joint LL (weighted mean as used in calibration)
 - Optimization loss (=-JointLL)
 - Option MSE
 - L2 norm between predicted implied vols (sigma) for candidate vs true
 - Parameter L2 norms (candidate -> true, candidate -> initial)
 - Per-parameter values and relative errors

Usage examples:
  # Compare default example candidate (from logs)
  python jointcal/debug_candidate_cmp.py

  # Supply a candidate on the CLI
  python jointcal/debug_candidate_cmp.py -c 2.77e-06,4.47e-06,0.4179,198.87,9.63

  # Multiple candidates:
  python jointcal/debug_candidate_cmp.py -c 2.77e-06,4.47e-06,0.4179,198.87,9.63 -c 8.14e-04,2.53e-04,0.5719,201.46,36.81

  # Load list of candidates from JSON file (list of list)
  python jointcal/debug_candidate_cmp.py --candidate-file my_candidates.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Project imports
from cal_loss import ll_option_torch, ll_returns_torch
from dataset2 import cal_dataset
from hn import HestonNandiGARCH

# Settings consistent with the rest of the repo
MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
LOADER_PATH = "loaders/loader.json"

# Weights used in ll_joint_torch (same as current implementation)
W_RET = 0.05
W_OPT = 1.0


def project_params_np(p):
    """Project parameter vector to a valid domain (same rules used in calibration)."""
    p = np.asarray(p, dtype=float).copy()
    p[0] = max(p[0], 1e-10)  # omega > 0
    p[1] = float(np.clip(p[1], 0.0, 1.0))  # alpha in [0,1]
    p[2] = float(np.clip(p[2], 0.0, 1.0))  # beta in [0,1]
    p[3] = max(p[3], 0.0)  # gamma >= 0
    p[4] = max(p[4], 0.0)  # lambda >= 0
    return p


def parse_candidate_string(s):
    """Parse a comma-separated string into a 5-element numpy array."""
    arr = np.fromstring(s, sep=",")
    if arr.size != 5:
        raise ValueError(f"Expected 5 values for a parameter set, got {arr.size}: {s}")
    return arr.astype(float)


def load_loader_config(idx=0):
    with open(LOADER_PATH, "r") as fh:
        loader = json.load(fh)
    if idx < 0 or idx >= len(loader):
        raise IndexError("dataset index out of range")
    return loader[idx]


def inject_params_to_X(X, params, device):
    """
    Given X (M x features) and params (np or torch), inject params into indices:
      5: alpha, 6: beta, 7: omega, 8: gamma, 9: lambda
    and engineered log-features:
      19: log_gamma, 20: log_omega, 21: log_lambda, 22: log_alpha, 23: log_beta
    Returns a new tensor (on device) with injections applied.
    """
    x_in = X.clone().to(device)
    eps = 1e-8
    if not isinstance(params, torch.Tensor):
        params_t = torch.tensor(params, dtype=torch.float32, device=device)
    else:
        params_t = params.to(device).to(torch.float32)

    omega, alpha, beta, gamma, lambda_ = params_t

    # base params
    x_in[:, 5] = alpha
    x_in[:, 6] = beta
    x_in[:, 7] = omega
    x_in[:, 8] = gamma
    x_in[:, 9] = lambda_

    # logs / engineered
    x_in[:, 19] = torch.log(gamma + eps)
    x_in[:, 20] = torch.log(omega + eps)
    x_in[:, 21] = torch.log(lambda_ + eps)
    x_in[:, 22] = torch.log(alpha + eps)
    x_in[:, 23] = torch.log(beta + eps)

    return x_in


def evaluate_candidate(name, params_np, model, dataset, device, true_sigma_pred=None, initial_sigma_pred=None, true_params_np=None, initial_params_np=None):
    """
    Compute and print the diagnostics for a single parameter vector.
    Returns a dict with metrics for further automated inspection if needed.
    """
    # Project input params to valid domain (for making predictions)
    params_proj = project_params_np(params_np)

    # Prepare tensors
    X = dataset.X.to(device)
    returns = dataset.returns.to(device)
    sigma_obs = dataset.sigma.to(device)
    N = len(returns)
    M = len(dataset)

    # r_daily as used in calibration (mean of column index 2)
    r_daily = float(X[:, 2].mean().item())

    # Inject params into X and predict implied vols
    x_candidate = inject_params_to_X(X, params_proj, device)
    with torch.no_grad():
        sigma_candidate = model(x_candidate).squeeze()

    # Option MSE (observed vs predicted)
    opt_mse = torch.nn.MSELoss()(sigma_obs, sigma_candidate).item()

    # Option LL
    lo = ll_option_torch(sigma_obs, sigma_candidate, sigma_eps=0.01)
    lo_total = float(lo.item())
    lo_mean = lo_total / M

    # Returns LL
    params_torch = torch.tensor(params_proj, dtype=torch.float32, device=device)
    lr = ll_returns_torch(returns, params_torch, r_daily)
    lr_total = float(lr.item())
    lr_mean = lr_total / N

    # Joint LL (as used in ll_joint_torch)
    joint_ll = W_RET * lr_mean + W_OPT * lo_mean
    # Calibration loss used by optimizer is -joint_ll
    opt_loss = -joint_ll

    # Sigma L2 against true sigma (if provided)
    sigma_l2_to_true = None
    if true_sigma_pred is not None:
        sigma_l2_to_true = float(np.linalg.norm(sigma_candidate.cpu().numpy() - true_sigma_pred))

    sigma_l2_to_initial = None
    if initial_sigma_pred is not None:
        sigma_l2_to_initial = float(np.linalg.norm(sigma_candidate.cpu().numpy() - initial_sigma_pred))

    # Parameter norms
    l2_to_true = None
    if true_params_np is not None:
        l2_to_true = float(np.linalg.norm(params_np - true_params_np))

    l2_to_initial = None
    if initial_params_np is not None:
        l2_to_initial = float(np.linalg.norm(params_np - initial_params_np))

    # Print nicely
    sep = "-" * 80
    print(sep)
    print(f"Candidate: {name}")
    print("Params (raw input):     ", np.array2string(np.asarray(params_np), precision=6, floatmode="maxprec"))
    print("Params (projected):     ", np.array2string(np.asarray(params_proj), precision=6, floatmode="maxprec"))
    if l2_to_true is not None:
        print(f"L2 norm to TRUE params: {l2_to_true:.6f}")
    if l2_to_initial is not None:
        print(f"L2 norm to INITIAL params: {l2_to_initial:.6f}")
    print()

    print("Return Log-Likelihood (total):      ", f"{lr_total:.6e}")
    print("Return Log-Likelihood (mean):       ", f"{lr_mean:.6f}")
    print("Option Log-Likelihood (total):      ", f"{lo_total:.6e}")
    print("Option Log-Likelihood (mean):       ", f"{lo_mean:.6f}")
    print(f"Joint LL (w_ret={W_RET}, w_opt={W_OPT}): ", f"{joint_ll:.6f}")
    print("Optimization Loss (-Joint LL):      ", f"{opt_loss:.6f}")
    print()
    print(f"Option MSE (sigma_obs vs sigma_pred): {opt_mse:.6e}")
    if sigma_l2_to_true is not None:
        print(f"Sigma L2 norm (candidate vs true sigma): {sigma_l2_to_true:.6f}")
    if sigma_l2_to_initial is not None:
        print(f"Sigma L2 norm (candidate vs initial sigma): {sigma_l2_to_initial:.6f}")
    print()

    # Print a small sample of predicted vs observed sigmas
    sample_n = min(8, len(sigma_obs))
    print("Sample Sigmas (obs | pred):")
    sigma_obs_np = sigma_obs.cpu().numpy()
    sigma_pred_np = sigma_candidate.cpu().numpy()
    for i in range(sample_n):
        print(f"  {sigma_obs_np[i]:.6f} | {sigma_pred_np[i]:.6f}")
    print(sep)
    print()

    # Return a dict of metrics
    return {
        "params_raw": params_np.tolist(),
        "params_proj": params_proj.tolist(),
        "lr_total": lr_total,
        "lr_mean": lr_mean,
        "lo_total": lo_total,
        "lo_mean": lo_mean,
        "joint_ll": joint_ll,
        "opt_loss": opt_loss,
        "opt_mse": opt_mse,
        "sigma_l2_to_true": sigma_l2_to_true,
        "sigma_l2_to_initial": sigma_l2_to_initial,
        "l2_to_true": l2_to_true,
        "l2_to_initial": l2_to_initial,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare DE candidate parameters vs true and initial params")
    parser.add_argument("-c", "--candidate", action="append", help="Candidate parameter set (comma-separated 5 floats). Can be repeated.", default=[])
    parser.add_argument("--candidate-file", type=str, help="JSON file with list of candidate parameter lists.")
    parser.add_argument("--dataset-index", type=int, default=0, help="Index into loader.json (default 0).")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to saved PyTorch model.")
    parser.add_argument("--verbose", action="store_true", help="Verbose prints.")
    args = parser.parse_args()

    # Device
    device = torch.device("cpu")
    print("Using device: cpu (forced for debug to avoid GPU OOM)")

    # Load config
    cfg = load_loader_config(args.dataset_index)
    dataset_path = cfg["dataset"]
    asset_path = cfg["asset"]
    params_path = cfg["params"]

    print("Dataset:", dataset_path)
    print("Asset:", asset_path)
    print("True params file:", params_path)

    # Load true parameters
    with open(params_path, "r") as fh:
        true_params_dict = json.load(fh)
    true_params_np = np.array(
        [
            true_params_dict["omega"],
            true_params_dict["alpha"],
            true_params_dict["beta"],
            true_params_dict["gamma"],
            true_params_dict["lambda"],
        ],
        dtype=float,
    )

    print("True params:", true_params_np)

    # Load dataset
    dataset = cal_dataset(dataset_path, asset_path)
    print(f"Loaded dataset: {len(dataset)} options, {len(dataset.returns)} returns")

    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    # Load model onto the CPU explicitly for debugging
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.eval()
    model.to(device)
    print("Model loaded (CPU).")

    # Initial guess (try dataset.target, else fit HN)
    initial_params_np = None
    if hasattr(dataset, "target"):
        try:
            t = dataset.target
            if isinstance(t, torch.Tensor):
                initial_params_np = t.cpu().numpy().astype(float)
            else:
                initial_params_np = np.asarray(t, dtype=float)
            print("Initial params from dataset.target:", initial_params_np)
        except Exception:
            initial_params_np = None

    if initial_params_np is None:
        print("Fitting Heston-Nandi to returns to get initial guess...")
        hn = HestonNandiGARCH(dataset.returns.cpu().numpy())
        hn.fit()
        initial_params_np = np.array(hn.fitted_params, dtype=float)
        print("Initial HN GARCH params:", initial_params_np)

    # Precompute sigma predictions for true and initial params for sigma L2 comparisons
    X = dataset.X.to(device)
    with torch.no_grad():
        x_true = inject_params_to_X(X, project_params_np(true_params_np), device)
        sigma_true_pred = model(x_true).squeeze().cpu().numpy()
        x_init = inject_params_to_X(X, project_params_np(initial_params_np), device)
        sigma_init_pred = model(x_init).squeeze().cpu().numpy()

    # Build candidates list
    candidates = []
    # From CLI -c flags
    for s in args.candidate:
        candidates.append(parse_candidate_string(s))

    # Candidate file (JSON)
    if args.candidate_file:
        with open(args.candidate_file, "r") as fh:
            cdata = json.load(fh)
        if isinstance(cdata, list):
            for entry in cdata:
                arr = np.asarray(entry, dtype=float)
                if arr.size != 5:
                    raise ValueError("Each entry in candidate file must be a list of 5 numbers")
                candidates.append(arr)
        else:
            raise ValueError("Candidate file must contain a list of parameter lists")

    # Default candidate if none provided (example from logs)
    if not candidates:
        print("No candidates supplied; using example candidate(s) from recent logs.")
        candidates.append(np.array([2.77e-06, 4.47e-06, 0.4179, 198.87, 9.63], dtype=float))
        candidates.append(np.array([8.14e-04, 2.53e-04, 0.5719, 201.46, 36.81], dtype=float))

    # Evaluate all
    results = {}
    for i, cand in enumerate(candidates):
        name = f"candidate_{i+1}"
        try:
            metrics = evaluate_candidate(
                name=name,
                params_np=cand,
                model=model,
                dataset=dataset,
                device=device,
                true_sigma_pred=sigma_true_pred,
                initial_sigma_pred=sigma_init_pred,
                true_params_np=true_params_np,
                initial_params_np=initial_params_np,
            )
            results[name] = metrics
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = {"error": str(e)}

    # Summary table
    print("\n\nSUMMARY TABLE")
    print("=" * 80)
    header = f"{'name':<15} {'loss':>12} {'L2->true':>12} {'L2->init':>12} {'OptMSE':>12} {'SigmaL2(true)':>15}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<15} ERROR: {m['error']}")
            continue
        print(
            f"{name:<15} {m['opt_loss']:12.6f} "
            f"{(m['l2_to_true'] if m['l2_to_true'] is not None else float('nan')):12.6f} "
            f"{(m['l2_to_initial'] if m['l2_to_initial'] is not None else float('nan')):12.6f} "
            f"{m['opt_mse']:12.6e} "
            f"{(m['sigma_l2_to_true'] if m['sigma_l2_to_true'] is not None else float('nan')):15.6f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
