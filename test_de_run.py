#!/usr/bin/env python3
"""
jointcal/test_de_run.py

Quick diagnostic script to:
 - Build the same initial population that `calibrate_scipy_de` uses (given a chosen seed)
 - Evaluate the calibration loss for each initial population member
 - Run a short Differential Evolution (DE) calibration with small `popsize` and `maxiter`
 - Print diagnostics (initial pop stats, per-member losses, min/median/max, and history)
 - Try a few DE-style donors to see whether mutation likely produces exploded losses

Usage:
    python jointcal/test_de_run.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

# Import the calibration routine and helpers
import calibrate_scipy_de as csd
from cal_loss import Calibration_Loss
from calibrate_scipy_de import calibrate_scipy_de
from dataset2 import cal_dataset
from hn import HestonNandiGARCH


def build_expected_init_pop(seed, popsize, bounds, initial_guess):
    """
    Recreate the same initial population logic used by calibrate_scipy_de so
    we can inspect members before DE runs.
    """
    np.random.seed(seed)
    n_params = len(bounds)
    npop = int(popsize * n_params)
    if npop < 4:
        npop = 4
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    init_pop = np.random.rand(npop, n_params) * (upper - lower) + lower

    # insert projected initial_guess in index 0 (as calibrate does)
    initial_proj = np.clip(initial_guess, lower, upper)
    init_pop[0, :] = initial_proj
    return init_pop, initial_proj


def eval_loss(params_np, project_fn, model, X_all, returns, sigma_all, device):
    """
    Evaluate the Calibration_Loss for a single parameter vector (numpy array).
    Returns the scalar loss value.
    """
    p_t = torch.tensor(params_np, device=device, dtype=torch.float32)
    p_proj = project_fn(p_t)
    with torch.no_grad():
        loss_t = Calibration_Loss(p_proj, returns, sigma_all, model, X_all, len(returns), len(sigma_all))
    return float(loss_t.cpu().item())


def main():
    # ========== Config ==========
    SEED = 12345
    POPSIZE = 3          # small for quick test
    MAXITER = 8          # short run
    STRATEGY = "rand1bin"  # more exploratory
    MUTATION = (0.5, 1.0)
    RECOMB = 0.7

    # Make sure calibrate_scipy_de runs on CPU for stability in this diagnostic
    csd.device = torch.device("cpu")
    torch.set_num_threads(1)

    # Load dataset configuration (first entry)
    loader_file = Path("loaders/loader.json")
    loader = json.load(open(loader_file, "r"))
    cfg = loader[0]
    dataset_path = cfg["dataset"]
    asset_path = cfg["asset"]
    params_path = cfg["params"]

    print(f"Dataset: {dataset_path}, Asset: {asset_path}, True params file: {params_path}")

    # Load dataset
    dataset = cal_dataset(dataset_path, asset_path)
    print(f"Loaded dataset: {len(dataset)} options, {len(dataset.returns)} returns")

    # Load true parameters
    with open(params_path, "r") as fh:
        true_params_dict = json.load(fh)
    true_vals = np.array([true_params_dict["omega"], true_params_dict["alpha"],
                          true_params_dict["beta"], true_params_dict["gamma"],
                          true_params_dict["lambda"]], dtype=float)
    print("True params:", true_vals)

    # Fit Heston–Nandi to get the initial_guess used in calibrate_scipy_de
    hn = HestonNandiGARCH(dataset.returns.cpu().numpy())
    hn.fit()
    initial_guess = np.array(hn.fitted_params, dtype=float)
    print("HN initial_guess (fitted on returns):", initial_guess)

    # Load model (CPU) robustly with `weights_only` handling so diagnostics don't fail
    try:
        # Prefer a full checkpoint load (allows architecture + weights)
        model = torch.load(csd.MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch may not accept the `weights_only` kwarg — fall back to default load
        try:
            model = torch.load(csd.MODEL_PATH, map_location="cpu")
        except Exception as e:
            print(f"Failed to load model using default loader: {e}")
            raise
    except Exception as e:
        # If full-load fails (unsafe or other reasons), try a weights-only safe load, then default
        print(f"weights_only=False load failed: {e}; trying weights_only=True as fallback")
        try:
            model = torch.load(csd.MODEL_PATH, map_location="cpu", weights_only=True)
        except Exception as e2:
            print(f"weights_only=True also failed: {e2}; trying default load as final fallback")
            model = torch.load(csd.MODEL_PATH, map_location="cpu")
    model.eval()
    # Force CPU placement for diagnostics to avoid GPU memory pressure / OOM
    csd.device = torch.device("cpu")
    model.to(csd.device)

    # Precompute tensors in the device used by calibration
    X_all = dataset.X.to(csd.device)
    sigma_all = dataset.sigma.to(csd.device)
    all_returns = dataset.returns.to(csd.device)

    # Use the same bounds as calibrate_scipy_de (keep in sync)
    bounds = [
        (1e-10, 1e-5),  # omega
        (1e-10, 1e-5),  # alpha
        (0.2, 0.999),   # beta
        (0, 200),       # gamma
        (0, 10.0),      # lambda
    ]

    # Recreate the expected initial population (seeded)
    expected_init_pop, initial_proj = build_expected_init_pop(SEED, POPSIZE, bounds, initial_guess)
    print(f"\nExpected init population shape: {expected_init_pop.shape}")
    print("Initial projected (inserted at index 0):", initial_proj)
    print("First 6 rows of expected init pop:")
    for i, row in enumerate(expected_init_pop[:6]):
        print(f"  [{i:2d}] ", np.array2string(row, precision=8, floatmode="maxprec"))

    # Evaluate the loss at each initial population member (this reveals whether some are already 'exploded')
    print("\nEvaluating loss for each initial-population member...")
    per_member_losses = []
    for i, row in enumerate(expected_init_pop):
        loss_val = eval_loss(row, csd.project_parameters, model, X_all, all_returns, sigma_all, csd.device)
        per_member_losses.append(loss_val)
        flag = ""
        if loss_val > 1e6:
            flag = "  <-- HUGE LOSS"
        print(f"  Member[{i:2d}] Loss: {loss_val:.6e}{flag}")

    per_member_losses = np.array(per_member_losses)
    print("\nInitial-pop loss stats: min: {:.6e}, median: {:.6e}, max: {:.6e}".format(
        per_member_losses.min(), np.median(per_member_losses), per_member_losses.max()
    ))

    # Check initial_guess loss specifically (should match member[0])
    initial_guess_loss = eval_loss(initial_guess, csd.project_parameters, model, X_all, all_returns, sigma_all, csd.device)
    print(f"\nInitial guess loss (direct eval): {initial_guess_loss:.6e}")
    print(f"Inserted member[0] loss: {per_member_losses[0]:.6e}")

    # Quick DE-style mutation probes: simulate a few donors (rand1) to see if mutation can produce extreme vectors
    print("\nSampling a few DE-style donor vectors and evaluating their losses:")
    rng = np.random.default_rng(SEED + 1)
    n_params = expected_init_pop.shape[1]
    npop = expected_init_pop.shape[0]
    for F in [0.5, 0.8, 1.0]:
        donors = []
        donor_losses = []
        for k in range(5):
            # pick three distinct indices
            idx = rng.choice(npop, size=3, replace=False)
            r1, r2, r3 = idx
            donor = expected_init_pop[r1] + F * (expected_init_pop[r2] - expected_init_pop[r3])
            # clip to bounds (SciPy may handle, but we check clipped)
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            donor_clipped = np.clip(donor, lower, upper)
            loss_d = eval_loss(donor_clipped, csd.project_parameters, model, X_all, all_returns, sigma_all, csd.device)
            donors.append(donor_clipped)
            donor_losses.append(loss_d)
            flag = "  <-- HUGE LOSS" if loss_d > 1e6 else ""
            print(f"  F={F:.2f} donor {k} loss: {loss_d:.6e}{flag}")
        print(f"  donor losses (F={F}): min={np.min(donor_losses):.6e}, max={np.max(donor_losses):.6e}")

    # === Run a short DE calibration (controlled, CPU only) ===
    print("\nRunning a short DE calibration (very limited for testing)...")
    # Reset the numpy seed so calibrate_scipy_de will generate the same init population if it uses numpy RNG
    np.random.seed(SEED)
    start_time = time.time()
    try:
        params_scipy, history, returned_initial_guess = calibrate_scipy_de(
            model=model,
            dataset=dataset,
            popsize=POPSIZE,
            maxiter=MAXITER,
            strategy=STRATEGY,
            mutation=MUTATION,
            recombination=RECOMB,
            polish=False,
            atol=1e-6,
            true_vals=true_vals,
        )
    except Exception as e:
        print("DE run raised an exception:", e)
        return

    elapsed = time.time() - start_time
    print(f"\nDE finished in {elapsed:.2f}s")
    print("Returned initial guess (from calibrate):", returned_initial_guess)
    print("Final calibrated params:", params_scipy)
    final_l2 = np.linalg.norm(params_scipy - true_vals)
    print(f"Final L2 distance to true params: {final_l2:.6f}")

    # Inspect history for explosions or flat-lining
    hist = np.array(history)
    print("\nConvergence history stats:")
    print("  length:", len(hist))
    if len(hist) > 0:
        print("  min:", hist.min(), "median:", np.median(hist), "max:", hist.max())
        # count extreme values
        extreme_count = np.sum(hist > 1e6)
        print("  number of extremely large losses (>1e6):", int(extreme_count))

    # Re-evaluate the final param loss directly (should match last history entry if DE stores it that way)
    final_loss_eval = eval_loss(params_scipy, csd.project_parameters, model, X_all, all_returns, sigma_all, csd.device)
    print(f"\nFinal loss evaluated directly: {final_loss_eval:.6e}")

    print("\nTEST COMPLETE - Interpretation hints:")
    print(" - If many initial-pop members or donors have huge losses (>1e6), DE will encounter 'explosion' regions during exploration.")
    print(" - If initial guess is present (member[0]) and is not very bad, it should be visible in the initial-pop losses above.")
    print(" - If DE history shows lots of enormous values or sudden spikes, DE is exploring regions where returns likelihood or variance recursion produce NaNs/explosions; consider tightening bounds or adjusting objective weights for stability.")

if __name__ == "__main__":
    main()
