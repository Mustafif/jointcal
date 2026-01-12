#!/usr/bin/env python3
"""
quick_de_test.py

Small, fast Differential Evolution (DE) smoke-test for the joint option+returns
calibration pipeline.

This script is intended to run a quick calibration with a tiny population and
few iterations so you can verify the pipeline (model loading, dataset loading,
objective evaluation, and DE loop) without committing to a long run.

Usage examples:
  # Quick smoke test with defaults
  python jointcal/scripts/quick_de_test.py

  # Specify dataset/asset and a small DE run
  python jointcal/scripts/quick_de_test.py \
      --dataset joint_dataset/scalable_hn_dataset_250x60.csv \
      --asset joint_dataset/assetprices.csv \
      --popsize 2 --maxiter 10 --no-local-refine --debug-init

Notes:
- This is intentionally conservative (small popsize/maxiter) to keep runtime low.
- Use `--local-refine` if you want the optional L-BFGS-B local polishing after DE
  (can increase runtime noticeably).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import traceback

import numpy as np
import torch

# Local imports (project)
from calibrate_scipy_de_v2 import DECalibrator
from dataset2 import cal_dataset

# Default model path (matches other scripts in this repo)
DEFAULT_MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
DEFAULT_DATASET = "joint_dataset/scalable_hn_dataset_250x60.csv"
DEFAULT_ASSET = "joint_dataset/assetprices.csv"
DEFAULT_PARAMS = "true_params/1.json"


def pick_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps:0"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def load_model(model_path: str, device: torch.device):
    # Try to load with the same kwargs used elsewhere, but fall back gracefully.
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)  # older torch fallback
    model.eval()
    return model


def load_true_params(params_path: str | None):
    if not params_path:
        return None
    p = Path(params_path)
    if not p.exists():
        print(f"Warning: true params file not found: {params_path}")
        return None
    with open(p, "r") as f:
        d = json.load(f)
    return np.array([d["omega"], d["alpha"], d["beta"], d["gamma"], d["lambda"]], dtype=float)


def save_results(output_dir: Path, dataset_name: str, popsize: int, maxiter: int, results: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"quick_result_{dataset_name}_pop{popsize}_iter{maxiter}.json"
    # Convert numpy arrays to lists for JSON
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    sanitized = json.loads(json.dumps(results, default=_convert))
    with open(fname, "w") as f:
        json.dump(sanitized, f, indent=2)
    return fname


def parse_args():
    p = argparse.ArgumentParser(description="Quick DE smoke-test for GARCH calibration")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to saved option model (torch .pt)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="CSV with option dataset")
    p.add_argument("--asset", default=DEFAULT_ASSET, help="CSV with asset prices for returns")
    p.add_argument("--params", default=DEFAULT_PARAMS, help="JSON file with ground-truth GARCH params (optional)")
    p.add_argument("--popsize", type=int, default=2, help="DE popsize multiplier (small for quick test)")
    p.add_argument("--maxiter", type=int, default=10, help="Maximum DE iterations")
    p.add_argument("--strategy", default="best1bin", help="DE strategy (passed to SciPy differential_evolution)")
    p.add_argument("--mutation", default="(0.5, 1.0)", help="Mutation constant or tuple (F) as string")
    p.add_argument("--recombination", type=float, default=0.7, help="Crossover probability (CR)")
    p.add_argument("--local-refine", dest="local_refine", action="store_true", help="Run L-BFGS-B local refine after DE")
    p.add_argument("--no-local-refine", dest="local_refine", action="store_false", help="Skip local refine after DE")
    p.add_argument("--debug-init", action="store_true", help="Evaluate and print initial population diagnostics (can be slow)")
    p.add_argument("--gamma-reg", type=float, default=0.0, help="Regularization weight for gamma (default 0.0)")
    p.add_argument("--lambda-reg", type=float, default=0.0, help="Regularization weight for lambda (default 0.0)")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    p.add_argument("--results-dir", default="calibration_results", help="Directory to save quick results")
    p.set_defaults(local_refine=False)
    return p.parse_args()


def _parse_mutation(mutation_str: str):
    # Accept forms like "(0.5, 1.0)" or "0.5"
    try:
        if "," in mutation_str:
            parts = mutation_str.replace("(", "").replace(")", "").split(",")
            return (float(parts[0].strip()), float(parts[1].strip()))
        return float(mutation_str)
    except Exception:
        print("Failed to parse mutation argument, defaulting to (0.5, 1.0)")
        return (0.5, 1.0)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print("Quick DE test — device:", device)
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"Failed to load model at {args.model_path}: {e}")
        traceback.print_exc()
        return

    try:
        print("Loading dataset...")
        dataset = cal_dataset(args.dataset, args.asset)
        print(f"  Loaded: {len(dataset)} option observations, {len(dataset.returns)} returns")
    except Exception as e:
        print(f"Failed to load dataset/returns: {e}")
        traceback.print_exc()
        return

    true_params = load_true_params(args.params)

    # Build calibrator with small regularization options (user can override)
    calibrator = DECalibrator(
        model=model,
        dataset=dataset,
        true_params=true_params,
        gamma_reg_weight=float(args.gamma_reg),
        lambda_reg_weight=float(args.lambda_reg),
    )

    mutation = _parse_mutation(args.mutation)

    print("\nRunning quick DE with parameters:")
    print(f"  popsize={args.popsize}, maxiter={args.maxiter}, strategy={args.strategy}")
    print(f"  mutation={mutation}, recombination={args.recombination}")
    print(f"  local_refine={args.local_refine}, debug_init_pop={args.debug_init}")

    start_time = time.time()
    try:
        best_params, history, initial_guess = calibrator.calibrate(
            popsize=args.popsize,
            maxiter=args.maxiter,
            strategy=args.strategy,
            mutation=mutation,
            recombination=args.recombination,
            local_refine=args.local_refine,
            refine_maxiter=100,
            debug_init_pop=args.debug_init,
            polish=False,
        )
    except Exception as e:
        print("Calibration failed with exception:")
        traceback.print_exc()
        return

    elapsed = time.time() - start_time
    print("\nQuick DE complete")
    print(f"  Time elapsed: {elapsed:.2f} sec")
    print(f"  Initial guess: {initial_guess}")
    print(f"  Best params (projected): {best_params}")
    if true_params is not None:
        l2_err_final = float(np.linalg.norm(best_params - true_params))
        l2_err_init = float(np.linalg.norm(initial_guess - true_params))
        print(f"  L2 error (init): {l2_err_init:.6f}")
        print(f"  L2 error (final): {l2_err_final:.6f}")

    # Build a small results dict and save
    results = {
        "dataset": args.dataset,
        "asset": args.asset,
        "model_path": args.model_path,
        "popsize": int(args.popsize),
        "maxiter": int(args.maxiter),
        "strategy": args.strategy,
        "mutation": mutation,
        "recombination": float(args.recombination),
        "local_refine": bool(args.local_refine),
        "seed": int(args.seed),
        "initial_guess": initial_guess.tolist() if hasattr(initial_guess, "tolist") else list(initial_guess),
        "best_params": best_params.tolist() if hasattr(best_params, "tolist") else list(best_params),
        "history": history,
        "time_sec": elapsed,
    }

    out_path = save_results(Path(args.results_dir), Path(args.dataset).stem, args.popsize, args.maxiter, results)
    print(f"\nSaved quick results to: {out_path}")


if __name__ == "__main__":
    main()
