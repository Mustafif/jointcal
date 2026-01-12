#!/usr/bin/env python3
"""
test_recovery.py

Synthetic-data parameter recovery test for the Heston–Nandi GARCH(1,1) fit.

This script:
- Simulates return series from the Heston–Nandi GARCH dynamics using a known
  parameter vector (omega, alpha, beta, gamma, lambda).
- Fits the Heston–Nandi GARCH model to the simulated returns using the
  repository's `HestonNandiGARCH` fitter.
- Reports per-trial and aggregate recovery statistics (L2 / L_inf errors).
- Optionally runs multiple independent trials and saves results to JSON.

Usage:
    # quick single trial
    python jointcal/scripts/test_recovery.py

    # multiple trials, verbose output, save results
    python jointcal/scripts/test_recovery.py --trials 10 --verbose --save-out results.json

Notes:
- The script appends the repository root to `sys.path` so it can be executed
  from anywhere (convenient for CI or local use).
- Defaults match the repository's synthetic dataset "true" params:
    omega=1e-6, alpha=1.33e-6, beta=0.8, gamma=5.0, lambda=0.2
- The test is intentionally simple and designed as a debugging / CI helper.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Ensure project root is importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports (must be importable after adding repo root)
from hn import HestonNandiGARCH  # type: ignore


PARAM_NAMES = ["omega", "alpha", "beta", "gamma", "lambda"]


def simulate_hn_returns(
    params: Sequence[float],
    n_steps: int = 1259,
    seed: int | None = None,
    r: float = 0.0,
) -> np.ndarray:
    """
    Simulate a single path of returns under the Heston–Nandi GARCH(1,1) model.

    Model:
        R_t = r + lambda * h_t + sqrt(h_t) * z_t
        h_{t+1} = omega + beta * h_t + alpha * (z_t - gamma * sqrt(h_t))^2
        z_t ~ N(0, 1)

    Returns:
        returns: np.ndarray shape (n_steps,)
    """
    omega, alpha, beta, gamma, lambda_ = map(float, params)
    rng = np.random.RandomState(seed)
    z = rng.normal(size=n_steps)

    # Reserve arrays
    h = np.zeros(n_steps, dtype=float)
    returns = np.zeros(n_steps, dtype=float)

    persistence = beta + alpha * gamma * gamma
    # Initialize h[0] as unconditional variance when stationary, fallback otherwise
    if persistence < 1.0:
        h[0] = (omega + alpha) / (1.0 - persistence + 1e-12)
    else:
        h[0] = max(omega + alpha, 1e-8)

    for t in range(n_steps):
        # Observed return at time t
        returns[t] = r + lambda_ * h[t] + math.sqrt(max(h[t], 1e-12)) * z[t]

        # Update variance for next step if not at last time
        if t < n_steps - 1:
            h[t + 1] = omega + beta * h[t] + alpha * (z[t] - gamma * math.sqrt(max(h[t], 1e-12))) ** 2

    return returns


def fit_hn_to_returns(returns: np.ndarray, suppress_output: bool = True):
    """
    Fit Heston–Nandi GARCH to an observed return series.

    Returns the fitted HestonNandiGARCH instance and the scipy result object.
    """
    hn = HestonNandiGARCH(returns)
    if suppress_output:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = hn.fit()
    else:
        res = hn.fit()
    return hn, res


def l2_linf_error(est: Sequence[float], true: Sequence[float]) -> Tuple[float, float]:
    est = np.asarray(est, dtype=float)
    true = np.asarray(true, dtype=float)
    l2 = float(np.linalg.norm(est - true))
    linf = float(np.linalg.norm(est - true, ord=np.inf))
    return l2, linf


def parse_params_paramstr(s: str) -> List[float]:
    """
    Parse a comma-separated list of five floats into parameter vector.
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 5:
        raise ValueError("Parameter string must have 5 comma-separated values.")
    return [float(p) for p in parts]


def run_once(
    true_params: Sequence[float],
    n_steps: int,
    seed: int | None = None,
    r: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    One trial: simulate returns from `true_params`, fit HN, and report stats.
    """
    returns = simulate_hn_returns(true_params, n_steps=n_steps, seed=seed, r=r)
    hn, res = fit_hn_to_returns(returns, suppress_output=not verbose)

    fitted = hn.fitted_params if hn.fitted_params is not None else np.full(5, np.nan)
    l2, linf = l2_linf_error(fitted, true_params)

    trial_result = {
        "seed": int(seed) if seed is not None else None,
        "n_steps": int(n_steps),
        "true_params": dict(zip(PARAM_NAMES, [float(x) for x in true_params])),
        "fitted_params": dict(zip(PARAM_NAMES, [float(x) for x in fitted.tolist()])),
        "l2_error": float(l2),
        "linf_error": float(linf),
        "success": bool(getattr(res, "success", False)),
        "scipy_result": {
            "success": bool(getattr(res, "success", False)),
            "message": getattr(res, "message", None),
        },
    }
    return trial_result


def summary_stats(results: Iterable[dict]) -> dict:
    arr = np.array([r["l2_error"] for r in results])
    success_count = sum(1 for r in results if r.get("success", False))
    n = len(arr)
    return {
        "n_trials": int(n),
        "successes": int(success_count),
        "mean_l2": float(np.mean(arr)),
        "median_l2": float(np.median(arr)),
        "best_l2": float(np.min(arr)),
        "worst_l2": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Heston–Nandi parameter recovery smoke test")
    parser.add_argument(
        "--true-params",
        type=str,
        default=None,
        help="Comma-separated true params (omega,alpha,beta,gamma,lambda). "
        "If omitted, defaults to 1e-6,1.33e-6,0.8,5.0,0.2",
    )
    parser.add_argument("--n-steps", type=int, default=1259, help="Number of returns to simulate")
    parser.add_argument("--trials", type=int, default=5, help="Number of independent trials")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed (trials use seed+i)")
    parser.add_argument("--r", type=float, default=0.0, help="Risk-free daily return (drift) to add to returns")
    parser.add_argument("--tol", type=float, default=2.0, help="Acceptable L2 error for a 'successful' recovery")
    parser.add_argument("--verbose", action="store_true", help="Print solver output for each trial")
    parser.add_argument("--save-out", type=str, default=None, help="Path to save JSON results")
    args = parser.parse_args()

    if args.true_params:
        try:
            true_params = parse_params_paramstr(args.true_params)
        except Exception as e:
            print("Failed to parse --true-params:", e)
            return 2
    else:
        true_params = [1e-6, 1.33e-6, 0.8, 5.0, 0.2]

    print("Heston–Nandi parameter recovery test")
    print("True params:", dict(zip(PARAM_NAMES, true_params)))
    print(f"Trials: {args.trials}, steps/trial: {args.n_steps}, base seed: {args.seed}")
    print(f"Tolerance (L2): {args.tol}")
    print("Starting trials...\n")

    all_results = []
    start_time = time.time()

    for i in range(args.trials):
        s = args.seed + i if args.seed is not None else None
        try:
            res = run_once(
                true_params=true_params,
                n_steps=args.n_steps,
                seed=s,
                r=args.r,
                verbose=args.verbose,
            )
        except Exception as e:
            res = {
                "seed": s,
                "error": str(e),
                "l2_error": float("nan"),
                "linf_error": float("nan"),
                "success": False,
            }

        msg = (
            f"Trial {i+1}/{args.trials} (seed={res.get('seed')}): "
            f"L2={res['l2_error']:.6g}, Linf={res['linf_error']:.6g}, success={res.get('success')}"
        )
        print(msg)
        if args.verbose:
            print("  fitted:", res.get("fitted_params"))
            print("  scipy:", res.get("scipy_result"))

        all_results.append(res)

    elapsed = time.time() - start_time
    stats = summary_stats(all_results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Elapsed: {elapsed:.2f} sec")

    # Basic pass/fail decision (useful for CI)
    passes = stats["successes"]
    if passes >= math.ceil(0.5 * stats["n_trials"]):
        print("\nResult: PASS (>=50% of trials reported optimizer success)")
    else:
        print("\nResult: WARNING/FAIL (less than half the trials converged successfully)")

    # Optional save
    if args.save_out:
        out = {
            "config": {
                "true_params": dict(zip(PARAM_NAMES, [float(x) for x in true_params])),
                "n_steps": int(args.n_steps),
                "trials": int(args.trials),
                "seed": int(args.seed) if args.seed is not None else None,
            },
            "stats": stats,
            "trials": all_results,
        }
        out_path = Path(args.save_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to {out_path}")

    # Also return non-zero exit code when the run is poor (for CI integration)
    # Here we treat less than half successful as failure.
    if passes < math.ceil(0.5 * stats["n_trials"]):
        sys.exit(3)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
