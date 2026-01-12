"""
Simplified GARCH Parameter Calibration with Differential Evolution.

This script provides a clean interface for calibrating Heston-Nandi GARCH
parameters using differential evolution optimization.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis, skew

from calibrate_scipy_de_v2 import DECalibrator
from dataset2 import cal_dataset

# Configuration
MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
LOADER_PATH = "loaders/loader.json"
RESULTS_DIR = Path("calibration_results")

# Device selection
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def monte_carlo_simulation(omega, alpha, beta, gamma, lambda_, n_steps=512, n_paths=10, r=0.05):
    """
    Monte Carlo simulation for GARCH process.

    Returns simulated return paths.
    """
    dt = 1 / n_steps
    Z = np.random.normal(0, 1, (n_steps + 1, n_paths))

    Rt = np.zeros((n_steps + 1, n_paths))
    ht = np.zeros((n_steps + 1, n_paths))

    # Initial variance (unconditional)
    persistence = beta + alpha * gamma ** 2
    if persistence < 1.0:
        ht[0] = (omega + alpha) / (1.0 - persistence + 1e-8)
    else:
        ht[0] = omega + alpha

    for i in range(1, n_steps + 1):
        ht[i] = omega + beta * ht[i - 1] + alpha * (Z[i - 1] - gamma * np.sqrt(ht[i - 1])) ** 2
        Rt[i] = r + lambda_ * ht[i] + np.sqrt(ht[i]) * Z[i]

    return Rt


def compute_moments(returns, time_points):
    """Compute statistical moments at specified time points."""
    moments = {}
    for t in time_points:
        Rt_t = returns[: t + 1].flatten()
        moments[str(t)] = {
            "mean": float(np.mean(Rt_t)),
            "variance": float(np.var(Rt_t)),
            "skewness": float(skew(Rt_t)),
            "kurtosis": float(kurtosis(Rt_t)),
        }
    return moments


def load_config():
    """Load dataset configurations from loader file."""
    with open(LOADER_PATH, "r") as f:
        return json.load(f)


def load_true_params(params_path):
    """Load true parameters from JSON file."""
    with open(params_path, "r") as f:
        params_dict = json.load(f)

    return np.array([
        params_dict["omega"],
        params_dict["alpha"],
        params_dict["beta"],
        params_dict["gamma"],
        params_dict["lambda"],
    ])


def run_calibration(config, model):
    """
    Run calibration for a single dataset configuration.

    Returns results dictionary.
    """
    dataset_path = config["dataset"]
    asset_path = config["asset"]
    params_path = config["params"]

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_path}")
    print(f"Assets: {asset_path}")
    print(f"{'=' * 70}")

    # Load true parameters
    true_vals = load_true_params(params_path)
    print(f"True params: {dict(zip(['ω', 'α', 'β', 'γ', 'λ'], true_vals))}")

    # Load dataset
    dataset = cal_dataset(dataset_path, asset_path)
    print(f"Dataset: {len(dataset)} options, {len(dataset.returns)} returns")

    # Create calibrator (seed gamma/lambda regularization towards HN initial guess)
    # Increased regularization to discourage large gamma/lambda excursions during DE.
    calibrator = DECalibrator(
        model=model,
        dataset=dataset,
        true_params=true_vals,
        gamma_reg_weight=0.1,
        lambda_reg_weight=0.1,
    )

    # Run calibration (enable debug evaluation of the initial population)
    best_params, history, initial_guess = calibrator.calibrate(
        popsize=10,
        maxiter=100,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        local_refine=True,
        refine_maxiter=200,
        debug_init_pop=True,
    )

    # Compute metrics
    l2_error = np.linalg.norm(best_params - true_vals)
    init_error = np.linalg.norm(initial_guess - true_vals)
    persistence = best_params[1] + best_params[2] * best_params[3] ** 2

    # Monte Carlo analysis
    time_points = [10, 50, 100, 252, 365, 512]

    print("\nRunning Monte Carlo simulations...")
    Rt_true = monte_carlo_simulation(*true_vals)
    Rt_init = monte_carlo_simulation(*initial_guess)
    Rt_final = monte_carlo_simulation(*best_params)

    true_moments = compute_moments(Rt_true, time_points)
    init_moments = compute_moments(Rt_init, time_points)
    final_moments = compute_moments(Rt_final, time_points)

    # Build results
    param_names = ["omega", "alpha", "beta", "gamma", "lambda"]

    results = {
        "dataset_info": {
            "dataset_path": dataset_path,
            "asset_path": asset_path,
            "params_path": params_path,
            "dataset_name": Path(dataset_path).stem,
        },
        "parameter_sets": {
            "true_parameters": {
                "parameters": dict(zip(param_names, true_vals.tolist())),
                "moments_by_time": true_moments,
            },
            "initial_guess": {
                "parameters": dict(zip(param_names, initial_guess.tolist())),
                "moments_by_time": init_moments,
            },
            "calibrated": {
                "parameters": dict(zip(param_names, best_params.tolist())),
                "moments_by_time": final_moments,
            },
        },
        "metrics": {
            "l2_error_final": float(l2_error),
            "l2_error_initial": float(init_error),
            "error_improvement": float(init_error - l2_error),
            "persistence": float(persistence),
            "is_stationary": bool(persistence < 1.0),
        },
        "convergence_history": [
            {"iteration": h["iteration"], "loss": h["loss"]}
            for h in history
        ],
    }

    return results


def main():
    """Main entry point."""
    print("🚀 GARCH Parameter Calibration with Differential Evolution")
    print("=" * 70)

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load model
    print("Loading model...")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    print(f"✅ Model loaded (device: {device})")

    # Load configurations
    print(f"Loading configurations from {LOADER_PATH}...")
    configs = load_config()
    print(f"✅ Found {len(configs)} dataset(s)")

    # Process each dataset
    all_results = {}
    successful = 0
    failed = 0

    for idx, config in enumerate(configs):
        print(f"\n📊 Processing Dataset {idx + 1}/{len(configs)}")

        try:
            results = run_calibration(config, model)
            dataset_name = results["dataset_info"]["dataset_name"]
            l2_error = results["metrics"]["l2_error_final"]

            # Save individual results
            filename = RESULTS_DIR / f"result_{dataset_name}_l2_{l2_error:.6f}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)

            print(f"✅ Saved: {filename}")

            all_results[f"dataset_{idx + 1}"] = results
            successful += 1

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

            all_results[f"dataset_{idx + 1}"] = {
                "error": str(e),
                "status": "failed",
            }
            failed += 1

    # Save combined results
    combined_file = RESULTS_DIR / "all_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total datasets: {len(configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        errors = [
            r["metrics"]["l2_error_final"]
            for r in all_results.values()
            if "metrics" in r
        ]
        improvements = [
            r["metrics"]["error_improvement"]
            for r in all_results.values()
            if "metrics" in r
        ]

        print(f"\nL2 Errors:")
        print(f"  Mean: {np.mean(errors):.6f}")
        print(f"  Best: {min(errors):.6f}")
        print(f"  Worst: {max(errors):.6f}")

        print(f"\nError Improvements (initial - final):")
        print(f"  Mean: {np.mean(improvements):.6f}")
        print(f"  Best: {max(improvements):.6f}")

    print(f"\n📁 Results saved to: {RESULTS_DIR}")
    print("🎉 Calibration complete!")


if __name__ == "__main__":
    main()
