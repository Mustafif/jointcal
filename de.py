import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis, skew


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""

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


# Option 2: SciPy DE implementation (recommended - professional grade)
from cal_loss import Calibration_Loss
from calibrate_scipy_de import calibrate_scipy_de
from dataset2 import cal_dataset

# Load model and dataset
MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
LOADER_PATH = "loaders/loader.json"
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("🚀 GARCH Parameter Calibration with Differential Evolution - Multi Dataset")
print("=" * 75)

print("Loading model...")
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
print("✅ Model loaded successfully")

print(f"Loading dataset configurations from {LOADER_PATH}...")
with open(LOADER_PATH, "r") as f:
    dataset_configs = json.load(f)
print(f"✅ Found {len(dataset_configs)} dataset configurations")

# Create results directory
results_dir = Path("calibration_results")
results_dir.mkdir(exist_ok=True)

# Store all results
all_results = {}

# Monte Carlo parameters
r = 0.05
M = 10
N = 512
TN = [10, 50, 100, 252, 365, 512]
dt = 1 / N
Z = np.random.normal(0, 1, (N + 1, M))


def mc(omega, alpha, beta, gamma, lambda_):
    """Monte Carlo simulation for GARCH process"""
    # Ensure all parameters are scalar floats
    omega = float(omega)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    lambda_ = float(lambda_)

    num_point = N + 1
    Rt = np.zeros((num_point, M))
    ht = np.zeros((num_point, M))

    # Calculate initial variance - ensure scalar operations
    initial_var = (omega + alpha) / (1.0 - beta - alpha * gamma**2)
    ht[0] = initial_var
    Rt[0] = 0

    for i in range(1, num_point):
        ht[i] = (
            omega
            + beta * ht[i - 1]
            + alpha * (Z[i - 1] - gamma * np.sqrt(ht[i - 1])) ** 2
        )
        Rt[i] = r + lambda_ * ht[i] + np.sqrt(ht[i]) * Z[i]
    return Rt


def compute_moments(Rt, TN):
    """Compute four moments for given time points"""
    moments = {}
    for t in TN:
        Rt_t = Rt[: t + 1].flatten()  # Ensure 1D array for moment calculations
        moments[str(t)] = {  # Convert t to string for JSON compatibility
            "mean": float(np.mean(Rt_t)),
            "variance": float(np.var(Rt_t)),
            "skewness": float(skew(Rt_t)),
            "kurtosis": float(kurtosis(Rt_t)),
        }
    return moments


def compute_function_value(params, model, dataset, device):
    """Compute the calibration loss function value for given parameters"""
    try:
        params_tensor = torch.tensor(params, device=device, dtype=torch.float32)
        with torch.no_grad():
            loss = Calibration_Loss(
                params_tensor,
                dataset.returns.to(device),
                dataset.sigma.to(device),
                model,
                dataset.X.to(device),
                len(dataset.returns),
                len(dataset),
            )
        return float(loss.cpu().item())
    except Exception as e:
        print(f"    Warning: Could not compute function value: {e}")
        return None


# Process each dataset configuration
for idx, config in enumerate(dataset_configs):
    dataset_path = config["dataset"]
    asset_path = config["asset"]
    params_path = config["params"]

    print(f"\n{'=' * 75}")
    print(f"📊 Processing Dataset {idx + 1}/{len(dataset_configs)}")
    print(f"Dataset: {dataset_path}")
    print(f"Assets: {asset_path}")
    print(f"True params: {params_path}")
    print(f"{'=' * 75}")

    # Load true parameters
    print("Loading true parameters...")
    with open(params_path, "r") as f:
        true_params_dict = json.load(f)

    param_names = ["omega", "alpha", "beta", "gamma", "lambda"]
    true_vals = np.array(
        [
            true_params_dict["omega"],
            true_params_dict["alpha"],
            true_params_dict["beta"],
            true_params_dict["gamma"],
            true_params_dict["lambda"],
        ]
    )

    print(f"✅ True parameters loaded: {dict(zip(param_names, true_vals))}")

    # Load dataset
    print("Loading dataset...")
    dataset = cal_dataset(dataset_path, asset_path)
    print(f"✅ Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

    # Run SciPy DE calibration
    print(f"\n🔬 Running SciPy Differential Evolution...")
    try:
        params_scipy, history_scipy, initial_guess = calibrate_scipy_de(
            model=model,
            dataset=dataset,
            popsize=50,  # Population size multiplier
            maxiter=400,  # Maximum iterations
            strategy="best1bin",  # DE strategy (fast and reliable)
            mutation=(0.5, 1.0),  # Adaptive mutation range
            recombination=0.7,  # Crossover probability
            polish=False,  # Use L-BFGS-B for final refinement
            atol=1e-6,  # Convergence tolerance
        )

        # Calculate error
        l2_error = np.linalg.norm(params_scipy - true_vals, ord=2)

        print(f"\n✅ Calibration completed!")
        print(f"📊 Final calibrated parameters:")

        omega, alpha, beta, gamma, lam = params_scipy

        print(f"  Omega: {omega:.8f}")
        print(f"  Alpha: {alpha:.8f}")
        print(f"  Beta: {beta:.8f}")
        print(f"  Gamma: {gamma:.8f}")
        print(f"  Lambda: {lam:.8f}")

        print(f"L2 error: {l2_error:.8f}")

        # Validate results
        persistence = alpha + beta
        print(f"\n📋 Validation:")
        print(f"  Persistence (α+β): {persistence:.6f}")
        print(f"  Stationary: {'✅' if persistence < 1.0 else '❌'}")
        print(f"  Omega > 0: {'✅' if omega > 0 else '❌'}")

        # Prepare results for this dataset
        dataset_name = Path(dataset_path).stem
        # Compute function values for all parameter sets
        print("Computing function values...")
        true_func_val = compute_function_value(true_vals, model, dataset, device)
        initial_func_val = compute_function_value(initial_guess, model, dataset, device)
        final_func_val = compute_function_value(params_scipy, model, dataset, device)

        # Run Monte Carlo analysis for all parameter sets
        print("Running Monte Carlo simulations...")

        # Process true parameters
        print("  📊 Computing MC moments for true parameters...")
        true_params_dict = dict(zip(param_names, true_vals.tolist()))
        Rt_true = mc(
            true_vals[0], true_vals[1], true_vals[2], true_vals[3], true_vals[4]
        )
        true_moments = compute_moments(Rt_true, TN)

        # Process initial guess parameters
        print("  📊 Computing MC moments for initial guess parameters...")
        initial_params_dict = dict(zip(param_names, initial_guess.tolist()))
        Rt_initial = mc(
            initial_guess[0],
            initial_guess[1],
            initial_guess[2],
            initial_guess[3],
            initial_guess[4],
        )
        initial_moments = compute_moments(Rt_initial, TN)

        # Process final calibrated parameters
        print("  📊 Computing MC moments for final calibrated parameters...")
        final_params_dict = dict(zip(param_names, params_scipy.tolist()))
        Rt_final = mc(
            params_scipy[0],
            params_scipy[1],
            params_scipy[2],
            params_scipy[3],
            params_scipy[4],
        )
        final_moments = compute_moments(Rt_final, TN)

        params_dict = {
            "dataset_info": {
                "dataset_path": dataset_path,
                "asset_path": asset_path,
                "params_path": params_path,
                "dataset_name": dataset_name,
            },
            "parameter_sets": {
                "true_parameters": {
                    "parameters": true_params_dict,
                    "function_value": true_func_val,
                    "moments_by_time": true_moments,
                },
                "initial_guess_parameters": {
                    "parameters": initial_params_dict,
                    "function_value": initial_func_val,
                    "moments_by_time": initial_moments,
                },
                "final_calibrated_parameters": {
                    "parameters": final_params_dict,
                    "function_value": final_func_val,
                    "moments_by_time": final_moments,
                },
            },
            "metrics": {
                "l2_error": float(l2_error),
                "persistence": float(persistence),
                "is_stationary": bool(persistence < 1.0),
                "omega_positive": bool(omega > 0),
            },
            "convergence_history": history_scipy,
        }

        # Save individual dataset results
        individual_filename = (
            results_dir / f"params_{dataset_name}_l2_{l2_error:.8f}.json"
        )
        with open(individual_filename, "w") as f:
            json.dump(params_dict, f, indent=2, cls=NumpyEncoder)

        print(f"✅ Parameters saved to {individual_filename}")

        # Store in combined results
        all_results[f"dataset_{idx + 1}_{dataset_name}"] = params_dict

    except Exception as e:
        print(f"❌ Calibration failed for dataset {idx + 1}: {e}")
        import traceback

        traceback.print_exc()

        # Store error info
        all_results[f"dataset_{idx + 1}_{Path(dataset_path).stem}"] = {
            "dataset_info": {
                "dataset_path": dataset_path,
                "asset_path": asset_path,
                "params_path": params_path,
            },
            "error": str(e),
            "status": "failed",
        }

# Save combined results (this now includes MC moments)
combined_filename = results_dir / "mc_moments_all_datasets.json"
with open(combined_filename, "w") as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)

print(f"\n🎉 Multi-dataset calibration and Monte Carlo analysis completed!")
print(f"📁 All results saved to {results_dir}")
print(f"📊 Unified results with MC moments: {combined_filename}")

# Summary statistics
successful_runs = sum(
    1
    for result in all_results.values()
    if "error" not in result and "final_calibrated_parameters" in result
)
total_runs = len(all_results)

print(f"\n📈 Summary:")
print(f"  Total datasets processed: {total_runs}")
print(f"  Successful calibrations: {successful_runs}")
print(f"  Failed calibrations: {total_runs - successful_runs}")

if successful_runs > 0:
    # Calculate average L2 error across successful runs
    l2_errors = [
        result["metrics"]["l2_error"]
        for result in all_results.values()
        if "metrics" in result
    ]
    if l2_errors:
        avg_l2_error = np.mean(l2_errors)
        print(f"  Average L2 error: {avg_l2_error:.8f}")
        print(f"  Best L2 error: {min(l2_errors):.8f}")
        print(f"  Worst L2 error: {max(l2_errors):.8f}")

# Print sample Monte Carlo results for verification
for dataset_key, dataset_result in all_results.items():
    if (
        "parameter_sets" in dataset_result
        and "true_parameters" in dataset_result["parameter_sets"]
    ):
        true_params = dataset_result["parameter_sets"]["true_parameters"]
        if "moments_by_time" in true_params:
            print(f"\n📊 Sample MC results for {dataset_key} true parameters at t=252:")
            moments_252 = true_params["moments_by_time"]["252"]
            print(f"  Mean: {moments_252['mean']:.6f}")
            print(f"  Variance: {moments_252['variance']:.6f}")
            print(f"  Skewness: {moments_252['skewness']:.6f}")
            print(f"  Kurtosis: {moments_252['kurtosis']:.6f}")
            print(f"  Function value: {true_params.get('function_value', 'N/A')}")
            break
