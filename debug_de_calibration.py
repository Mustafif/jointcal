#!/usr/bin/env python3
"""
Debug version of DE calibration to diagnose convergence issues
"""

import torch
import numpy as np
import time
import json
from pathlib import Path

# Import our modules
from dataset2 import cal_dataset
from calibrate2_de import DifferentialEvolution, project_parameters
from cal_loss import Calibration_Loss

def debug_objective_function(model, dataset, device):
    """Debug the objective function with different parameter values"""

    print("🔍 Debugging Objective Function")
    print("=" * 50)

    # Precompute tensors
    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    # Test different parameter sets
    test_params = [
        torch.tensor([1e-6, 1.33e-6, 0.8, 5.0, 0.2], device=device),  # True params
        torch.tensor([1e-5, 0.01, 0.85, 0.1, 0.0], device=device),    # Typical GARCH
        torch.tensor([1e-4, 0.1, 0.8, 1.0, 0.1], device=device),      # Higher values
        torch.tensor([1e-8, 0.0, 0.0, -0.1, -1.0], device=device),    # Boundary values
    ]

    param_names = ["True", "Typical", "Higher", "Boundary"]

    print(f"Testing {len(test_params)} parameter sets:")
    print(f"{'Set':<8} {'Omega':<10} {'Alpha':<10} {'Beta':<10} {'Gamma':<10} {'Lambda':<10} {'Loss':<15}")
    print("-" * 80)

    for i, (params, name) in enumerate(zip(test_params, param_names)):
        try:
            with torch.no_grad():
                loss = -1 * Calibration_Loss(params, all_returns, sigma_all, model, X_all, N, M)

            print(f"{name:<8} {params[0]:<10.2e} {params[1]:<10.6f} {params[2]:<10.6f} "
                  f"{params[3]:<10.6f} {params[4]:<10.6f} {loss.item():<15.2e}")

        except Exception as e:
            print(f"{name:<8} ERROR: {e}")

    return test_params, param_names

def debug_bounds_sensitivity(model, dataset, device):
    """Test how sensitive the objective is to parameter bounds"""

    print("\n🎯 Testing Bounds Sensitivity")
    print("=" * 50)

    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    # Test omega values near bounds
    omega_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    base_params = torch.tensor([1e-6, 0.01, 0.85, 0.1, 0.0], device=device)

    print("Omega sensitivity:")
    print(f"{'Omega':<12} {'Loss':<15} {'Valid':<8}")
    print("-" * 35)

    for omega in omega_values:
        test_params = base_params.clone()
        test_params[0] = omega

        try:
            with torch.no_grad():
                loss = -1 * Calibration_Loss(test_params, all_returns, sigma_all, model, X_all, N, M)

            is_valid = not (torch.isnan(loss) or torch.isinf(loss))
            print(f"{omega:<12.2e} {loss.item():<15.2e} {'✅' if is_valid else '❌':<8}")

        except Exception as e:
            print(f"{omega:<12.2e} ERROR: {str(e)[:20]:<15} ❌")

def debug_gradient_comparison(model, dataset, device):
    """Compare with gradient-based initial values"""

    print("\n📊 Gradient Method Comparison")
    print("=" * 50)

    # Use the same initialization as gradient method
    if hasattr(dataset, 'target'):
        grad_init = dataset.target.clone().to(device)
        print(f"Dataset target params: {grad_init.cpu().numpy()}")

        X_all = dataset.X.to(device)
        sigma_all = dataset.sigma.to(device)
        all_returns = dataset.returns.to(device)
        N = len(all_returns)
        M = len(dataset)

        try:
            with torch.no_grad():
                loss = -1 * Calibration_Loss(grad_init, all_returns, sigma_all, model, X_all, N, M)

            print(f"Loss at gradient init: {loss.item():.2e}")

        except Exception as e:
            print(f"Error at gradient init: {e}")
    else:
        print("No target parameters found in dataset")

def improved_de_calibration(model, dataset, device):
    """Improved DE calibration with better bounds and initialization"""

    print("\n🚀 Improved DE Calibration")
    print("=" * 50)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"Dataset info: {M} options, {N} returns")

    # Improved bounds - wider ranges
    bounds = [
        (1e-8, 1e-2),     # omega: wider upper bound
        (1e-8, 0.5),      # alpha: non-zero lower bound
        (0.1, 0.99),      # beta: avoid zero
        (-5.0, 5.0),      # gamma: smaller range
        (-0.5, 0.5)       # lambda: smaller range
    ]

    print("Improved bounds:")
    param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
    for name, (low, high) in zip(param_names, bounds):
        print(f"  {name}: [{low:.2e}, {high:.2e}]")

    def objective_function(params):
        """Improved objective with error handling"""
        try:
            # Project parameters
            params_proj = project_parameters(params)

            # Check for valid parameters
            if torch.any(torch.isnan(params_proj)) or torch.any(torch.isinf(params_proj)):
                return torch.tensor(1e10, device=device)

            # Compute loss
            with torch.no_grad():
                loss = -1 * Calibration_Loss(params_proj, all_returns, sigma_all, model, X_all, N, M)

            # Handle invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(1e10, device=device)

            return loss

        except Exception as e:
            print(f"Error in objective: {e}")
            return torch.tensor(1e10, device=device)

    # Initialize DE with improved settings
    de = DifferentialEvolution(
        bounds=bounds,
        popsize=30,          # Smaller population for testing
        mutation=0.5,        # Lower mutation for more gradual changes
        crossover=0.8,       # Higher crossover
        seed=42,
        device=device
    )

    print("\nDE Settings:")
    print(f"  Population: {de.popsize}")
    print(f"  Mutation: {de.mutation}")
    print(f"  Crossover: {de.crossover}")

    # Run optimization
    best_params, best_fitness, history = de.optimize(
        objective_function,
        max_iter=100,  # Fewer iterations for testing
        tolerance=1e-8,
        verbose=True
    )

    # Final projection
    best_params = project_parameters(best_params)

    return best_params.detach().cpu().numpy(), history

def main():
    """Main debug function"""

    print("🐛 DE Calibration Debug Session")
    print("=" * 60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    try:
        # Load model and dataset
        MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()

        dataset = cal_dataset(
            "joint_dataset/scalable_hn_dataset_250x60.csv",
            "joint_dataset/assetprices.csv"
        )

        print(f"📊 Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

        # Debug steps
        debug_objective_function(model, dataset, device)
        debug_bounds_sensitivity(model, dataset, device)
        debug_gradient_comparison(model, dataset, device)

        # Try improved DE
        print("\n" + "="*60)
        calibrated_params, history = improved_de_calibration(model, dataset, device)

        # Compare results
        true_params = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2])
        error = np.linalg.norm(calibrated_params - true_params, ord=2)

        print(f"\n📊 Improved Results:")
        print(f"True params:       {true_params}")
        print(f"Calibrated params: {calibrated_params}")
        print(f"L2 error:          {error:.6f}")

        # Save debug results
        debug_results = {
            'improved_calibration': {
                'parameters': calibrated_params.tolist(),
                'l2_error': float(error),
                'convergence_history': history
            },
            'true_parameters': true_params.tolist()
        }

        with open('debug_de_results.json', 'w') as f:
            json.dump(debug_results, f, indent=2)

        print(f"\n📁 Debug results saved to: debug_de_results.json")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
