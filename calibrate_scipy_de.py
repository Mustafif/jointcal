import json

import numpy as np
import torch
from scipy.optimize import differential_evolution

from cal_loss import Calibration_Loss
from hn import HestonNandiGARCH

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "mps:0")


def project_parameters(params):
    """Project parameters to valid domain"""
    if isinstance(params, torch.Tensor):
        omega = torch.clamp(params[0], min=1e-8)
        alpha = torch.clamp(params[1], min=0.0, max=1.0)
        beta = torch.clamp(params[2], min=0.0, max=1.0)
        gamma = params[3]  # can be negative
        lambda_param = params[4]  # no constraint
        return torch.stack([omega, alpha, beta, gamma, lambda_param])
    else:
        # Handle numpy arrays
        omega = np.clip(params[0], a_min=1e-8, a_max=None)
        alpha = np.clip(params[1], a_min=0.0, a_max=1.0)
        beta = np.clip(params[2], a_min=0.0, a_max=1.0)
        gamma = params[3]  # can be negative
        lambda_param = params[4]  # no constraint
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

    # Define parameter bounds for SciPy DE
    # [omega, alpha, beta, gamma, lambda]
    bounds = [
        (1e-7, 1e-6),  # omega: positive, small
        (1.15e-6, 1.36e-6),  # alpha: small, positive
        (0.7, 0.99),  # beta: close to 1
        (0, 10),  # gamma: leverage effect
        (0.2, 0.6),  # lambda: risk premium
    ]

    print(f"Parameter bounds: {bounds}")

    # Track convergence history
    convergence_history = []
    iteration_count = [0]  # Use list to make it mutable in callback

    def objective_function(params):
        """
        Objective function for SciPy DE optimization

        Args:
            params: numpy array of parameters [omega, alpha, beta, gamma, lambda]

        Returns:
            float: Loss value
        """
        # Convert to torch tensor
        params_tensor = torch.tensor(params, device=device, dtype=torch.float32)

        # Project parameters to valid domain
        params_proj = project_parameters(params_tensor)

        # Compute calibration loss
        with torch.no_grad():
            loss = Calibration_Loss(
                params_proj, all_returns, sigma_all, model, X_all, N, M
            )

        return loss.cpu().item()

    def callback(xk, convergence=None):
        """Callback function to track convergence"""
        iteration_count[0] += 1
        current_loss = objective_function(xk)
        convergence_history.append(current_loss)

        if iteration_count[0] % 50 == 0:
            print(
                f"Iteration {iteration_count[0]:4d} | Best fitness: {current_loss:.6f} | "
                f"Params: omega={xk[0]:.2e}, alpha={xk[1]:.2e}, beta={xk[2]:.3f}, "
                f"gamma={xk[3]:.2f}, lambda={xk[4]:.2f}"
            )

    print("Starting SciPy DE optimization...")

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
        init="latinhypercube",  # Good initialization strategy
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
