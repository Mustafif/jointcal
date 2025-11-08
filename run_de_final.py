#!/usr/bin/env python3
"""
Final optimized Differential Evolution GARCH calibration that closely matches the gradient-based approach.
This version uses the same initialization strategy as the gradient method and improved parameter handling.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path

# Import our modules
from dataset2 import cal_dataset
from cal_loss import Calibration_Loss

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"


class OptimizedDifferentialEvolution:
    """Optimized DE that mimics gradient-based initialization and constraints"""

    def __init__(self, bounds, popsize=30, mutation=0.5, crossover=0.9, seed=None, device=None):
        self.bounds = torch.tensor(bounds, device=device, dtype=torch.float32)
        self.popsize = popsize

        # Handle mutation as either single value or range
        if isinstance(mutation, (tuple, list)):
            self.mutation_range = mutation
            self.mutation = None  # Will be sampled each time
        else:
            self.mutation_range = None
            self.mutation = mutation

        self.crossover = crossover
        self.device = device
        self.num_params = len(bounds)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.population = None
        self.fitness = torch.full((popsize,), float('inf'), device=device)
        self.best_idx = 0
        self.best_fitness = float('inf')
        self.best_params = None

    def initialize_with_gradient_strategy(self, dataset_target=None):
        """Initialize population using the same strategy as gradient method"""
        self.population = torch.zeros(self.popsize, self.num_params, device=self.device)

        # Strategy 1: Use dataset target as first individual if available
        if dataset_target is not None:
            self.population[0] = dataset_target.clone().to(self.device)
        else:
            # Fallback: typical GARCH parameters
            self.population[0] = torch.tensor([1e-6, 0.01, 0.85, 0.1, 0.0], device=self.device)

        # Strategy 2: Small perturbations around the first individual (50% of population)
        n_perturb = self.popsize // 2
        for i in range(1, n_perturb):
            noise_scale = 0.1 + 0.2 * torch.rand(1, device=self.device)  # 10-30% noise
            noise = noise_scale * torch.randn(self.num_params, device=self.device)
            perturbed = self.population[0] * (1 + noise)
            self.population[i] = torch.clamp(perturbed, self.bounds[:, 0], self.bounds[:, 1])

        # Strategy 3: Diverse sampling in remaining population
        for i in range(n_perturb, self.popsize):
            # Use log-uniform for omega, uniform for others
            for j in range(self.num_params):
                low, high = self.bounds[j, 0].item(), self.bounds[j, 1].item()
                if j == 0 and low > 0:  # omega - log-uniform
                    log_low, log_high = np.log(low), np.log(high)
                    log_val = log_low + torch.rand(1, device=self.device) * (log_high - log_low)
                    self.population[i, j] = torch.exp(log_val)
                else:  # uniform sampling
                    self.population[i, j] = low + torch.rand(1, device=self.device) * (high - low)

    def _enforce_bounds(self, candidate):
        """Enforce bounds exactly like gradient method"""
        omega = torch.clamp(candidate[0], min=1e-8)
        alpha = torch.clamp(candidate[1], min=0.0, max=1.0)
        beta = torch.clamp(candidate[2], min=0.0, max=1.0)
        gamma = candidate[3]  # No constraint
        lambda_param = candidate[4]  # No constraint

        return torch.stack([omega, alpha, beta, gamma, lambda_param])

    def _mutate_and_crossover(self, target_idx):
        """DE/rand/1/bin with adaptive parameters"""
        # Select three random indices
        candidates = list(range(self.popsize))
        candidates.remove(target_idx)
        r1, r2, r3 = torch.randperm(len(candidates), device=self.device)[:3]
        r1, r2, r3 = candidates[r1], candidates[r2], candidates[r3]

        # Mutation with current best bias
        # Sample mutation factor if range is provided
        if self.mutation_range is not None:
            mutation_factor = torch.rand(1, device=self.device) * (self.mutation_range[1] - self.mutation_range[0]) + self.mutation_range[0]
        else:
            mutation_factor = self.mutation

        if torch.rand(1, device=self.device) < 0.3:  # 30% chance to use best
            mutant = self.best_params + mutation_factor * (self.population[r1] - self.population[r2])
        else:  # Standard DE/rand/1
            mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])

        mutant = self._enforce_bounds(mutant)

        # Binomial crossover
        target = self.population[target_idx]
        trial = target.clone()

        crossover_mask = torch.rand(self.num_params, device=self.device) < self.crossover

        # Ensure at least one parameter is from mutant
        if not crossover_mask.any():
            j_rand = torch.randint(0, self.num_params, (1,), device=self.device)
            crossover_mask[j_rand] = True

        trial[crossover_mask] = mutant[crossover_mask]

        return trial

    def optimize(self, objective_func, max_iter=500, tolerance=1e-8, verbose=True):
        """Run optimized differential evolution"""
        convergence_history = []

        # Evaluate initial population
        for i in range(self.popsize):
            self.fitness[i] = objective_func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i].item()
                self.best_idx = i
                self.best_params = self.population[i].clone()

        if verbose:
            print(f"Initial best fitness: {self.best_fitness:.6f}")
            print(f"Initial best params: {self.best_params.cpu().numpy()}")

        stagnation_counter = 0
        prev_best = self.best_fitness

        for iteration in range(max_iter):
            improvements = 0

            for i in range(self.popsize):
                trial = self._mutate_and_crossover(i)
                trial_fitness = objective_func(trial)

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial.clone()
                    self.fitness[i] = trial_fitness
                    improvements += 1

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness.item()
                        self.best_idx = i
                        self.best_params = trial.clone()

            convergence_history.append(self.best_fitness)

            # Check for stagnation
            improvement = abs(prev_best - self.best_fitness) / max(abs(prev_best), 1e-15)
            if improvement < tolerance:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            prev_best = self.best_fitness

            if verbose and (iteration % 25 == 0 or iteration == max_iter - 1):
                print(f"Iteration {iteration+1:4d}/{max_iter} | Best: {self.best_fitness:.6f} | "
                      f"Improvements: {improvements:2d}/{self.popsize} | Params: {self.best_params.cpu().numpy()}")

            # Early stopping
            if stagnation_counter > 25:
                if verbose:
                    print(f"Converged after {iteration+1} iterations (stagnation)")
                break

        return self.best_params, self.best_fitness, convergence_history


def calibrate_de_final(model, dataset, popsize=30, max_iter=200, mutation=0.5,
                      crossover=0.9, seed=42, device=None):
    """Final optimized DE calibration matching gradient method approach"""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Use the same preprocessing as gradient method
    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"Starting final DE calibration: {M} options, {N} returns")

    # Use similar bounds to gradient method but allow more flexibility
    bounds = [
        (1e-8, 1e-3),      # omega: same as gradient method
        (0.0, 1.0),        # alpha: same as gradient method
        (0.0, 1.0),        # beta: same as gradient method
        (-10.0, 10.0),     # gamma: same as gradient method
        (-1.0, 1.0)        # lambda: same as gradient method
    ]

    de = OptimizedDifferentialEvolution(
        bounds=bounds,
        popsize=popsize,
        mutation=mutation,
        crossover=crossover,
        seed=seed,
        device=device
    )

    # Initialize using dataset target if available (like gradient method)
    dataset_target = None
    if hasattr(dataset, 'target'):
        dataset_target = dataset.target.clone()
        print(f"Using dataset target for initialization: {dataset_target.numpy()}")

    de.initialize_with_gradient_strategy(dataset_target)

    def objective_function(params):
        """Objective function matching gradient method exactly"""
        try:
            # Apply the same projection as gradient method
            omega = torch.clamp(params[0], min=1e-8)
            alpha = torch.clamp(params[1], min=0.0, max=1.0)
            beta = torch.clamp(params[2], min=0.0, max=1.0)
            gamma = params[3]
            lambda_param = params[4]

            projected_params = torch.stack([omega, alpha, beta, gamma, lambda_param])

            # Check for numerical issues
            if torch.any(torch.isnan(projected_params)) or torch.any(torch.isinf(projected_params)):
                return torch.tensor(1e10, device=device, dtype=torch.float32)

            # Use the exact same loss as gradient method (negative log-likelihood)
            with torch.no_grad():
                loss = -1 * Calibration_Loss(projected_params, all_returns, sigma_all,
                                           model, X_all, N, M)

            # Handle invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(1e8, device=device, dtype=torch.float32)

            # Return positive loss for minimization (gradient method minimizes negative loss)
            return -loss  # Convert to positive for DE minimization

        except Exception as e:
            return torch.tensor(1e12, device=device, dtype=torch.float32)

    # Run optimization
    best_params, best_fitness, history = de.optimize(
        objective_function,
        max_iter=max_iter,
        tolerance=1e-8,
        verbose=True
    )

    # Apply final projection like gradient method
    with torch.no_grad():
        omega = torch.clamp(best_params[0], min=1e-8)
        alpha = torch.clamp(best_params[1], min=0.0, max=1.0)
        beta = torch.clamp(best_params[2], min=0.0, max=1.0)
        gamma = best_params[3]
        lambda_param = best_params[4]
        best_params = torch.stack([omega, alpha, beta, gamma, lambda_param])

    print(f"\n✅ Final DE calibration complete")
    print(f"Final loss: {best_fitness:.6f}")

    return best_params.detach().cpu().numpy(), history


def main():
    """Main function"""

    print("🎯 Final Optimized Differential Evolution GARCH Calibration")
    print("=" * 70)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    try:
        # Load model and dataset
        if not Path(MODEL_PATH).exists():
            print(f"❌ Model file not found: {MODEL_PATH}")
            return 1

        print(f"\n📦 Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        print("✅ Model loaded")

        print(f"📊 Loading dataset...")
        dataset = cal_dataset(
            "joint_dataset/scalable_hn_dataset_250x60.csv",
            "joint_dataset/assetprices.csv"
        )
        print(f"✅ Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

        # True parameters for comparison
        true_params = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2])
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        print(f"\n🎯 Target parameters:")
        for name, val in zip(param_names, true_params):
            print(f"   {name}: {val}")

        # Run final DE calibration
        print(f"\n🚀 Starting final calibration...")
        start_time = time.time()

        calibrated_params, history = calibrate_de_final(
            model=model,
            dataset=dataset,
            popsize=30,        # Smaller population for efficiency
            max_iter=200,      # Fewer iterations
            mutation=0.5,      # Conservative mutation
            crossover=0.9,     # High crossover for exploitation
            seed=42,
            device=device
        )

        elapsed = time.time() - start_time

        # Results analysis
        print(f"\n✅ Completed in {elapsed:.1f} seconds")

        l2_error = np.linalg.norm(calibrated_params - true_params, ord=2)
        abs_errors = np.abs(calibrated_params - true_params)
        rel_errors = abs_errors / np.abs(true_params) * 100

        print(f"\n📊 Final Results:")
        print(f"{'Parameter':<10} {'True':<12} {'Calibrated':<12} {'Abs Error':<12} {'Rel Error %':<12}")
        print("-" * 70)

        for i, name in enumerate(param_names):
            print(f"{name:<10} {true_params[i]:<12.6f} {calibrated_params[i]:<12.6f} "
                  f"{abs_errors[i]:<12.6f} {rel_errors[i]:<12.2f}")

        print(f"\nOverall L2 Error: {l2_error:.6f}")

        # Validation
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta
        omega_positive = calibrated_params[0] > 0
        stationary = persistence < 1.0

        print(f"\n🔍 Validation:")
        print(f"   ω > 0: {'✅' if omega_positive else '❌'} ({calibrated_params[0]:.8f})")
        print(f"   α + β < 1: {'✅' if stationary else '❌'} ({persistence:.6f})")

        if stationary:
            unconditional_var = calibrated_params[0] / (1 - persistence)
            empirical_var = dataset.returns.var().item()
            print(f"   Theoretical var: {unconditional_var:.8f}")
            print(f"   Empirical var: {empirical_var:.8f}")
            print(f"   Variance ratio: {unconditional_var/empirical_var:.4f}")

        # Save results
        results = {
            'method': 'final_optimized_differential_evolution',
            'configuration': {
                'popsize': 30,
                'max_iter': 200,
                'mutation': 0.5,
                'crossover': 0.9,
                'seed': 42
            },
            'true_parameters': dict(zip(param_names, true_params.tolist())),
            'calibrated_parameters': dict(zip(param_names, calibrated_params.tolist())),
            'errors': {
                'l2_error': float(l2_error),
                'absolute_errors': dict(zip(param_names, abs_errors.tolist())),
                'relative_errors_percent': dict(zip(param_names, rel_errors.tolist()))
            },
            'validation': {
                'omega_positive': bool(omega_positive),
                'stationary': bool(stationary),
                'persistence': float(persistence)
            },
            'convergence_history': history,
            'timing': {
                'total_seconds': elapsed
            }
        }

        output_file = 'final_de_calibration_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {output_file}")

        # Create convergence plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Main convergence plot
            plt.subplot(2, 2, 1)
            plt.plot(history, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('Final DE Convergence')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            # Parameter comparison
            plt.subplot(2, 2, 2)
            x = np.arange(len(param_names))
            width = 0.35
            plt.bar(x - width/2, true_params, width, label='True', alpha=0.7)
            plt.bar(x + width/2, calibrated_params, width, label='Calibrated', alpha=0.7)
            plt.xlabel('Parameters')
            plt.ylabel('Values')
            plt.title('Parameter Comparison')
            plt.xticks(x, param_names)
            plt.legend()
            plt.yscale('log')

            # Error plot
            plt.subplot(2, 2, 3)
            plt.bar(param_names, abs_errors, alpha=0.7, color='red')
            plt.xlabel('Parameters')
            plt.ylabel('Absolute Error')
            plt.title('Absolute Errors')
            plt.yscale('log')
            plt.xticks(rotation=45)

            # Relative error plot
            plt.subplot(2, 2, 4)
            plt.bar(param_names, rel_errors, alpha=0.7, color='orange')
            plt.xlabel('Parameters')
            plt.ylabel('Relative Error (%)')
            plt.title('Relative Errors')
            plt.yscale('log')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plot_file = 'final_de_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📈 Analysis plots saved to: {plot_file}")

        except ImportError:
            print("📈 Matplotlib not available for plotting")

        # Summary with quality assessment
        print(f"\n🎉 Final DE Calibration Summary:")
        print(f"   ✅ L2 error: {l2_error:.6f}")
        print(f"   ⏱️ Time: {elapsed:.1f} seconds")
        print(f"   🔍 Constraints: {'All satisfied' if omega_positive and stationary else 'Some violated'}")

        # Quality assessment
        if l2_error < 0.1:
            quality = "Excellent"
        elif l2_error < 0.5:
            quality = "Good"
        elif l2_error < 2.0:
            quality = "Fair"
        else:
            quality = "Poor"

        print(f"   📈 Calibration quality: {quality}")

        return 0

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
