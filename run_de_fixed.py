#!/usr/bin/env python3
"""
Fixed Differential Evolution GARCH calibration with proper loss scaling and bounds handling.
This version addresses the issues found in the debug session.
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

class FixedDifferentialEvolution:
    """Fixed Differential Evolution with proper loss scaling and bounds"""

    def __init__(self, bounds, popsize=50, mutation=0.8, crossover=0.7, seed=None, device=None):
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

        # Initialize population with better strategy
        self.population = self._initialize_population_improved()
        self.fitness = torch.full((popsize,), float('inf'), device=device)
        self.best_idx = 0
        self.best_fitness = float('inf')
        self.best_params = None

    def _initialize_population_improved(self):
        """Initialize population with focus on reasonable GARCH values"""
        pop = torch.zeros(self.popsize, self.num_params, device=self.device)

        # Known good parameter ranges for GARCH models
        typical_ranges = [
            (1e-6, 1e-4),    # omega: small positive values
            (0.01, 0.15),    # alpha: typical ARCH coefficient
            (0.7, 0.95),     # beta: high persistence
            (-1.0, 1.0),     # gamma: moderate asymmetry
            (-0.1, 0.1)      # lambda: small risk premium
        ]

        for i in range(self.popsize):
            for j in range(self.num_params):
                if j < len(typical_ranges):
                    # Use typical ranges for most of population
                    low, high = typical_ranges[j]
                    # Ensure we stay within actual bounds
                    low = max(low, self.bounds[j, 0].item())
                    high = min(high, self.bounds[j, 1].item())

                    if j == 0:  # omega - use log-uniform sampling
                        log_low, log_high = np.log(low), np.log(high)
                        log_val = log_low + torch.rand(1, device=self.device) * (log_high - log_low)
                        pop[i, j] = torch.exp(log_val)
                    else:
                        pop[i, j] = low + torch.rand(1, device=self.device) * (high - low)
                else:
                    # Fall back to uniform sampling within bounds
                    low, high = self.bounds[j, 0], self.bounds[j, 1]
                    pop[i, j] = low + torch.rand(1, device=self.device) * (high - low)

        return pop

    def _enforce_bounds(self, candidate):
        """Enforce bounds with clamping"""
        return torch.clamp(candidate, self.bounds[:, 0], self.bounds[:, 1])

    def _mutate_and_crossover(self, target_idx):
        """Standard DE/rand/1/bin strategy"""
        # Select three random indices
        candidates = list(range(self.popsize))
        candidates.remove(target_idx)
        r1, r2, r3 = torch.randperm(len(candidates), device=self.device)[:3]
        r1, r2, r3 = candidates[r1], candidates[r2], candidates[r3]

        # Mutation
        # Sample mutation factor if range is provided
        if self.mutation_range is not None:
            mutation_factor = torch.rand(1, device=self.device) * (self.mutation_range[1] - self.mutation_range[0]) + self.mutation_range[0]
        else:
            mutation_factor = self.mutation

        mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])
        mutant = self._enforce_bounds(mutant)

        # Crossover
        target = self.population[target_idx]
        trial = target.clone()

        crossover_mask = torch.rand(self.num_params, device=self.device) < self.crossover

        # Ensure at least one parameter is from mutant
        if not crossover_mask.any():
            j_rand = torch.randint(0, self.num_params, (1,), device=self.device)
            crossover_mask[j_rand] = True

        trial[crossover_mask] = mutant[crossover_mask]

        return trial

    def optimize(self, objective_func, max_iter=1000, tolerance=1e-6, verbose=True):
        """Run differential evolution optimization"""
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

        stagnation_counter = 0
        prev_best = self.best_fitness

        for iteration in range(max_iter):
            improvements = 0

            for i in range(self.popsize):
                trial = self._mutate_and_crossover(i)
                trial_fitness = objective_func(trial)

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    improvements += 1

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness.item()
                        self.best_idx = i
                        self.best_params = trial.clone()

            convergence_history.append(self.best_fitness)

            # Check for stagnation
            improvement = abs(prev_best - self.best_fitness) / max(abs(prev_best), 1e-10)
            if improvement < tolerance:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            prev_best = self.best_fitness

            if verbose and (iteration % 50 == 0 or iteration == max_iter - 1):
                print(f"Iteration {iteration+1:4d}/{max_iter} | Best: {self.best_fitness:.6f} | "
                      f"Improvements: {improvements}/{self.popsize}")

            # Early stopping
            if stagnation_counter > 30:
                if verbose:
                    print(f"Converged after {iteration+1} iterations")
                break

        return self.best_params, self.best_fitness, convergence_history


def project_parameters(params):
    """Project parameters to valid GARCH domain"""
    omega = torch.clamp(params[0], min=1e-8, max=1e-3)
    alpha = torch.clamp(params[1], min=1e-6, max=0.99)
    beta = torch.clamp(params[2], min=0.01, max=0.99)
    gamma = torch.clamp(params[3], min=-5.0, max=5.0)
    lambda_param = torch.clamp(params[4], min=-0.5, max=0.5)

    # Enforce stationarity constraint: alpha + beta < 1
    persistence = alpha + beta
    if persistence >= 1.0:
        scale_factor = 0.98 / persistence
        alpha = alpha * scale_factor
        beta = beta * scale_factor

    return torch.stack([omega, alpha, beta, gamma, lambda_param])


def calibrate_de_fixed(model, dataset, popsize=50, max_iter=500, mutation=0.8,
                      crossover=0.7, seed=42, device=None):
    """Fixed DE calibration with proper loss handling"""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Precompute tensors
    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"Starting fixed DE calibration: {M} options, {N} returns")

    # Better parameter bounds based on GARCH literature
    bounds = [
        (1e-8, 1e-3),     # omega: unconditional variance component
        (1e-6, 0.2),      # alpha: ARCH coefficient, avoid zero
        (0.3, 0.99),      # beta: GARCH coefficient, ensure some persistence
        (-3.0, 3.0),      # gamma: asymmetry parameter
        (-0.2, 0.2)       # lambda: risk premium
    ]

    print("Parameter bounds:")
    param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
    for name, (low, high) in zip(param_names, bounds):
        print(f"  {name}: [{low:.2e}, {high:.2e}]")

    de = FixedDifferentialEvolution(
        bounds=bounds,
        popsize=popsize,
        mutation=mutation,
        crossover=crossover,
        seed=seed,
        device=device
    )

    def scaled_objective_function(params):
        """Objective function with proper scaling and error handling"""
        try:
            # Project to valid domain
            params_proj = project_parameters(params)

            # Check for numerical issues
            if torch.any(torch.isnan(params_proj)) or torch.any(torch.isinf(params_proj)):
                return torch.tensor(1e6, device=device, dtype=torch.float32)

            # Check stationarity
            alpha, beta = params_proj[1], params_proj[2]
            if alpha + beta >= 1.0:
                return torch.tensor(1e5, device=device, dtype=torch.float32)

            # Compute raw loss
            with torch.no_grad():
                raw_loss = -1 * Calibration_Loss(params_proj, all_returns, sigma_all,
                                               model, X_all, N, M)

            # Handle numerical issues
            if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                return torch.tensor(1e4, device=device, dtype=torch.float32)

            # Scale the loss to a more reasonable range
            # The raw loss is typically very large negative numbers (-1e9 to -1e11)
            # We want to minimize, so we take the negative and scale
            scaled_loss = -raw_loss / 1e9  # Convert to range roughly 1-100

            # Ensure positive value for minimization
            if scaled_loss <= 0:
                return torch.tensor(1e3, device=device, dtype=torch.float32)

            return scaled_loss.float()

        except Exception as e:
            print(f"Error in objective: {e}")
            return torch.tensor(1e7, device=device, dtype=torch.float32)

    # Run optimization
    best_params, best_fitness, history = de.optimize(
        scaled_objective_function,
        max_iter=max_iter,
        tolerance=1e-6,
        verbose=True
    )

    # Final projection
    best_params = project_parameters(best_params)

    print(f"\n✅ Fixed DE calibration complete")
    print(f"Final scaled fitness: {best_fitness:.6f}")

    return best_params.detach().cpu().numpy(), history


def main():
    """Main function"""

    print("🔧 Fixed Differential Evolution GARCH Calibration")
    print("=" * 65)

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
            print(f"   {name}: {val:.6f}")

        # Run fixed DE calibration
        print(f"\n🚀 Starting calibration...")
        start_time = time.time()

        calibrated_params, history = calibrate_de_fixed(
            model=model,
            dataset=dataset,
            popsize=50,
            max_iter=300,
            mutation=0.6,  # Slightly lower for more stability
            crossover=0.8,
            seed=42,
            device=device
        )

        elapsed = time.time() - start_time

        # Results analysis
        print(f"\n✅ Completed in {elapsed:.1f} seconds")

        l2_error = np.linalg.norm(calibrated_params - true_params, ord=2)
        abs_errors = np.abs(calibrated_params - true_params)
        rel_errors = abs_errors / np.abs(true_params) * 100

        print(f"\n📊 Results:")
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
            'method': 'fixed_differential_evolution',
            'configuration': {
                'popsize': 50,
                'max_iter': 300,
                'mutation': 0.6,
                'crossover': 0.8,
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

        output_file = 'fixed_de_calibration_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {output_file}")

        # Create convergence plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(history, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Scaled Objective Value')
            plt.title('Fixed DE Convergence')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            plot_file = 'fixed_de_convergence.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📈 Convergence plot saved to: {plot_file}")

        except ImportError:
            print("📈 Matplotlib not available for plotting")

        # Summary
        print(f"\n🎉 Fixed DE Calibration Summary:")
        print(f"   ✅ L2 error: {l2_error:.6f}")
        print(f"   ⏱️ Time: {elapsed:.1f} seconds")
        print(f"   🔍 Constraints: {'All satisfied' if omega_positive and stationary else 'Some violated'}")

        return 0

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
