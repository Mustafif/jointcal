#!/usr/bin/env python3
"""
Simple working Differential Evolution GARCH calibration with robust objective function.
This version focuses on getting functional results with proper error handling.
"""

import torch
import numpy as np
import time
import json

# Import our modules
from dataset2 import cal_dataset
from cal_loss import Calibration_Loss

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"

class SimpleDifferentialEvolution:
    """Simple, robust Differential Evolution implementation"""

    def __init__(self, bounds, popsize=40, mutation=0.7, crossover=0.8, seed=42, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bounds = torch.tensor(bounds, device=self.device, dtype=torch.float32)
        self.popsize = popsize

        # Handle mutation as either single value or range
        if isinstance(mutation, (tuple, list)):
            self.mutation_range = mutation
            self.mutation = None  # Will be sampled each time
        else:
            self.mutation_range = None
            self.mutation = mutation

        self.crossover = crossover
        self.num_params = len(bounds)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize population
        self.population = self._smart_initialization()
        self.fitness = torch.full((popsize,), float('inf'), device=self.device)
        self.best_params = None
        self.best_fitness = float('inf')

    def _smart_initialization(self):
        """Smart initialization focusing on reasonable GARCH parameter ranges"""
        pop = torch.zeros(self.popsize, self.num_params, device=self.device)

        # Good starting points for GARCH parameters
        good_starts = [
            [1e-5, 0.05, 0.9, 0.1, 0.0],
            [5e-6, 0.02, 0.95, 0.5, 0.05],
            [2e-5, 0.08, 0.85, -0.2, -0.1],
            [1e-6, 0.01, 0.92, 1.0, 0.15],
            [8e-6, 0.03, 0.88, -0.5, 0.02]
        ]

        # Use good starts for first few individuals
        for i in range(min(len(good_starts), self.popsize)):
            base = torch.tensor(good_starts[i], device=self.device)
            # Add small random noise
            noise = 0.1 * torch.randn(self.num_params, device=self.device)
            pop[i] = base * (1 + noise)
            # Clamp to bounds
            pop[i] = torch.clamp(pop[i], self.bounds[:, 0], self.bounds[:, 1])

        # Random initialization for remaining individuals
        for i in range(len(good_starts), self.popsize):
            for j in range(self.num_params):
                low, high = self.bounds[j, 0], self.bounds[j, 1]
                if j == 0:  # omega - use log-uniform
                    log_low, log_high = torch.log(low), torch.log(high)
                    log_val = log_low + torch.rand(1, device=self.device) * (log_high - log_low)
                    pop[i, j] = torch.exp(log_val)
                else:
                    pop[i, j] = low + torch.rand(1, device=self.device) * (high - low)

        return pop

    def _project_to_valid(self, params):
        """Project parameters to valid GARCH domain"""
        omega = torch.clamp(params[0], min=1e-8, max=1e-3)
        alpha = torch.clamp(params[1], min=1e-6, max=0.3)
        beta = torch.clamp(params[2], min=0.1, max=0.99)
        gamma = torch.clamp(params[3], min=-3.0, max=3.0)
        lambda_param = torch.clamp(params[4], min=-0.3, max=0.3)

        # Ensure stationarity
        if alpha + beta >= 1.0:
            scale = 0.98 / (alpha + beta)
            alpha *= scale
            beta *= scale

        return torch.stack([omega, alpha, beta, gamma, lambda_param])

    def _mutate_crossover(self, target_idx):
        """DE mutation and crossover"""
        # Select random indices
        indices = list(range(self.popsize))
        indices.remove(target_idx)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)

        # Mutation: DE/rand/1
        # Sample mutation factor if range is provided
        if self.mutation_range is not None:
            mutation_factor = torch.rand(1, device=self.device) * (self.mutation_range[1] - self.mutation_range[0]) + self.mutation_range[0]
        else:
            mutation_factor = self.mutation

        mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])
        mutant = torch.clamp(mutant, self.bounds[:, 0], self.bounds[:, 1])

        # Crossover
        target = self.population[target_idx]
        trial = target.clone()

        # Binomial crossover
        for j in range(self.num_params):
            if torch.rand(1, device=self.device) < self.crossover:
                trial[j] = mutant[j]

        # Ensure at least one gene from mutant
        if torch.equal(trial, target):
            j = torch.randint(0, self.num_params, (1,), device=self.device)
            trial[j] = mutant[j]

        return trial

    def optimize(self, objective_func, max_iter=150, verbose=True):
        """Run DE optimization"""
        history = []

        # Evaluate initial population
        for i in range(self.popsize):
            self.fitness[i] = objective_func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i].item()
                self.best_params = self.population[i].clone()

        if verbose:
            print(f"Initial best: {self.best_fitness:.6f}")

        stagnation = 0

        for gen in range(max_iter):
            improved = 0

            for i in range(self.popsize):
                trial = self._mutate_crossover(i)
                trial_fitness = objective_func(trial)

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial.clone()
                    self.fitness[i] = trial_fitness
                    improved += 1

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness.item()
                        self.best_params = trial.clone()
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

            history.append(self.best_fitness)

            if verbose and (gen % 25 == 0 or gen == max_iter - 1):
                print(f"Gen {gen+1:3d}: Best={self.best_fitness:.6f}, Improved={improved:2d}")

            # Early stopping
            if stagnation > 50:
                if verbose:
                    print(f"Early stop at generation {gen+1}")
                break

        return self.best_params, self.best_fitness, history


def calibrate_with_de(model, dataset, device):
    """Main DE calibration function"""

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Prepare data
    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"Calibrating with {M} options and {N} returns")

    # Parameter bounds: [omega, alpha, beta, gamma, lambda]
    bounds = [
        (1e-8, 1e-3),   # omega
        (1e-6, 0.3),    # alpha
        (0.1, 0.99),    # beta
        (-3.0, 3.0),    # gamma
        (-0.3, 0.3)     # lambda
    ]

    # Initialize DE
    de = SimpleDifferentialEvolution(
        bounds=bounds,
        popsize=40,
        mutation=0.7,
        crossover=0.8,
        seed=42,
        device=device
    )

    def robust_objective(params):
        """Robust objective function with error handling"""
        try:
            # Project to valid domain
            valid_params = de._project_to_valid(params)

            # Check constraints
            if torch.any(torch.isnan(valid_params)) or torch.any(torch.isinf(valid_params)):
                return torch.tensor(1e8, device=device)

            # Check stationarity
            if valid_params[1] + valid_params[2] >= 1.0:
                return torch.tensor(5e7, device=device)

            # Compute loss
            with torch.no_grad():
                loss = -1 * Calibration_Loss(valid_params, all_returns, sigma_all, model, X_all, N, M)

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(1e7, device=device)

            # Convert to minimization problem with reasonable scaling
            scaled_loss = torch.abs(loss) / 1e9

            # Add penalty for extreme parameter values
            penalty = 0
            if valid_params[0] > 5e-4:  # omega too large
                penalty += (valid_params[0] - 5e-4) * 1e6
            if valid_params[1] + valid_params[2] > 0.999:  # too close to unit root
                penalty += (valid_params[1] + valid_params[2] - 0.999) * 1e4

            return scaled_loss + penalty

        except Exception as e:
            return torch.tensor(1e9, device=device)

    # Run optimization
    print("Running DE optimization...")
    best_params, best_fitness, history = de.optimize(robust_objective, max_iter=150, verbose=True)

    # Final projection
    final_params = de._project_to_valid(best_params)

    return final_params.detach().cpu().numpy(), history


def main():
    """Main function"""

    print("🔧 Simple Working DE GARCH Calibration")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load model
        print("\nLoading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        print("✅ Model loaded")

        # Load dataset
        print("Loading dataset...")
        dataset = cal_dataset(
            "joint_dataset/scalable_hn_dataset_250x60.csv",
            "joint_dataset/assetprices.csv"
        )
        print(f"✅ Dataset: {len(dataset)} options, {len(dataset.returns)} returns")

        # True parameters
        true_params = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2])
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        print(f"\nTrue parameters:")
        for name, val in zip(param_names, true_params):
            print(f"  {name}: {val:.6f}")

        # Run calibration
        print(f"\nStarting DE calibration...")
        start_time = time.time()

        calibrated_params, history = calibrate_with_de(model, dataset, device)

        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")

        # Results
        l2_error = np.linalg.norm(calibrated_params - true_params)

        print(f"\nResults:")
        print(f"{'Param':<8} {'True':<10} {'Found':<10} {'Error':<10}")
        print("-" * 40)

        for i, name in enumerate(param_names):
            error = abs(calibrated_params[i] - true_params[i])
            print(f"{name:<8} {true_params[i]:<10.6f} {calibrated_params[i]:<10.6f} {error:<10.6f}")

        print(f"\nL2 Error: {l2_error:.6f}")

        # Validation
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta

        print(f"\nValidation:")
        print(f"  ω > 0: {'✅' if calibrated_params[0] > 0 else '❌'}")
        print(f"  α + β < 1: {'✅' if persistence < 1.0 else '❌'} ({persistence:.4f})")

        # Save results
        results = {
            'method': 'simple_differential_evolution',
            'calibrated_parameters': dict(zip(param_names, calibrated_params.tolist())),
            'true_parameters': dict(zip(param_names, true_params.tolist())),
            'l2_error': float(l2_error),
            'time_seconds': elapsed,
            'convergence_history': history,
            'validation': {
                'omega_positive': bool(calibrated_params[0] > 0),
                'stationary': bool(persistence < 1.0),
                'persistence': float(persistence)
            }
        }

        with open('working_de_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: working_de_results.json")

        # Quality assessment
        if l2_error < 0.5:
            quality = "Good"
        elif l2_error < 2.0:
            quality = "Fair"
        else:
            quality = "Poor"

        print(f"\nSummary:")
        print(f"  L2 Error: {l2_error:.6f}")
        print(f"  Quality: {quality}")
        print(f"  Time: {elapsed:.1f}s")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
