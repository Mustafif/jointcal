import json

import numpy as np
import torch

from cal_loss import Calibration_Loss
from dataset2 import cal_dataset

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DifferentialEvolution:
    """PyTorch-based Differential Evolution optimizer for GARCH parameter calibration"""

    def __init__(self, bounds, popsize=50, mutation=0.8, crossover=0.7, seed=None, device=None):
        """
        Initialize Differential Evolution optimizer

        Args:
            bounds: List of (min, max) tuples for each parameter
            popsize: Population size
            mutation: Mutation factor (F) - can be float or tuple (min, max) for range
            crossover: Crossover probability (CR)
            seed: Random seed for reproducibility
            device: PyTorch device
        """
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

        # Initialize population
        self.population = self._initialize_population()
        self.fitness = torch.full((popsize,), float('inf'), device=device)
        self.best_idx = 0
        self.best_fitness = float('inf')
        self.best_params = None

    def _initialize_population(self):
        """Initialize population within bounds"""
        pop = torch.rand(self.popsize, self.num_params, device=self.device)
        # Scale to bounds
        pop = self.bounds[:, 0] + pop * (self.bounds[:, 1] - self.bounds[:, 0])
        return pop

    def _enforce_bounds(self, candidate):
        """Enforce parameter bounds with reflection"""
        candidate = torch.clamp(candidate, self.bounds[:, 0], self.bounds[:, 1])
        return candidate

    def _mutate_and_crossover(self, target_idx):
        """DE/rand/1/bin mutation and crossover strategy"""
        # Select three random indices different from target
        candidates = list(range(self.popsize))
        candidates.remove(target_idx)
        r1, r2, r3 = torch.randperm(len(candidates), device=self.device)[:3]
        r1, r2, r3 = candidates[r1], candidates[r2], candidates[r3]

        # Mutation: v = x_r1 + F * (x_r2 - x_r3)
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

        # Generate crossover mask
        crossover_mask = torch.rand(self.num_params, device=self.device) < self.crossover
        # Ensure at least one parameter is from mutant
        if not crossover_mask.any():
            j_rand = torch.randint(0, self.num_params, (1,), device=self.device)
            crossover_mask[j_rand] = True

        trial[crossover_mask] = mutant[crossover_mask]

        return trial

    def optimize(self, objective_func, max_iter=1000, tolerance=1e-6, verbose=True):
        """
        Run differential evolution optimization

        Args:
            objective_func: Function to minimize (should accept tensor and return scalar)
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            best_params: Best parameter vector
            best_fitness: Best fitness value
            convergence_history: List of best fitness per iteration
        """
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

        for iteration in range(max_iter):
            # Create new generation
            new_population = torch.zeros_like(self.population)
            new_fitness = torch.zeros_like(self.fitness)

            for i in range(self.popsize):
                # Generate trial vector
                trial = self._mutate_and_crossover(i)
                trial_fitness = objective_func(trial)

                # Selection
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness.item()
                        self.best_idx = i
                        self.best_params = trial.clone()
                else:
                    new_population[i] = self.population[i]
                    new_fitness[i] = self.fitness[i]

            # Update population
            self.population = new_population
            self.fitness = new_fitness
            convergence_history.append(self.best_fitness)

            # Print progress
            if verbose and (iteration % 50 == 0 or iteration == max_iter - 1):
                print(f"Iteration {iteration+1:4d}/{max_iter} | Best fitness: {self.best_fitness:.6f} | "
                      f"Params: {self.best_params.cpu().numpy()}")

            # Check convergence
            if len(convergence_history) > 10:
                recent_improvement = abs(convergence_history[-10] - convergence_history[-1])
                if recent_improvement < tolerance:
                    if verbose:
                        print(f"Converged after {iteration+1} iterations")
                    break

        return self.best_params, self.best_fitness, convergence_history


def project_parameters(params):
    """Project parameters to valid domain"""
    omega = torch.clamp(params[0], min=1e-8)
    alpha = torch.clamp(params[1], min=0.0, max=1.0)
    beta = torch.clamp(params[2], min=0.0, max=1.0)
    gamma = params[3]        # can be negative
    lambda_param = params[4] # no constraint
    return torch.stack([omega, alpha, beta, gamma, lambda_param])


def calibrate_de(model, dataset, popsize=50, max_iter=1000, mutation=(0.5, 1.0),
                 crossover=0.7, seed=42, device=None):
    """
    Calibrate GARCH parameters using Differential Evolution

    Args:
        model: Trained neural network model
        dataset: Calibration dataset
        popsize: DE population size
        max_iter: Maximum iterations
        mutation: DE mutation factor - can be float or tuple (min, max) for range (default: (0.5, 1.0))
        crossover: DE crossover probability
        seed: Random seed
        device: PyTorch device
        regularization: Dict with regularization options:
            - 'type': 'l2', 'l1', 'weighted', 'bounds', 'combined', 'adaptive', 'multi'
            - 'weight': regularization weight (default: 1.0)
            - 'true_params': true parameter values for regularization
            - 'param_weights': individual parameter weights [5 elements]
            - 'multi_weights': [data_weight, param_weight, constraint_weight] for multi-objective

    Returns:
        Calibrated parameters as numpy array
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False  # freeze network

    # Precompute tensors
    X_all = dataset.X.to(device)        # M x num_features
    sigma_all = dataset.sigma.to(device) # M
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"Starting DE calibration: {M} options, {N} returns")
    print(f"Population size: {popsize}, Max iterations: {max_iter}")

    # Define parameter bounds
    # [omega, alpha, beta, gamma, lambda]
    bounds = [
        (1e-7, 1e-6),    # omega: positive, small
        (1.15e-6, 1.36e-6),      # alpha: [0,1]
        (0.75, 0.85),      # beta: [0,1]
        (1, 5),   # gamma: can be negative
        (0.2, 0.5)      # lambda: risk premium
    ]

    # Initialize DE optimizer
    de = DifferentialEvolution(bounds, popsize=popsize, mutation=mutation,
                              crossover=crossover, seed=seed, device=device)

    def objective_function(params):
        """Objective function for DE optimization"""
        # Project parameters to valid domain
        params_proj = project_parameters(params)

        # Compute calibration loss with optional regularization
        with torch.no_grad():
            loss = Calibration_Loss(params_proj, all_returns, sigma_all,
                                           model, X_all, N, M)
        return loss

    # Run optimization
    best_params, best_fitness, history = de.optimize(
        objective_function, max_iter=max_iter, verbose=True
    )

    # Final projection
    best_params = project_parameters(best_params)

    print(f"\n✅ DE Calibration complete")
    print(f"Final loss: {best_fitness:.6f}")

    return best_params.detach().cpu().numpy(), history


def main():
    try:
        print("Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print("Model loaded successfully.")

        print("Loading dataset...")
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                              "joint_dataset/assetprices.csv")
        print(f"Dataset: {len(dataset)} options, returns length {len(dataset.returns)}")

        calibrated_params, convergence_history = calibrate_de(
            model, dataset,
            popsize=50,           # Population size
            max_iter=500,         # Maximum iterations
            mutation=0.8,  # Mutation factor range
            crossover=0.7,        # Crossover probability
            seed=42,              # For reproducibility
        )

        # Compare with true values
        true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        pred_vals = calibrated_params

        two_norm = np.linalg.norm(pred_vals - true_vals, ord=2)
        print(f"Two-norm error: {two_norm:.6f}")

        # Save results
        results = {
            'calibrated_params': dict(zip(['omega', 'alpha', 'beta', 'gamma', 'lambda'],
                                        calibrated_params.tolist())),
            'convergence_history': convergence_history,
            'final_loss': convergence_history[-1] if convergence_history else None,
            'two_norm_error': two_norm
        }

        with open('calibrated_params_de.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n📁 Results saved to calibrated_params_de.json")

        # Validation
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta
        print(f"\nValidation: α+β = {persistence:.6f}")
        if persistence < 1.0:
            print("✅ Stationarity satisfied")
        else:
            print("⚠️  Stationarity violated")

        if calibrated_params[0] > 0:
            print("✅ Omega positive")
        else:
            print("⚠️  Omega not positive")

        if persistence < 1.0:
            unconditional_var = calibrated_params[0] / (1 - persistence)
            print(f"Theoretical unconditional variance: {unconditional_var:.8f}")
            empirical_var = dataset.returns.var().item()
            print(f"Empirical returns variance: {empirical_var:.8f}")

        # Plot convergence if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(convergence_history)
            plt.title('Differential Evolution Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Function Value')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig('de_convergence.png', dpi=300, bbox_inches='tight')
            print("📊 Convergence plot saved to de_convergence.png")
        except ImportError:
            print("📊 Matplotlib not available, skipping convergence plot")

    except Exception as e:
        print(f"❌ DE Calibration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
