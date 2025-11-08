import torch
import numpy as np
from dataset2 import cal_dataset
from cal_loss import Calibration_Loss
import json

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedDifferentialEvolution:
    """Improved PyTorch-based Differential Evolution with adaptive features"""

    def __init__(self, bounds, popsize=50, mutation=0.8, crossover=0.7, seed=None, device=None):
        """
        Initialize Improved Differential Evolution optimizer

        Args:
            bounds: List of (min, max) tuples for each parameter
            popsize: Population size
            mutation: Initial mutation factor (F) - can be float or tuple (min, max) for range
            crossover: Initial crossover probability (CR) - will be adapted
            seed: Random seed for reproducibility
            device: PyTorch device
        """
        self.bounds = torch.tensor(bounds, device=device, dtype=torch.float32)
        self.popsize = popsize

        # Handle mutation as either single value or range
        if isinstance(mutation, (tuple, list)):
            self.initial_mutation_range = mutation
            self.initial_mutation = sum(mutation) / 2  # Use midpoint for adaptive decay
        else:
            self.initial_mutation_range = None
            self.initial_mutation = mutation

        self.initial_crossover = crossover
        self.device = device
        self.num_params = len(bounds)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize population with improved strategy
        self.population = self._initialize_population_smart()
        self.fitness = torch.full((popsize,), float('inf'), device=device)
        self.best_idx = 0
        self.best_fitness = float('inf')
        self.best_params = None

        # Adaptive parameters
        self.mutation_history = []
        self.success_history = []
        self.stagnation_counter = 0

    def _initialize_population_smart(self):
        """Smart population initialization with multiple strategies"""
        pop = torch.zeros(self.popsize, self.num_params, device=self.device)

        # Strategy 1: Random uniform (60% of population)
        n_random = int(0.6 * self.popsize)
        pop[:n_random] = torch.rand(n_random, self.num_params, device=self.device)
        pop[:n_random] = self.bounds[:, 0] + pop[:n_random] * (self.bounds[:, 1] - self.bounds[:, 0])

        # Strategy 2: Log-uniform for scale parameters (20% of population)
        n_log = int(0.2 * self.popsize)
        start_idx = n_random
        end_idx = start_idx + n_log

        for i in range(start_idx, min(end_idx, self.popsize)):
            for j in range(self.num_params):
                low, high = self.bounds[j, 0], self.bounds[j, 1]
                if low > 0 and high > low * 100:  # Use log-uniform for wide ranges
                    log_low, log_high = torch.log(low), torch.log(high)
                    log_val = log_low + torch.rand(1, device=self.device) * (log_high - log_low)
                    pop[i, j] = torch.exp(log_val)
                else:
                    pop[i, j] = low + torch.rand(1, device=self.device) * (high - low)

        # Strategy 3: Biased towards typical GARCH values (20% of population)
        n_bias = self.popsize - end_idx
        if n_bias > 0:
            # Typical GARCH parameter values
            typical_params = torch.tensor([
                [1e-5, 0.05, 0.9, 0.1, 0.0],
                [1e-6, 0.01, 0.95, 0.5, 0.1],
                [1e-4, 0.1, 0.8, -0.1, -0.1],
                [5e-6, 0.03, 0.92, 1.0, 0.05],
                [2e-5, 0.08, 0.85, -0.5, 0.2]
            ], device=self.device)

            for i in range(end_idx, self.popsize):
                # Select random typical parameter set
                base_idx = torch.randint(0, typical_params.shape[0], (1,)).item()
                base_params = typical_params[base_idx].clone()

                # Add noise
                noise = 0.1 * torch.randn(self.num_params, device=self.device)
                noisy_params = base_params * (1 + noise)

                # Clip to bounds
                pop[i] = torch.clamp(noisy_params, self.bounds[:, 0], self.bounds[:, 1])

        return pop

    def _enforce_bounds(self, candidate):
        """Enforce parameter bounds with reflection and clamping"""
        # First try reflection
        for i in range(self.num_params):
            low, high = self.bounds[i, 0], self.bounds[i, 1]
            if candidate[i] < low:
                candidate[i] = low + (low - candidate[i])
                if candidate[i] > high:
                    candidate[i] = low
            elif candidate[i] > high:
                candidate[i] = high - (candidate[i] - high)
                if candidate[i] < low:
                    candidate[i] = high

        # Final clamp to ensure bounds
        candidate = torch.clamp(candidate, self.bounds[:, 0], self.bounds[:, 1])
        return candidate

    def _adaptive_mutation_crossover(self, generation):
        """Adaptive mutation and crossover rates"""
        # Handle mutation range or single value
        if self.initial_mutation_range is not None:
            # For range, decay both bounds
            decay_factor = 0.95 ** (generation / 50)
            min_mut = max(0.3, self.initial_mutation_range[0] * decay_factor)
            max_mut = max(0.5, self.initial_mutation_range[1] * decay_factor)
            current_mutation = (min_mut, max_mut)
        else:
            # For single value, decay as before
            decay_factor = 0.95 ** (generation / 50)
            current_mutation = max(0.3, self.initial_mutation * decay_factor)

        # Increase crossover if stagnating
        if self.stagnation_counter > 20:
            current_crossover = min(0.95, self.initial_crossover + 0.1)
        else:
            current_crossover = self.initial_crossover

        return current_mutation, current_crossover

    def _mutate_and_crossover(self, target_idx, generation):
        """Improved DE mutation with multiple strategies"""
        current_mutation, current_crossover = self._adaptive_mutation_crossover(generation)

        # Select three random indices different from target
        candidates = list(range(self.popsize))
        candidates.remove(target_idx)
        indices = torch.randperm(len(candidates), device=self.device)[:3]
        r1, r2, r3 = [candidates[i] for i in indices]

        # Multiple mutation strategies
        strategy = torch.rand(1, device=self.device).item()

        # Sample mutation factor if range is provided
        if isinstance(current_mutation, tuple):
            mutation_factor = torch.rand(1, device=self.device) * (current_mutation[1] - current_mutation[0]) + current_mutation[0]
        else:
            mutation_factor = current_mutation

        if strategy < 0.6:  # DE/rand/1
            mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])
        elif strategy < 0.8:  # DE/best/1
            mutant = self.best_params + mutation_factor * (self.population[r1] - self.population[r2])
        else:  # DE/current-to-best/1
            mutant = self.population[target_idx] + mutation_factor * (self.best_params - self.population[target_idx]) + \
                    mutation_factor * (self.population[r1] - self.population[r2])

        mutant = self._enforce_bounds(mutant)

        # Binomial crossover with adaptive rate
        target = self.population[target_idx]
        trial = target.clone()

        crossover_mask = torch.rand(self.num_params, device=self.device) < current_crossover

        # Ensure at least one parameter is from mutant
        if not crossover_mask.any():
            j_rand = torch.randint(0, self.num_params, (1,), device=self.device)
            crossover_mask[j_rand] = True

        trial[crossover_mask] = mutant[crossover_mask]

        return trial

    def optimize(self, objective_func, max_iter=1000, tolerance=1e-6, verbose=True):
        """
        Run improved differential evolution optimization
        """
        convergence_history = []
        best_fitness_history = []

        # Evaluate initial population
        for i in range(self.popsize):
            self.fitness[i] = objective_func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i].item()
                self.best_idx = i
                self.best_params = self.population[i].clone()

        if verbose:
            print(f"Initial best fitness: {self.best_fitness:.6f}")

        prev_best = self.best_fitness

        for generation in range(max_iter):
            # Track improvements
            improvements = 0

            # Create new generation
            new_population = torch.zeros_like(self.population)
            new_fitness = torch.zeros_like(self.fitness)

            for i in range(self.popsize):
                # Generate trial vector
                trial = self._mutate_and_crossover(i, generation)
                trial_fitness = objective_func(trial)

                # Selection
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    improvements += 1

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

            # Track stagnation
            improvement = abs(prev_best - self.best_fitness) / max(abs(prev_best), 1e-10)
            if improvement < tolerance:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            prev_best = self.best_fitness
            convergence_history.append(self.best_fitness)
            best_fitness_history.append(self.best_fitness)

            # Print progress
            if verbose and (generation % 50 == 0 or generation == max_iter - 1):
                print(f"Generation {generation+1:4d}/{max_iter} | Best: {self.best_fitness:.6f} | "
                      f"Improvements: {improvements}/{self.popsize} | Stagnation: {self.stagnation_counter}")

            # Early stopping with improved criteria
            if self.stagnation_counter > 50:
                if verbose:
                    print(f"Converged after {generation+1} generations (stagnation)")
                break

            if len(convergence_history) > 20:
                recent_improvement = abs(convergence_history[-20] - convergence_history[-1])
                if recent_improvement < tolerance * 10:
                    if verbose:
                        print(f"Converged after {generation+1} generations (minimal improvement)")
                    break

        return self.best_params, self.best_fitness, convergence_history


def project_parameters(params):
    """Project parameters to valid GARCH domain with improved constraints"""
    omega = torch.clamp(params[0], min=1e-8, max=1e-2)
    alpha = torch.clamp(params[1], min=1e-8, max=0.99)
    beta = torch.clamp(params[2], min=0.01, max=0.99)
    gamma = torch.clamp(params[3], min=-5.0, max=5.0)
    lambda_param = torch.clamp(params[4], min=-0.5, max=0.5)

    # Ensure stationarity: alpha + beta < 1
    persistence = alpha + beta
    if persistence >= 1.0:
        scale_factor = 0.99 / persistence
        alpha = alpha * scale_factor
        beta = beta * scale_factor

    return torch.stack([omega, alpha, beta, gamma, lambda_param])


def calibrate_de_improved(model, dataset, popsize=50, max_iter=500, mutation=0.8,
                         crossover=0.7, seed=42, device=None):
    """
    Improved GARCH calibration using enhanced Differential Evolution

    Args:
        mutation: Mutation factor - can be float or tuple (min, max) for range
    """
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

    print(f"Starting improved DE calibration: {M} options, {N} returns")

    # Improved parameter bounds
    bounds = [
        (1e-8, 1e-2),     # omega: wider range
        (1e-8, 0.3),      # alpha: avoid zero, reasonable upper bound
        (0.1, 0.99),      # beta: avoid zero for stability
        (-3.0, 3.0),      # gamma: moderate range
        (-0.3, 0.3)       # lambda: moderate risk premium
    ]

    de = ImprovedDifferentialEvolution(
        bounds=bounds,
        popsize=popsize,
        mutation=mutation,
        crossover=crossover,
        seed=seed,
        device=device
    )

    def robust_objective_function(params):
        """Robust objective function with error handling and scaling"""
        try:
            # Project to valid domain
            params_proj = project_parameters(params)

            # Sanity checks
            if torch.any(torch.isnan(params_proj)) or torch.any(torch.isinf(params_proj)):
                return torch.tensor(1e8, device=device, dtype=torch.float32)

            # Check stationarity
            alpha, beta = params_proj[1], params_proj[2]
            if alpha + beta >= 1.0:
                return torch.tensor(1e7, device=device, dtype=torch.float32)

            # Compute calibration loss
            with torch.no_grad():
                loss = -1 * Calibration_Loss(params_proj, all_returns, sigma_all, model, X_all, N, M)

            # Handle numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(1e6, device=device, dtype=torch.float32)

            # Scale loss to reasonable range
            if loss > 1e10:
                return torch.tensor(1e5, device=device, dtype=torch.float32)

            return loss.float()

        except Exception as e:
            print(f"Error in objective function: {e}")
            return torch.tensor(1e9, device=device, dtype=torch.float32)

    # Run optimization
    best_params, best_fitness, history = de.optimize(
        robust_objective_function,
        max_iter=max_iter,
        tolerance=1e-8,
        verbose=True
    )

    # Final projection
    best_params = project_parameters(best_params)

    print(f"\n✅ Improved DE calibration complete")
    print(f"Final loss: {best_fitness:.6f}")

    return best_params.detach().cpu().numpy(), history


def main():
    try:
        print("🧬 Improved Differential Evolution GARCH Calibration")
        print("=" * 65)

        print("Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print("Model loaded successfully.")

        print("Loading dataset...")
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                              "joint_dataset/assetprices.csv")
        print(f"Dataset: {len(dataset)} options, returns length {len(dataset.returns)}")

        # Run improved DE calibration
        calibrated_params, convergence_history = calibrate_de_improved(
            model, dataset,
            popsize=50,
            max_iter=300,
            mutation=0.7,
            crossover=0.8,
            seed=42
        )

        # Compare with true values
        true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        pred_vals = calibrated_params

        l2_error = np.linalg.norm(pred_vals - true_vals, ord=2)
        print(f"L2 error: {l2_error:.6f}")

        # Detailed comparison
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
        print(f"\nDetailed Results:")
        print(f"{'Parameter':<10} {'True':<12} {'Calibrated':<12} {'Error':<12} {'Rel Error %':<12}")
        print("-" * 60)

        for i, name in enumerate(param_names):
            true_val = true_vals[i]
            cal_val = calibrated_params[i]
            abs_error = abs(cal_val - true_val)
            rel_error = abs_error / abs(true_val) * 100 if true_val != 0 else 0

            print(f"{name:<10} {true_val:<12.6f} {cal_val:<12.6f} {abs_error:<12.6f} {rel_error:<12.2f}")

        # Validation
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta
        print(f"\nValidation:")
        print(f"   α + β = {persistence:.6f} {'(✅ Stationary)' if persistence < 1.0 else '(❌ Non-stationary)'}")
        print(f"   ω > 0: {'✅' if calibrated_params[0] > 0 else '❌'} ({calibrated_params[0]:.8f})")

        # Save results
        results = {
            'method': 'improved_differential_evolution',
            'calibrated_parameters': dict(zip(param_names, calibrated_params.tolist())),
            'true_parameters': dict(zip(param_names, true_vals.tolist())),
            'errors': {
                'l2_error': float(l2_error),
                'absolute_errors': dict(zip(param_names, (np.abs(calibrated_params - true_vals)).tolist()))
            },
            'validation': {
                'persistence': float(persistence),
                'stationary': bool(persistence < 1.0),
                'omega_positive': bool(calibrated_params[0] > 0)
            },
            'convergence_history': convergence_history
        }

        with open('improved_de_calibration_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to improved_de_calibration_results.json")

        # Plot convergence if possible
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(convergence_history, 'b-', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Objective Function Value')
            plt.title('Improved DE Convergence')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.savefig('improved_de_convergence.png', dpi=300, bbox_inches='tight')
            print(f"📈 Convergence plot saved to improved_de_convergence.png")
        except ImportError:
            print("📈 Matplotlib not available for plotting")

    except Exception as e:
        print(f"❌ Improved DE calibration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
