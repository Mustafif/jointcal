import json
import time

import numpy as np
import torch

from cal_loss import Calibration_Loss
from dataset2 import cal_dataset
from hn import HestonNandiGARCH

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"


class GPUDifferentialEvolution:
    """Fully GPU-accelerated Differential Evolution using PyTorch CUDA"""

    def __init__(self, bounds, popsize=50, mutation=(0.5, 1.0), crossover=0.7,
                 seed=None, device=None, strategy='best1bin'):
        """
        Initialize GPU-accelerated Differential Evolution

        Args:
            bounds: List of (min, max) tuples for each parameter
            popsize: Population size
            mutation: Mutation factor - can be float or tuple (min, max) for adaptive
            crossover: Crossover probability
            seed: Random seed for reproducibility
            device: PyTorch device (auto-detects best GPU if None)
            strategy: DE strategy ('best1bin', 'rand1bin', 'currenttobest1bin')
        """
        # Auto-detect best device (CUDA or ROCm)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
                gpu_name = torch.cuda.get_device_name() if hasattr(torch.cuda, 'get_device_name') else "GPU"
                gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9 if hasattr(torch.cuda, 'get_device_properties') else 0
                print(f"🚀 Using GPU: {gpu_name}")
                if gpu_memory > 0:
                    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                # Check if it's ROCm
                if 'rocm' in torch.__version__.lower():
                    print("🔥 ROCm (AMD GPU) detected")
                else:
                    print("💚 CUDA (NVIDIA GPU) detected")
            else:
                device = torch.device("cpu")
                print("⚠️ GPU not available, falling back to CPU")

        self.device = device
        self.bounds = torch.tensor(bounds, device=device, dtype=torch.float32)
        self.popsize = popsize
        self.num_params = len(bounds)
        self.strategy = strategy

        # Handle adaptive mutation
        if isinstance(mutation, (tuple, list)):
            self.mutation_min, self.mutation_max = mutation
            self.adaptive_mutation = True
        else:
            self.mutation_fixed = mutation
            self.adaptive_mutation = False

        self.crossover = crossover

        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        # Initialize population on GPU
        self.population = self._initialize_population()
        self.fitness = torch.full((popsize,), float('inf'), device=device)
        self.best_idx = 0
        self.best_fitness = float('inf')
        self.best_params = None

        # Performance tracking
        self.gpu_memory_usage = []

    def _initialize_population(self):
        """Initialize population on GPU using Latin Hypercube Sampling"""
        # Use Latin Hypercube for better coverage
        pop = torch.rand(self.popsize, self.num_params, device=self.device)

        # Apply Latin Hypercube sampling for better space coverage
        for i in range(self.num_params):
            perm = torch.randperm(self.popsize, device=self.device).float()
            pop[:, i] = (perm + pop[:, i]) / self.popsize

        # Scale to bounds
        pop = self.bounds[:, 0] + pop * (self.bounds[:, 1] - self.bounds[:, 0])
        return pop

    def _enforce_bounds_vectorized(self, candidates):
        """Vectorized bounds enforcement on GPU"""
        return torch.clamp(candidates, self.bounds[:, 0], self.bounds[:, 1])

    def _generate_mutations_vectorized(self):
        """Generate all mutations for the population in parallel on GPU"""
        batch_size = self.popsize

        if self.strategy == 'best1bin':
            # DE/best/1/bin: v = x_best + F * (x_r1 - x_r2)
            # Generate random indices for r1, r2
            r1 = torch.randint(0, self.popsize, (batch_size,), device=self.device)
            r2 = torch.randint(0, self.popsize, (batch_size,), device=self.device)

            # Ensure r1 != r2
            mask = r1 == r2
            while mask.any():
                r2[mask] = torch.randint(0, self.popsize, (mask.sum(),), device=self.device)
                mask = r1 == r2

            best_expanded = self.population[self.best_idx].unsqueeze(0).expand(batch_size, -1)

        elif self.strategy == 'rand1bin':
            # DE/rand/1/bin: v = x_r0 + F * (x_r1 - x_r2)
            r0 = torch.randint(0, self.popsize, (batch_size,), device=self.device)
            r1 = torch.randint(0, self.popsize, (batch_size,), device=self.device)
            r2 = torch.randint(0, self.popsize, (batch_size,), device=self.device)

            # Ensure all indices are different
            for _ in range(10):  # Max 10 attempts to avoid infinite loop
                mask1 = r0 == r1
                mask2 = r0 == r2
                mask3 = r1 == r2
                any_equal = mask1 | mask2 | mask3
                if not any_equal.any():
                    break
                r1[any_equal] = torch.randint(0, self.popsize, (any_equal.sum(),), device=self.device)
                r2[any_equal] = torch.randint(0, self.popsize, (any_equal.sum(),), device=self.device)

            best_expanded = self.population[r0]

        elif self.strategy == 'currenttobest1bin':
            # DE/current-to-best/1/bin: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            r1 = torch.randint(0, self.popsize, (batch_size,), device=self.device)
            r2 = torch.randint(0, self.popsize, (batch_size,), device=self.device)

            mask = r1 == r2
            while mask.any():
                r2[mask] = torch.randint(0, self.popsize, (mask.sum(),), device=self.device)
                mask = r1 == r2

            best_expanded = self.population  # Will be handled differently below

        # Generate mutation factors
        if self.adaptive_mutation:
            F = torch.rand(batch_size, 1, device=self.device) * (self.mutation_max - self.mutation_min) + self.mutation_min
        else:
            F = torch.full((batch_size, 1), self.mutation_fixed, device=self.device)

        # Compute mutations based on strategy
        if self.strategy == 'best1bin':
            mutants = best_expanded + F * (self.population[r1] - self.population[r2])
        elif self.strategy == 'rand1bin':
            mutants = best_expanded + F * (self.population[r1] - self.population[r2])
        elif self.strategy == 'currenttobest1bin':
            best_vec = self.population[self.best_idx].unsqueeze(0).expand(batch_size, -1)
            mutants = self.population + F * (best_vec - self.population) + F * (self.population[r1] - self.population[r2])

        return self._enforce_bounds_vectorized(mutants)

    def _crossover_vectorized(self, mutants):
        """Vectorized crossover operation on GPU"""
        trials = self.population.clone()

        # Generate crossover mask
        crossover_mask = torch.rand(self.popsize, self.num_params, device=self.device) < self.crossover

        # Ensure at least one parameter is taken from mutant (j_rand)
        j_rand = torch.randint(0, self.num_params, (self.popsize,), device=self.device)
        crossover_mask[torch.arange(self.popsize, device=self.device), j_rand] = True

        # Apply crossover
        trials[crossover_mask] = mutants[crossover_mask]

        return trials

    def optimize(self, objective_func, maxiter=1000, tolerance=1e-8, verbose=True,
                 memory_efficient=True):
        """
        Run GPU-accelerated differential evolution

        Args:
            objective_func: Objective function (should handle batch evaluation on GPU)
            maxiter: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            memory_efficient: Use memory-efficient batch evaluation

        Returns:
            best_params: Best parameter vector
            best_fitness: Best fitness value
            convergence_history: History of best fitness values
        """
        convergence_history = []

        # Initial evaluation
        if verbose:
            print(f"🚀 Starting GPU-DE optimization on {self.device}")
            print(f"Population size: {self.popsize}, Max iterations: {maxiter}")

        # Evaluate initial population
        self.fitness = objective_func(self.population)
        best_idx = torch.argmin(self.fitness)
        self.best_idx = best_idx
        self.best_fitness = self.fitness[best_idx].item()
        self.best_params = self.population[best_idx].clone()

        if verbose:
            print(f"Initial best fitness: {self.best_fitness:.8f}")

        # Track GPU memory
        if torch.cuda.is_available():
            self.gpu_memory_usage.append(torch.cuda.memory_allocated(self.device) / 1e9)

        for iteration in range(maxiter):
            # Generate mutations for entire population
            mutants = self._generate_mutations_vectorized()

            # Crossover
            trials = self._crossover_vectorized(mutants)

            # Evaluate trials
            trial_fitness = objective_func(trials)

            # Selection (vectorized)
            improvement_mask = trial_fitness < self.fitness
            self.population[improvement_mask] = trials[improvement_mask]
            self.fitness[improvement_mask] = trial_fitness[improvement_mask]

            # Update global best
            current_best_idx = torch.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx].item()

            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_idx = current_best_idx
                self.best_params = self.population[current_best_idx].clone()

            convergence_history.append(self.best_fitness)

            # Track GPU memory usage (works for both CUDA and ROCm)
            if torch.cuda.is_available() and iteration % 100 == 0:
                current_memory = torch.cuda.memory_allocated(self.device) / 1e9
                self.gpu_memory_usage.append(current_memory)

            # Progress reporting
            if verbose and (iteration % 50 == 0 or iteration == maxiter - 1):
                memory_info = ""
                if torch.cuda.is_available():
                    memory_info = f" | GPU Mem: {torch.cuda.memory_allocated(self.device) / 1e9:.1f}GB"

                print(f"Iter {iteration+1:4d}/{maxiter} | Fitness: {self.best_fitness:.8f}{memory_info}")
                print(f"    Best params: {self.best_params.cpu().numpy()}")

            # Convergence check
            if len(convergence_history) > 20:
                recent_improvement = abs(convergence_history[-20] - convergence_history[-1])
                if recent_improvement < tolerance:
                    if verbose:
                        print(f"✅ Converged after {iteration+1} iterations")
                    break

        return self.best_params, self.best_fitness, convergence_history


def project_parameters_gpu(params_batch):
    """GPU-accelerated parameter projection for GARCH constraints"""
    if len(params_batch.shape) == 1:
        params_batch = params_batch.unsqueeze(0)

    batch_size = params_batch.shape[0]
    device = params_batch.device

    # Project each parameter
    omega = torch.clamp(params_batch[:, 0], min=1e-8)
    alpha = torch.clamp(params_batch[:, 1], min=0.0, max=1.0)
    beta = torch.clamp(params_batch[:, 2], min=0.0, max=1.0)
    gamma = params_batch[:, 3]  # No constraints
    lambda_param = params_batch[:, 4]  # No constraints

    # Stack results
    projected = torch.stack([omega, alpha, beta, gamma, lambda_param], dim=1)

    return projected.squeeze() if batch_size == 1 else projected


def calibrate_gpu_de(model, dataset, popsize=100, maxiter=500, strategy='best1bin',
                     mutation=(0.5, 1.0), crossover=0.7, seed=42, batch_size=None):
    """
    GPU-accelerated GARCH calibration using Differential Evolution

    Args:
        model: PyTorch neural network model
        dataset: Calibration dataset
        popsize: Population size
        maxiter: Maximum iterations
        strategy: DE strategy ('best1bin', 'rand1bin', 'currenttobest1bin')
        mutation: Mutation factor(s)
        crossover: Crossover probability
        seed: Random seed
        batch_size: Batch size for objective evaluation (None for full batch)

    Returns:
        best_params: Calibrated parameters
        convergence_history: Optimization history
    """
    # Auto-detect GPU (CUDA or ROCm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cpu':
        print("⚠️ GPU not available, this will be slow!")
    else:
        # Show GPU type
        if 'rocm' in torch.__version__.lower():
            print("🔥 Using ROCm (AMD GPU) acceleration")
        else:
            print("💚 Using CUDA (NVIDIA GPU) acceleration")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Precompute and move data to GPU
    X_all = dataset.X.to(device)
    sigma_all = dataset.sigma.to(device)
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"🚀 GPU-accelerated DE calibration: {M} options, {N} returns")

    # Get HN GARCH initial guess
    hn_model = HestonNandiGARCH(all_returns.cpu().numpy())
    hn_result = hn_model.fit()

    # GARCH parameter bounds
    bounds = [
        (1e-7, 1e-6),    # omega
        (1e-6, 1e-5),    # alpha
        (0.7, 0.99),     # beta
        (0.1, 10.0),     # gamma
        (0.2, 0.5)       # lambda
    ]

    # Initialize GPU DE
    de = GPUDifferentialEvolution(
        bounds=bounds,
        popsize=popsize,
        mutation=mutation,
        crossover=crossover,
        seed=seed,
        device=device,
        strategy=strategy
    )

    # Set initial guess in population
    initial_guess_tensor = torch.tensor(hn_model.fitted_params, device=device, dtype=torch.float32)
    de.population[0] = initial_guess_tensor

    def batch_objective_function(params_batch):
        """
        Vectorized objective function for GPU evaluation

        Args:
            params_batch: [batch_size, num_params] or [num_params] tensor

        Returns:
            losses: [batch_size] tensor of loss values
        """
        # Handle single parameter vector
        if len(params_batch.shape) == 1:
            params_batch = params_batch.unsqueeze(0)
            single_param = True
        else:
            single_param = False

        batch_size = params_batch.shape[0]

        # Project parameters to valid domain
        params_proj = project_parameters_gpu(params_batch)

        # Compute losses for entire batch
        losses = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            with torch.no_grad():
                loss = Calibration_Loss(params_proj[i], all_returns, sigma_all,
                                      model, X_all, N, M)
                losses[i] = loss

        return losses.squeeze() if single_param else losses

    # Run optimization
    print("🏃 Starting GPU optimization...")
    start_time = time.time()

    best_params, best_fitness, history = de.optimize(
        objective_func=batch_objective_function,
        maxiter=maxiter,
        tolerance=1e-8,
        verbose=True
    )

    optimization_time = time.time() - start_time

    # Final projection
    best_params = project_parameters_gpu(best_params)

    print(f"\n✅ GPU DE Calibration Complete!")
    print(f"⏱️ Total time: {optimization_time:.2f} seconds")
    print(f"🎯 Final loss: {best_fitness:.8f}")

    if torch.cuda.is_available():
        print(f"💾 Peak GPU memory: {max(de.gpu_memory_usage):.1f} GB")
        # GPU utilization not available on all platforms
        try:
            utilization = torch.cuda.utilization()
            print(f"🔥 GPU utilization: {utilization}%")
        except:
            print("🔥 GPU utilization: Not available on this platform")

    return best_params.cpu().numpy(), history


def main():
    """Main function for GPU-accelerated calibration"""
    try:
        print("🚀 GPU-Accelerated Differential Evolution GARCH Calibration")
        print("=" * 70)

        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name() if hasattr(torch.cuda, 'get_device_name') else "GPU"
            print(f"✅ GPU Available: {gpu_name}")

            if hasattr(torch.cuda, 'get_device_properties'):
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"💾 GPU Memory: {gpu_memory:.1f} GB")

            # Show GPU type
            if 'rocm' in torch.__version__.lower():
                print("🔥 ROCm (AMD GPU) support detected")
            else:
                print("💚 CUDA (NVIDIA GPU) support detected")
        else:
            print("❌ GPU not available - will use CPU (much slower)")

        print("Loading model...")
        model = torch.load(MODEL_PATH, weights_only=False)
        print("✅ Model loaded")

        print("Loading dataset...")
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                              "joint_dataset/assetprices.csv")
        print(f"✅ Dataset loaded: {len(dataset)} options")

        # Test different strategies
        strategies = ['best1bin', 'rand1bin', 'currenttobest1bin']

        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"🧬 Testing strategy: {strategy}")
            print(f"{'='*50}")

            calibrated_params, convergence_history = calibrate_gpu_de(
                model=model,
                dataset=dataset,
                popsize=80,                 # Good balance of exploration/speed
                maxiter=400,               # GPU can handle more evaluations quickly
                strategy=strategy,
                mutation=(0.3, 0.9),      # Adaptive mutation
                crossover=0.8,            # Higher crossover for GPU efficiency
                seed=42
            )

            # Analyze results
            true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
            l2_error = np.linalg.norm(calibrated_params - true_vals)

            param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

            print(f"\n📊 Results for {strategy}:")
            print(f"L2 Error: {l2_error:.8f}")

            # Parameter table
            print(f"\n{'Parameter':<10} {'Calibrated':<15} {'True':<15} {'Error':<15}")
            print("-" * 60)
            for i, name in enumerate(param_names):
                error = abs(calibrated_params[i] - true_vals[i])
                print(f"{name:<10} {calibrated_params[i]:<15.8f} {true_vals[i]:<15.8f} {error:<15.8f}")

            # Validation
            alpha, beta = calibrated_params[1], calibrated_params[2]
            persistence = alpha + beta

            print(f"\n✅ Validation:")
            print(f"Persistence (α+β): {persistence:.6f}")
            print(f"Stationary: {'✅' if persistence < 1.0 else '❌'}")
            print(f"Omega > 0: {'✅' if calibrated_params[0] > 0 else '❌'}")

            # Save results
            results = {
                'method': 'gpu_differential_evolution',
                'strategy': strategy,
                'calibrated_parameters': dict(zip(param_names, calibrated_params.tolist())),
                'true_parameters': dict(zip(param_names, true_vals.tolist())),
                'l2_error': float(l2_error),
                'convergence_history': convergence_history,
                'validation': {
                    'persistence': float(persistence),
                    'stationary': bool(persistence < 1.0),
                    'omega_positive': bool(calibrated_params[0] > 0)
                }
            }

            filename = f'gpu_de_results_{strategy}.json'
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📁 Results saved to {filename}")

        print(f"\n🎉 GPU DE calibration completed for all strategies!")

    except Exception as e:
        print(f"❌ GPU DE calibration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
