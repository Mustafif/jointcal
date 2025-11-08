#!/usr/bin/env python3
"""
Demonstration of Mutation Range vs Single Value in Differential Evolution

This script shows how using a mutation range (a, b) instead of a single mutation
factor can improve exploration and convergence in differential evolution.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from calibrate2_de import DifferentialEvolution

def rosenbrock(x):
    """Rosenbrock function - a classic optimization benchmark"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def objective_function(population):
    """Evaluate Rosenbrock function for a population"""
    fitness = torch.zeros(population.shape[0], device=population.device)
    for i in range(population.shape[0]):
        fitness[i] = rosenbrock(population[i])
    return fitness

def run_de_experiment(mutation_param, label, max_iter=200, popsize=30):
    """Run DE with given mutation parameter"""

    # Problem setup: minimize Rosenbrock function
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]  # 2D Rosenbrock
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DE
    de = DifferentialEvolution(
        bounds=bounds,
        popsize=popsize,
        mutation=mutation_param,
        crossover=0.7,
        seed=42,
        device=device
    )

    # Initialize population
    de.population = torch.zeros(popsize, 2, device=device)
    for i in range(popsize):
        for j in range(2):
            low, high = bounds[j]
            de.population[i, j] = low + torch.rand(1, device=device) * (high - low)

    # Evolution loop
    history = []
    de.fitness = objective_function(de.population)
    de.best_idx = torch.argmin(de.fitness)
    de.best_params = de.population[de.best_idx].clone()

    for generation in range(max_iter):
        # Store best fitness
        best_fitness = de.fitness[de.best_idx].item()
        history.append(best_fitness)

        # Early stopping if converged
        if best_fitness < 1e-6:
            print(f"{label}: Converged at generation {generation}")
            break

        # Evolve population
        new_population = de.population.clone()
        for i in range(popsize):
            trial = de._mutate_and_crossover(i)
            trial_fitness = objective_function(trial.unsqueeze(0))[0]

            # Selection
            if trial_fitness < de.fitness[i]:
                new_population[i] = trial
                de.fitness[i] = trial_fitness

                # Update best
                if trial_fitness < de.fitness[de.best_idx]:
                    de.best_idx = i
                    de.best_params = trial.clone()

        de.population = new_population

    return history, de.best_params, de.fitness[de.best_idx].item()

def main():
    """Compare single mutation vs mutation range"""

    print("🔬 Mutation Range vs Single Value Demonstration")
    print("=" * 55)
    print("Optimizing 2D Rosenbrock function: f(x,y) = 100(y-x²)² + (1-x)²")
    print("Global minimum: f(1,1) = 0")
    print()

    # Run experiments
    experiments = [
        (0.8, "Single Mutation (F=0.8)"),
        ((0.5, 1.1), "Mutation Range (F∈[0.5,1.1])"),
        (0.5, "Single Mutation (F=0.5)"),
        ((0.3, 0.7), "Narrow Range (F∈[0.3,0.7])"),
        ((0.7, 1.3), "Wide Range (F∈[0.7,1.3])")
    ]

    results = {}
    plt.figure(figsize=(12, 8))

    for mutation_param, label in experiments:
        print(f"Running: {label}")
        history, best_params, best_fitness = run_de_experiment(mutation_param, label)
        results[label] = {
            'history': history,
            'best_params': best_params.cpu().numpy(),
            'best_fitness': best_fitness,
            'generations': len(history)
        }

        # Plot convergence
        plt.plot(history, label=label, linewidth=2, alpha=0.8)

        print(f"  Final result: f({best_params[0]:.4f}, {best_params[1]:.4f}) = {best_fitness:.6f}")
        print(f"  Generations: {len(history)}")
        print()

    # Format plot
    plt.yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Convergence Comparison: Single Mutation vs Mutation Range')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('mutation_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Convergence plot saved as 'mutation_comparison.png'")

    # Analysis
    print("\n📈 Analysis:")
    print("-" * 30)

    # Find best performer
    best_result = min(results.items(), key=lambda x: x[1]['best_fitness'])
    fastest_result = min(results.items(), key=lambda x: x[1]['generations'])

    print(f"🏆 Best final result: {best_result[0]}")
    print(f"   Final fitness: {best_result[1]['best_fitness']:.8f}")

    print(f"⚡ Fastest convergence: {fastest_result[0]}")
    print(f"   Generations: {fastest_result[1]['generations']}")

    # Range vs single comparison
    range_results = [r for name, r in results.items() if 'Range' in name]
    single_results = [r for name, r in results.items() if 'Single' in name]

    if range_results and single_results:
        avg_range_fitness = np.mean([r['best_fitness'] for r in range_results])
        avg_single_fitness = np.mean([r['best_fitness'] for r in single_results])

        print(f"\n🎯 Average Performance:")
        print(f"   Range methods: {avg_range_fitness:.8f}")
        print(f"   Single methods: {avg_single_fitness:.8f}")

        if avg_range_fitness < avg_single_fitness:
            improvement = ((avg_single_fitness - avg_range_fitness) / avg_single_fitness) * 100
            print(f"   📈 Range methods are {improvement:.1f}% better on average")
        else:
            improvement = ((avg_range_fitness - avg_single_fitness) / avg_range_fitness) * 100
            print(f"   📉 Single methods are {improvement:.1f}% better on average")

    print("\n💡 Key Insights:")
    print("   • Mutation ranges provide more exploration diversity")
    print("   • Different problems may benefit from different strategies")
    print("   • Range width affects exploration vs exploitation trade-off")
    print("   • Adaptive sampling can escape local optima more effectively")

    plt.show()

if __name__ == "__main__":
    main()
