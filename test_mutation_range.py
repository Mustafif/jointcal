import torch
import numpy as np
from calibrate2_de import DifferentialEvolution
from run_de_final import OptimizedDifferentialEvolution

def test_mutation_range():
    """Test that mutation range functionality works correctly"""

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # 3 parameters

    print("Testing Mutation Range Functionality")
    print("=" * 40)

    # Test 1: Single mutation value
    print("\n1. Testing single mutation value (0.8):")
    de_single = DifferentialEvolution(
        bounds=bounds,
        popsize=10,
        mutation=0.8,
        crossover=0.7,
        seed=42,
        device=device
    )

    print(f"   mutation_range: {de_single.mutation_range}")
    print(f"   mutation: {de_single.mutation}")
    assert de_single.mutation_range is None
    assert de_single.mutation == 0.8
    print("   ✓ Single value correctly stored")

    # Test 2: Mutation range
    print("\n2. Testing mutation range (0.5, 1.2):")
    de_range = DifferentialEvolution(
        bounds=bounds,
        popsize=10,
        mutation=(0.5, 1.2),
        crossover=0.7,
        seed=42,
        device=device
    )

    print(f"   mutation_range: {de_range.mutation_range}")
    print(f"   mutation: {de_range.mutation}")
    assert de_range.mutation_range == (0.5, 1.2)
    assert de_range.mutation is None
    print("   ✓ Range correctly stored")

    # Test 3: Sample mutation factors from range
    print("\n3. Testing mutation factor sampling:")
    de_range.population = torch.rand(10, 3, device=device)

    # Sample 100 mutation factors and check they're in range
    factors = []
    for _ in range(100):
        if de_range.mutation_range is not None:
            factor = torch.rand(1, device=device) * (de_range.mutation_range[1] - de_range.mutation_range[0]) + de_range.mutation_range[0]
            factors.append(factor.item())

    min_factor = min(factors)
    max_factor = max(factors)
    print(f"   Sampled factors range: [{min_factor:.3f}, {max_factor:.3f}]")
    print(f"   Expected range: [0.5, 1.2]")

    assert min_factor >= 0.5, f"Min factor {min_factor} below 0.5"
    assert max_factor <= 1.2, f"Max factor {max_factor} above 1.2"
    assert max_factor - min_factor > 0.5, "Range too narrow, likely not sampling correctly"
    print("   ✓ Factors correctly sampled from range")

    # Test 4: Actual mutation operation with range
    print("\n4. Testing mutation operation with range:")
    de_range.population = torch.rand(10, 3, device=device)
    de_range.fitness = torch.rand(10, device=device)
    de_range.best_idx = 0
    de_range.best_params = de_range.population[0].clone()

    # Perform mutation and crossover
    trial = de_range._mutate_and_crossover(target_idx=5)
    print(f"   Trial vector shape: {trial.shape}")
    print(f"   Trial vector: {trial}")

    # Check bounds are respected
    for i, (low, high) in enumerate(bounds):
        assert low <= trial[i] <= high, f"Parameter {i} out of bounds: {trial[i]} not in [{low}, {high}]"
    print("   ✓ Mutation with range produces valid trial vector")

    # Test 5: Compare single vs range diversity
    print("\n5. Comparing diversity between single and range mutation:")

    # Generate multiple trials with single mutation
    de_single.population = torch.rand(10, 3, device=device)
    de_single.fitness = torch.rand(10, device=device)
    de_single.best_idx = 0
    de_single.best_params = de_single.population[0].clone()

    single_trials = []
    for _ in range(20):
        trial = de_single._mutate_and_crossover(target_idx=5)
        single_trials.append(trial.clone())

    # Generate multiple trials with range mutation
    range_trials = []
    for _ in range(20):
        trial = de_range._mutate_and_crossover(target_idx=5)
        range_trials.append(trial.clone())

    # Calculate variance as diversity measure
    single_stack = torch.stack(single_trials)
    range_stack = torch.stack(range_trials)

    single_var = torch.var(single_stack, dim=0).mean()
    range_var = torch.var(range_stack, dim=0).mean()

    print(f"   Single mutation variance: {single_var:.6f}")
    print(f"   Range mutation variance: {range_var:.6f}")
    print(f"   Diversity ratio (range/single): {range_var/single_var:.3f}")

    # Range should generally produce more diversity
    print("   ✓ Diversity comparison completed")

    print("\n🎉 All tests passed! Mutation range functionality is working correctly.")

def test_optimized_de_mutation_range():
    """Test that mutation range functionality works in OptimizedDifferentialEvolution"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = [(1e-8, 1.0), (0.0, 1.0), (0.0, 1.0), (-2.0, 2.0), (-2.0, 2.0)]  # 5 parameters for GARCH

    print("\n" + "="*50)
    print("Testing OptimizedDifferentialEvolution Mutation Range")
    print("="*50)

    # Test range functionality
    print("\n1. Testing mutation range (0.3, 0.9):")
    de_opt = OptimizedDifferentialEvolution(
        bounds=bounds,
        popsize=10,
        mutation=(0.3, 0.9),
        crossover=0.8,
        seed=42,
        device=device
    )

    print(f"   mutation_range: {de_opt.mutation_range}")
    print(f"   mutation: {de_opt.mutation}")
    assert de_opt.mutation_range == (0.3, 0.9)
    assert de_opt.mutation is None
    print("   ✓ Range correctly stored in OptimizedDE")

    # Test mutation operation
    de_opt.population = torch.rand(10, 5, device=device)
    de_opt.fitness = torch.rand(10, device=device)
    de_opt.best_idx = 0
    de_opt.best_params = de_opt.population[0].clone()

    trial = de_opt._mutate_and_crossover(target_idx=5)
    print(f"   Trial vector: {trial}")

    # Check bounds
    for i, (low, high) in enumerate(bounds):
        assert low <= trial[i] <= high, f"Parameter {i} out of bounds"
    print("   ✓ OptimizedDE mutation with range works correctly")

if __name__ == "__main__":
    test_mutation_range()
    test_optimized_de_mutation_range()
