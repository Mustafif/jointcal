# Differential Evolution GARCH Calibration

This directory contains a PyTorch-based implementation of differential evolution for calibrating Heston-Nandi GARCH parameters. This provides an alternative to gradient-based optimization methods for parameter estimation.

## Overview

The differential evolution (DE) implementation offers several advantages over traditional gradient-based methods:

- **Gradient-free optimization**: No need for gradient computation or backpropagation
- **Global optimization**: Better at avoiding local minima compared to gradient descent
- **Robust to discontinuities**: Can handle non-smooth objective functions
- **Parameter bounds**: Natural handling of parameter constraints
- **Population-based**: Explores multiple solutions simultaneously

## Files

### Core Implementation

- **`calibrate2_de.py`**: Main differential evolution calibration implementation
  - `DifferentialEvolution` class: PyTorch-based DE optimizer
  - `calibrate_de()` function: GARCH parameter calibration using DE
  - `project_parameters()`: Parameter constraint enforcement

- **`compare_calibration.py`**: Comparison between gradient-based and DE methods
  - Side-by-side performance comparison
  - Timing and accuracy analysis
  - Visualization of results

- **`example_de_calibration.py`**: Usage examples and demonstrations
  - Simple DE calibration example
  - Custom DE implementation usage
  - Hyperparameter tuning demonstration

## Quick Start

### Basic Usage

```python
from calibrate2_de import calibrate_de
from dataset2 import cal_dataset
import torch

# Load model and dataset
model = torch.load("path/to/model.pt")
dataset = cal_dataset("options.csv", "returns.csv")

# Run DE calibration
params, history = calibrate_de(
    model=model,
    dataset=dataset,
    popsize=50,      # Population size
    max_iter=500,    # Maximum iterations
    mutation=0.8,    # Mutation factor (or (0.5, 1.0) for range)
    crossover=0.7,   # Crossover probability
    seed=42          # Random seed
)

print("Calibrated parameters:", params)
```

### Comparing Methods

```python
from compare_calibration import compare_calibration_methods

# Run comprehensive comparison
compare_calibration_methods()
```

## Algorithm Details

### Differential Evolution Strategy

The implementation uses the **DE/rand/1/bin** strategy:

1. **Mutation**: `v = x_r1 + F * (x_r2 - x_r3)`
2. **Crossover**: Binomial crossover with probability CR
3. **Selection**: Greedy selection based on fitness

### Parameter Bounds

The GARCH parameters are constrained as follows:

| Parameter | Symbol | Bounds | Description |
|-----------|--------|--------|-------------|
| omega | ω | (1e-8, 1e-3) | Unconditional variance component |
| alpha | α | (0, 1) | ARCH coefficient |
| beta | β | (0, 1) | GARCH coefficient |
| gamma | γ | (-10, 10) | Asymmetry parameter |
| lambda | λ | (-1, 1) | Risk premium parameter |

Additional constraint: α + β < 1 (stationarity condition)

### Objective Function

The calibration minimizes the negative joint log-likelihood:

```
L = -[(N+M)/(2N) * L_returns + (N+M)/(2M) * L_options]
```

Where:
- `L_returns`: Heston-Nandi GARCH log-likelihood for returns
- `L_options`: Log-likelihood for option-implied volatilities
- `N`: Number of return observations
- `M`: Number of option observations

## Hyperparameter Tuning

### Key Parameters

- **Population Size (`popsize`)**: 30-100
  - Larger populations explore more thoroughly but are slower
  - Recommended: 50 for most problems

- **Mutation Factor (`mutation`)**: 0.5-1.2 (single value) or tuple (min, max)
  - Controls exploration vs exploitation
  - Higher values = more exploration
  - Can be single value (e.g., 0.8) or range (e.g., (0.5, 1.0))
  - Range sampling increases diversity and exploration
  - Recommended: 0.8 or (0.6, 1.0)

- **Crossover Probability (`crossover`)**: 0.3-0.9
  - Probability of inheriting from mutant vector
  - Higher values = more recombination
  - Recommended: 0.7

- **Maximum Iterations (`max_iter`)**: 200-1000
  - Depends on problem complexity
  - Monitor convergence history
  - Recommended: 500

### Adaptive Strategies

The implementation includes several adaptive features:

- **Convergence detection**: Early stopping when improvement stagnates
- **Bounds enforcement**: Parameter projection to valid domains
- **Population diversity**: Maintains exploration capability

## Performance Comparison

Typical performance characteristics:

| Method | Speed | Robustness | Global Optimum | Implementation |
|--------|-------|------------|----------------|---------------|
| Gradient-based | ⚡⚡⚡ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Differential Evolution | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

### When to Use DE

**Use Differential Evolution when:**
- The objective function is non-smooth or discontinuous
- You suspect multiple local minima
- Gradient computation is unreliable
- You need robust parameter estimation
- You have sufficient computational budget

**Use Gradient-based methods when:**
- Speed is critical
- The objective function is smooth and well-behaved
- You have good initial parameter estimates
- Computational resources are limited

## Examples

### Configuration Examples

```python
# Fast convergence (less thorough)
calibrate_de(model, dataset, popsize=30, max_iter=200, mutation=0.5, crossover=0.5)

# Balanced approach (recommended)
calibrate_de(model, dataset, popsize=50, max_iter=500, mutation=0.8, crossover=0.7)

# Thorough search with range sampling (slower but more robust)
calibrate_de(model, dataset, popsize=100, max_iter=1000, mutation=(0.7, 1.2), crossover=0.9)

# High exploration with mutation range
calibrate_de(model, dataset, popsize=50, max_iter=500, mutation=(0.5, 1.0), crossover=0.7)
```

### Custom Objective Function

```python
from calibrate2_de import DifferentialEvolution

# Define custom bounds
bounds = [(1e-8, 1e-3), (0, 1), (0, 1), (-10, 10), (-1, 1)]

# Initialize DE optimizer
de = DifferentialEvolution(bounds, popsize=50, device=device)

# Define objective function
def custom_objective(params):
    # Your custom loss function here
    return loss_value

# Run optimization
best_params, best_fitness, history = de.optimize(custom_objective, max_iter=500)
```

## Output Files

The calibration scripts generate several output files:

- **`calibrated_params_de.json`**: DE calibration results with convergence history
- **`calibration_comparison.json`**: Detailed comparison between methods
- **`de_convergence.png`**: Convergence plot for DE optimization
- **`calibration_comparison.png`**: Visual comparison of methods
- **`parameter_errors.png`**: Parameter-wise error analysis

## Troubleshooting

### Common Issues

1. **Slow Convergence**
   - Reduce population size
   - Increase mutation factor
   - Check parameter bounds

2. **Poor Parameter Estimates**
   - Increase population size
   - Increase maximum iterations
   - Adjust mutation/crossover parameters

3. **Memory Issues**
   - Reduce population size
   - Use CPU instead of GPU for large populations
   - Process data in smaller batches

### Debugging

Enable verbose output to monitor progress:

```python
calibrate_de(model, dataset, verbose=True)
```

Check convergence history for stagnation:

```python
params, history = calibrate_de(model, dataset)
import matplotlib.pyplot as plt
plt.plot(history)
plt.yscale('log')
plt.show()
```

## Dependencies

- PyTorch >= 1.8
- NumPy
- Pandas
- Matplotlib (for plotting)
- Scikit-learn (for preprocessing)

## License

This implementation is part of the joint calibration project. See the main project README for license information.

## References

1. Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces.
2. Das, S., & Suganthan, P. N. (2011). Differential evolution: a survey of the state-of-the-art.
3. Heston, S. L., & Nandi, S. (2000). A closed-form GARCH option valuation model.