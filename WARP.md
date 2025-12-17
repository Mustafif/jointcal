# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**jointcal** is a PyTorch-based implementation for joint calibration of Heston-Nandi GARCH parameters using both historical returns and option-implied volatilities. The project combines neural network-based implied volatility prediction with differential evolution optimization for parameter calibration.

## Development Commands

### Testing
```bash
# Run all tests (no pytest/unittest framework - uses simple test scripts)
python test_calibration.py
python test_residuals.py
python test_mutation_range.py

# Test specific calibration components
python test_calibration.py  # Tests single calibration step and batch processing
```

### Training & Calibration
```bash
# Main differential evolution calibration (SciPy DE - recommended)
python de.py

# Run calibration with specific approaches
python calibrate_scipy_de.py      # SciPy DE (production-ready)
python calibrate2_de.py           # Custom PyTorch DE (legacy)
python calibrate_gpu_de.py        # GPU-accelerated DE (experimental)

# Example workflows
python example_de_calibration.py
python example_garch_calibration.py
```

### Model Evaluation
```bash
# Evaluate a trained model
bash run_evaluation.sh

# Or directly
python evaluate_model.py --model saved_models/MODEL_PATH/model.pt --data DATA.csv --output results.csv
```

### Data Generation
```bash
# Generate synthetic test data
python generate_test_data.py
```

## Architecture Overview

### Two-Stage Calibration System

**Stage 1: Implied Volatility (IV) Prediction**
- Neural network models (`IV`, `IV_GLU`, `IV_WideThenDeep` in `model.py`) predict implied volatilities from option features
- Input: Option characteristics (S0, moneyness, rate, maturity, etc.) + GARCH params
- Output: Implied volatility (σ)
- Models use Mish activation, LayerNorm, and Softplus output for positive IV values

**Stage 2: Joint Calibration**
- Calibrates 5 Heston-Nandi GARCH parameters: ω (omega), α (alpha), β (beta), γ (gamma), λ (lambda)
- Uses joint log-likelihood combining returns data and option-implied volatilities
- Optimization via differential evolution (gradient-free global optimization)

### Core Components

**Models** (`model.py`):
- `IV`: Standard deep MLP with 6 hidden layers (200 neurons each)
- `IV_GLU`: GLU-based architecture with residual blocks for better gradient flow
- `IV_WideThenDeep`: Funnel architecture (512→256→128→64→32→16)
- `Joint`: Joint calibration model that outputs GARCH parameters

**Loss Functions** (`loss.py`):
- `returns_loss()`: Negative log-likelihood for return series under Heston-Nandi GARCH
- `implied_loss()`: Negative log-likelihood for option-implied volatilities
- `joint_loss()`: Weighted combination of returns and implied volatility losses
- `HN_cond_var()`: Computes conditional variance sequence for GARCH process

**GARCH Model** (`hn.py`):
- `HestonNandiGARCH`: Full Heston-Nandi GARCH(1,1) with risk premium λ
- Variance equation: h_t = ω + β·h_{t-1} + α·(z_{t-1} - γ·√h_{t-1})²
- Mean equation: r_t = λ·h_t + ε_t where ε_t = √h_t · z_t
- Uses `arch` library for initial parameter estimates, then optimizes with L-BFGS-B

**Datasets** (`dataset2.py`):
- `CalibrationDataset`: Main dataset class for joint calibration
- Automatically engineers features: log_moneyness, time transformations, interaction terms
- Combines option data with return series for joint likelihood computation
- `cal_dataset()`: Factory function for creating calibration datasets

**Calibration Methods**:
- `calibrate_scipy_de.py`: **Production-ready** - Uses SciPy's differential_evolution (recommended)
- `calibrate2_de.py`: Custom PyTorch DE implementation (legacy, higher maintenance)
- `calibrate_gpu_de.py`: GPU-accelerated DE (experimental, 10-50x speedup potential)
- `calibrate_metade.py`: Failed MetaDE attempt (JAX version conflicts)

### Differential Evolution (DE) Implementation

The project uses **SciPy DE as the recommended production approach** (see `DE_IMPLEMENTATION_SUMMARY.md`):

**Why Differential Evolution?**
- Gradient-free: No need for backpropagation through GARCH likelihood
- Global optimization: Better at avoiding local minima than gradient descent
- Robust to discontinuities and non-smooth objectives
- Natural parameter bound handling

**Key Hyperparameters**:
- Population size: 15-20 (multiplier, actual = popsize × num_params)
- Max iterations: 300-500
- Strategy: 'best1bin' (fast and reliable)
- Mutation: (0.5, 1.0) for adaptive exploration
- Recombination: 0.7
- Optional L-BFGS-B polishing for final refinement

**Parameter Bounds**:
```python
bounds = [
    (1e-7, 1e-6),      # omega: positive, small
    (1.15e-6, 1.36e-6), # alpha: small, positive
    (0.7, 0.99),        # beta: close to 1 (persistence)
    (0, 10),            # gamma: leverage effect
    (0.2, 0.6)          # lambda: risk premium
]
```

**Stationarity constraint**: α + β < 1 (enforced via projection)

## Data Flow

1. **Data Loading**: CSV files with option data and returns → `CalibrationDataset`
2. **Feature Engineering**: Automatic computation of derived features (log transformations, interactions)
3. **Model Training**: IV prediction model trained via gradient descent (Stage 1)
4. **Parameter Calibration**: DE optimization minimizes joint loss over GARCH parameters (Stage 2)
5. **Validation**: Compare calibrated params to true params (if available) via L2 norm

## File Organization

**Core Implementation**:
- `model.py`: Neural network architectures
- `loss.py`: Loss functions for joint calibration
- `hn.py`: Heston-Nandi GARCH model
- `dataset2.py`: Dataset classes and data loading
- `de.py`: Main entry point for multi-dataset DE calibration

**Calibration Scripts**:
- `calibrate_scipy_de.py`: SciPy DE calibration (recommended)
- `calibrate2_de.py`: Custom PyTorch DE (legacy)
- `calibrate_gpu_de.py`: GPU-accelerated DE (experimental)
- `cal_loss.py`: Calibration loss computation

**Training & Evaluation**:
- `joint.py`: Training loop for joint models
- `evaluate_model.py`: Model evaluation on test data
- `run_evaluation.sh`: Bash script for evaluation pipeline

**Example Scripts**:
- `example_de_calibration.py`: DE usage examples
- `example_garch_calibration.py`: GARCH calibration examples
- `run_de_*.py`: Various DE calibration experiments

**Data Directories**:
- `datasets/`: Primary datasets
- `joint_dataset/`: Joint calibration datasets with returns
- `saved_models/`: Trained model checkpoints (organized by dataset/date)
- `calibration_results/`: DE calibration output
- `true_params/`: Ground truth parameters for validation

## Key Design Patterns

### Parameter Projection
Always project parameters to valid domain after optimization steps:
```python
def project_parameters(params):
    omega = torch.clamp(params[0], min=1e-8)
    alpha = torch.clamp(params[1], min=0.0, max=1.0)
    beta = torch.clamp(params[2], min=0.0, max=1.0)
    # gamma and lambda unconstrained
    return torch.stack([omega, alpha, beta, params[3], params[4]])
```

### Joint Loss Weighting
The joint loss balances returns and implied volatility contributions:
```python
# Weighted by sample sizes to prevent imbalance
joint_loss = (N+M)/(2*N) * returns_loss + (N+M)/(2*M) * implied_loss
```

### Model Device Management
Always move tensors to the correct device (GPU if available):
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X = X.to(device)
```

### Numerical Stability
Add small epsilon values to prevent log(0) and division by zero:
```python
h_safe = h + 1e-8
log_likelihood = -0.5 * torch.sum(torch.log(h_safe) + residuals**2 / h_safe)
```

## Common Workflows

### Running Joint Calibration on New Dataset
1. Prepare CSV with columns: S0, m, r, T, corp, V, sigma, alpha, beta, omega, gamma, lambda
2. Prepare returns CSV with asset prices
3. Add dataset config to `loaders/loader.json`
4. Run: `python de.py` (processes all datasets in loader.json)

### Comparing Calibration Methods
```bash
# Compare SciPy DE vs Custom DE vs GPU DE
python compare_custom_vs_scipy.py
python compare_de_methods.py
python benchmark_de.py
```

### Training New IV Model
```python
# Use the training loop in joint.py or run model-specific training
python run_model.py  # Check this file for training configuration
```

### Debugging Calibration Issues
1. Run `test_calibration.py` to verify components work individually
2. Check parameter bounds and stationarity condition
3. Verify dataset shapes match expected (features, returns, sigma)
4. Monitor loss convergence history
5. Compare calibrated params to HN GARCH fitted params for sanity check

## Performance Considerations

**For Production Calibration**:
- Use SciPy DE (`calibrate_scipy_de.py`) - best balance of speed/accuracy/reliability
- Population size 15-20, iterations 300-500
- Enable polishing for final refinement: `polish=True`

**For Research/Experiments**:
- Custom PyTorch DE if you need to modify the DE algorithm
- GPU DE if you have CUDA-capable GPU and large populations (80+)

**Memory Management**:
- Batch size for training: 32-256 depending on GPU memory
- DE population limited by dataset size and GPU memory
- Use `torch.cuda.empty_cache()` between runs if memory issues

## Important Notes

- **No pytest/unittest**: Tests are standalone Python scripts (test_*.py)
- **Device compatibility**: Code automatically detects and uses CUDA if available
- **Stationarity**: Always verify α + β < 1 after calibration
- **True parameters**: If available in `true_params/`, used for L2 error validation
- **SciPy DE is recommended**: Custom implementations exist for research but SciPy is production-ready
- **Model architecture**: Multiple IV model options (IV, IV_GLU, IV_WideThenDeep) - IV_GLU recommended
- **Dataset format**: Requires both option data CSV and returns CSV for joint calibration
