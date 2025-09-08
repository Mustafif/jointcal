# GARCH Calibration with Joint Log-Likelihood

## Overview

This system implements GARCH parameter calibration using a joint log-likelihood approach that combines information from both historical returns and option-implied volatilities. This method provides more accurate calibration than traditional approaches that use only returns data.

## Key Features

- **Joint Loss Function**: Combines returns likelihood and implied volatility likelihood
- **Two-Stage Neural Network**: 
  1. IV prediction model for implied volatility
  2. Joint calibration model for GARCH parameters
- **Automatic Returns Calculation**: Computes log returns from spot price data
- **Differentiable Parameter Transformation**: Maintains gradient flow through parameter scaling

## How It Works

### 1. Data Preparation
- Calculates log returns from unique S0 values: `ln(S_t/S_{t-1})`
- Structures option data with features: spot, moneyness, rate, maturity, etc.

### 2. Stage 1: IV Model Training
- Neural network trained to predict implied volatilities from option features
- Uses MSE loss for accurate IV prediction

### 3. Stage 2: Joint Calibration
- Uses IV predictions as additional features
- Outputs GARCH parameters (α, β, ω, γ, λ)
- Optimizes joint log-likelihood:
  ```
  L_joint = L_returns(θ|returns) + L_IV(θ|market_IV, model_IV)
  ```

### 4. GARCH Model
Implements Heston-Nandi GARCH with parameters:
- **ω** (omega): Constant term
- **α** (alpha): ARCH coefficient  
- **β** (beta): GARCH coefficient
- **γ** (gamma): Asymmetry parameter
- **λ** (lambda): Risk premium

Conditional variance: `h_t = ω + βh_{t-1} + α(z_{t-1} - γ√h_{t-1})²`

## Usage

### Training

```python
from calibration import calibrate_garch_with_joint_loss

# Prepare parameters
params = {
    "lr_iv": 0.001,
    "lr_joint": 0.0001,
    "batch_size": 256,
    "batch_size_joint": 32,
    "epochs_iv": 100,
    "epochs_joint": 200,
    "dropout_rate": 0.1,
    "hidden_size": 200,
    "num_hidden_layers": 6,
    "num_workers": 4,
    "val_size": 0.2
}

# Save params to JSON
import json
with open('params.json', 'w') as f:
    json.dump(params, f)

# Run calibration
calibrator = calibrate_garch_with_joint_loss(
    'your_data.csv',
    'params.json',
    'calibrator.pt'
)
```

### Command Line

```bash
python calibration.py --data datasets/your_data.csv --params params.json --output model.pt
```

### Loading and Using

```python
from calibration import load_garch_calibrator

# Load trained model
calibrator = load_garch_calibrator('model.pt')

# Predict implied volatility
iv_pred = calibrator.predict_iv(option_features)

# Get calibrated GARCH parameters
garch_params = calibrator.calibrate_garch_params(features_with_iv)
```

## Data Format

Your CSV should contain:
- **S0**: Spot price
- **m**: Moneyness (K/S0)
- **r**: Risk-free rate
- **T**: Time to maturity
- **corp**: Put/Call indicator (-1/1)
- **sigma**: Market implied volatility
- **V**: Option value
- **alpha, beta, omega, gamma, lambda**: True GARCH parameters (for training)

The system automatically calculates returns from the unique S0 values.

## Model Architecture

### IV Model
- Input: 21 features (option characteristics + GARCH params)
- Hidden layers: 6 layers with 200 neurons each
- Activation: Mish with LayerNorm
- Output: Implied volatility (softplus activation)

### Joint Model  
- Input: 7 features (option chars + IV prediction)
- Hidden layers: 6 layers with 200 neurons each
- Output: 5 GARCH parameters with constraints:
  - α ∈ (0, 0.5) via sigmoid
  - β ∈ (0, 0.9) via sigmoid
  - ω > 0 via softplus
  - γ ∈ (-2, 2) via tanh
  - λ ∈ (-0.5, 0.5) via tanh

## Loss Functions

### Returns Loss
```python
L_returns = -0.5 * Σ[log(h_t) + (r_t - μ_t)²/h_t]
```

### Implied Volatility Loss
```python
L_IV = -0.5 * Σ[log(ε) + (σ_market - σ_model)²/ε²]
```

### Joint Loss
```python
L_joint = w_returns * L_returns + w_IV * L_IV
```

## Key Improvements Over Traditional Methods

1. **Information Fusion**: Combines backward-looking (returns) and forward-looking (options) information
2. **Neural Network Flexibility**: Captures non-linear relationships in option pricing
3. **Automatic Differentiation**: Ensures proper gradient flow for optimization
4. **Parameter Constraints**: Enforces valid GARCH parameter ranges
5. **Robust Optimization**: Handles numerical stability with proper scaling

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Processing**: Efficient mini-batch training
- **Learning Rate Scheduling**: Adaptive learning rates for convergence
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Monitors validation loss to prevent overfitting

## Troubleshooting

### Common Issues

1. **Non-stationary parameters**: The model automatically adjusts parameters to ensure stationarity
2. **Negative variances**: Built-in safeguards ensure positive conditional variances
3. **Memory issues**: Reduce batch_size_joint if running out of GPU memory
4. **Slow convergence**: Adjust learning rates or increase epochs

### Validation

Check calibration quality by:
- Comparing predicted vs market implied volatilities
- Examining parameter stability across different samples
- Validating returns distribution under calibrated parameters

## References

- Heston, S. L., & Nandi, S. (2000). A closed-form GARCH option valuation model
- Joint calibration methodology inspired by hybrid ML-econometric approaches
- Neural network architecture based on modern deep learning best practices