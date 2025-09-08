#!/usr/bin/env python3
"""
Example script demonstrating GARCH calibration using joint log-likelihood
from both returns and implied volatility data.

This approach calibrates GARCH parameters more accurately by leveraging
information from both historical returns and option-implied volatilities.
"""

import torch
import pandas as pd
import numpy as np
from calibration import calibrate_garch_with_joint_loss, load_garch_calibrator
import matplotlib.pyplot as plt
import json


def generate_sample_data():
    """
    Generate synthetic calibration data for demonstration.
    In practice, this would be real market data.
    """
    np.random.seed(42)
    n_samples = 1000

    # Option features
    S0 = np.random.uniform(90, 110, n_samples)  # Spot price
    m = np.random.uniform(0.8, 1.2, n_samples)  # Moneyness
    r = np.random.uniform(0.01, 0.05, n_samples)  # Risk-free rate
    T = np.random.uniform(0.1, 2.0, n_samples)  # Time to maturity
    corp = np.random.choice([-1, 1], n_samples)  # Put/Call indicator

    # True GARCH parameters (what we're trying to calibrate)
    omega_true = 0.00001
    alpha_true = 0.08
    beta_true = 0.9
    gamma_true = 0.1
    lambda_true = 0.05

    # Generate synthetic implied volatilities
    base_vol = np.sqrt(omega_true / (1 - alpha_true - beta_true))
    sigma = base_vol * (1 + 0.1 * np.random.randn(n_samples))

    # Option values (simplified Black-Scholes-like)
    V = S0 * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * np.random.randn(n_samples))

    # Returns data
    returns = np.random.randn(n_samples) * sigma * np.sqrt(1/252)

    # Create DataFrame
    data = pd.DataFrame({
        'S0': S0,
        'm': m,
        'r': r,
        'T': T,
        'corp': corp,
        'sigma': sigma,
        'V': V,
        'returns': returns,
        'omega': omega_true,
        'alpha': alpha_true,
        'beta': beta_true,
        'gamma': gamma_true,
        'lambda': lambda_true
    })

    return data


def evaluate_calibration(calibrator, test_data):
    """
    Evaluate calibration accuracy on test data
    """
    from dataset import CalibrationDataset
    from torch.utils.data import DataLoader

    # Create test dataset
    test_dataset = CalibrationDataset(test_data)

    # First, generate IV predictions
    print("\nGenerating IV predictions for test data...")
    iv_predictions = []
    calibrator.iv_model.eval()

    with torch.no_grad():
        for i in range(len(test_dataset.ivds)):
            X_iv, _ = test_dataset.ivds[i]
            X_iv = X_iv.unsqueeze(0).to(calibrator.device)
            iv_pred = calibrator.iv_model(X_iv).cpu().item()
            iv_predictions.append(iv_pred)

    test_dataset.data["iv_model"] = iv_predictions

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Collect predictions
    all_predictions = []
    all_targets = []

    calibrator.joint_model.eval()
    with torch.no_grad():
        for batch_X, batch_y, _, _ in test_loader:
            batch_X = batch_X.to(calibrator.device)

            # Get GARCH parameter predictions
            garch_params_scaled = calibrator.joint_model(batch_X)

            # Inverse transform to get actual parameters
            # Create DataFrames with proper column names to avoid warnings
            garch_params_df = pd.DataFrame(
                garch_params_scaled.cpu().numpy(),
                columns=['alpha', 'beta', 'omega', 'gamma', 'lambda']
            )
            garch_params = test_dataset.target_scaler.inverse_transform(garch_params_df)

            # Get true parameters
            true_params_df = pd.DataFrame(
                batch_y.cpu().numpy(),
                columns=['alpha', 'beta', 'omega', 'gamma', 'lambda']
            )
            true_params = test_dataset.target_scaler.inverse_transform(true_params_df)

            all_predictions.append(garch_params)
            all_targets.append(true_params)

    # Concatenate all predictions
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Calculate metrics
    param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
    print("\nCalibration Performance:")
    print("-" * 50)

    for i, param in enumerate(param_names):
        pred = predictions[:, i]
        true = targets[:, i]

        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100

        print(f"\n{param.upper()}:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Mean True: {np.mean(true):.6f}")
        print(f"  Mean Pred: {np.mean(pred):.6f}")

    return predictions, targets, param_names


def plot_calibration_results(predictions, targets, param_names):
    """
    Visualize calibration results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]

        # Scatter plot of predictions vs true values
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=10)

        # Perfect calibration line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Calibration')

        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Predicted {param}')
        ax.set_title(f'{param.upper()} Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('garch_calibration_results.png', dpi=300, bbox_inches='tight')
    print("\nCalibration plot saved to 'garch_calibration_results.png'")


def main():
    """
    Main example workflow
    """
    print("GARCH Calibration with Joint Log-Likelihood Example")
    print("=" * 60)

    # Generate or load data
    print("\n1. Generating sample calibration data...")
    data = generate_sample_data()

    # Save data for reproducibility
    data.to_csv('sample_calibration_data.csv', index=False)
    print(f"   Saved {len(data)} samples to 'sample_calibration_data.csv'")

    # Create parameters file if it doesn't exist
    params = {
        "lr_iv": 0.001,
        "lr_joint": 0.0001,
        "weight_decay": 0.01,
        "batch_size": 256,
        "batch_size_joint": 32,
        "epochs_iv": 50,  # Reduced for example
        "epochs_joint": 100,  # Reduced for example
        "dropout_rate": 0.1,
        "hidden_size": 200,
        "num_hidden_layers": 6,
        "num_workers": 4,
        "val_size": 0.2,
        "random_state": 42
    }

    with open('example_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    print("   Created 'example_params.json'")

    # Run calibration
    print("\n2. Running GARCH calibration...")
    calibrator = calibrate_garch_with_joint_loss(
        'sample_calibration_data.csv',
        'example_params.json',
        'example_garch_calibrator.pt'
    )

    # Generate test data
    print("\n3. Evaluating on test data...")
    test_data = generate_sample_data()  # In practice, use separate test data
    predictions, targets, param_names = evaluate_calibration(calibrator, test_data)

    # Visualize results
    print("\n4. Plotting results...")
    plot_calibration_results(predictions, targets, param_names)

    # Example of using the calibrator for new data
    print("\n5. Example prediction for new option:")
    new_option = pd.DataFrame({
        'S0': [100],
        'm': [1.0],
        'r': [0.03],
        'T': [0.5],
        'corp': [1],
        'sigma': [0.2],  # Market IV
        'V': [10],
        'returns': [0.01],
        'omega': [0],  # Dummy values
        'alpha': [0],
        'beta': [0],
        'gamma': [0],
        'lambda': [0]
    })

    # Create dataset for new option
    from dataset import CalibrationDataset
    new_dataset = CalibrationDataset(new_option)

    # Get IV prediction
    calibrator.iv_model.eval()
    X_iv, _ = new_dataset.ivds[0]
    X_iv = X_iv.unsqueeze(0).to(calibrator.device)
    iv_pred = calibrator.iv_model(X_iv).cpu().item()
    new_dataset.data["iv_model"] = [iv_pred]

    # Get calibrated GARCH parameters
    X_cal, _, _, _ = new_dataset[0]
    X_cal = X_cal.unsqueeze(0).to(calibrator.device)

    calibrator.joint_model.eval()
    with torch.no_grad():
        garch_params_scaled = calibrator.joint_model(X_cal)
        garch_params_df = pd.DataFrame(
            garch_params_scaled.cpu().numpy(),
            columns=['alpha', 'beta', 'omega', 'gamma', 'lambda']
        )
        garch_params = new_dataset.target_scaler.inverse_transform(garch_params_df)

    print("\nCalibrated GARCH parameters:")
    for i, param in enumerate(param_names):
        print(f"  {param}: {garch_params[0, i]:.6f}")

    print("\n" + "=" * 60)
    print("Example complete! The calibrator uses joint log-likelihood to")
    print("leverage both returns and implied volatility information for")
    print("more accurate GARCH parameter calibration.")


if __name__ == "__main__":
    main()
