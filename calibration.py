import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from dataset import CalibrationDataset, IVDataset, train_test_split2
from model import IV, Joint
from loss import joint_loss, HN_cond_var
from iv import train_model as train_iv_model_impl
import json


class GARCHCalibrator:
    """
    Calibrates GARCH parameters using joint log-likelihood from returns and implied volatilities
    """
    def __init__(self, iv_model, joint_model, device):
        self.iv_model = iv_model
        self.joint_model = joint_model
        self.device = device
        self.iv_model.eval()  # IV model stays in eval mode

    def predict_iv(self, option_features):
        """Predict implied volatility for given option features"""
        with torch.no_grad():
            return self.iv_model(option_features)

    def calibrate_garch_params(self, option_features_with_iv):
        """Predict GARCH parameters from option features + IV predictions"""
        with torch.no_grad():
            # Get GARCH parameters (scaled)
            scaled_params = self.joint_model(option_features_with_iv)
            return scaled_params


def train_iv_model(train_iv_dataset, val_iv_dataset, params, device):
    """
    Train IV model to predict implied volatilities from option features
    """
    print("="*60)
    print("Step 1: Training IV Model")
    print("="*60)

    # Create data loaders
    train_loader = DataLoader(
        train_iv_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params.get("num_workers", 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_iv_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params.get("num_workers", 4),
        pin_memory=True
    )

    # Initialize IV model
    iv_model = IV(
        input_features=21,  # Based on IVDataset features
        hidden_size=params.get("hidden_size", 200),
        dropout_rate=params.get("dropout_rate", 0.1),
        num_hidden_layers=params.get("num_hidden_layers", 6)
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        iv_model.parameters(),
        lr=params["lr_iv"],
        weight_decay=params.get("weight_decay", 0.01)
    )

    # Train
    trained_model, train_losses, val_losses = train_iv_model_impl(
        iv_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=params["epochs_iv"]
    )

    print(f"\nIV Model Training Complete. Final Val Loss: {val_losses[-1]:.6f}")
    return trained_model


def augment_data_with_iv_predictions(cal_dataset, iv_model, device):
    """Add IV model predictions to calibration dataset"""
    print("\nGenerating IV predictions for calibration dataset...")

    iv_model.eval()
    iv_predictions = []

    # Get IV predictions for all samples
    with torch.no_grad():
        # Process in batches for efficiency
        batch_size = 256
        for i in range(0, len(cal_dataset.ivds), batch_size):
            batch_features = []
            end_idx = min(i + batch_size, len(cal_dataset.ivds))

            for j in range(i, end_idx):
                X_iv, _ = cal_dataset.ivds[j]
                batch_features.append(X_iv)

            batch_tensor = torch.stack(batch_features).to(device)
            batch_preds = iv_model(batch_tensor).cpu().numpy().flatten()
            iv_predictions.extend(batch_preds)

    # Update dataset
    cal_dataset.data["iv_model"] = iv_predictions
    print(f"Added {len(iv_predictions)} IV predictions")


def train_garch_calibration(train_cal_dataset, val_cal_dataset, params, device):
    """
    Train joint model to calibrate GARCH parameters using joint log-likelihood
    """
    print("\n" + "="*60)
    print("Step 2: Training GARCH Calibration with Joint Loss")
    print("="*60)

    # Initialize joint model
    joint_model = Joint(
        input_features=7,  # 6 base features + 1 IV prediction
        hidden_size=params.get("hidden_size", 200),
        dropout_rate=params.get("dropout_rate", 0.1),
        num_hidden_layers=params.get("num_hidden_layers", 6)
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        joint_model.parameters(),
        lr=params["lr_joint"],
        weight_decay=params.get("weight_decay", 0.01)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Get scaler parameters for differentiable inverse transform
    scaler = train_cal_dataset.target_scaler
    # MinMaxScaler stores min_ and scale_ as arrays
    scale_min = torch.tensor(scaler.min_, device=device, dtype=torch.float32)
    scale_scale = torch.tensor(scaler.scale_, device=device, dtype=torch.float32)

    def inverse_transform_diff(scaled_params):
        """Differentiable inverse MinMaxScaler transform"""
        # Inverse transform: X_original = X_scaled / scale_ + min_
        return scaled_params / scale_scale + scale_min

    # Prepare data loaders
    train_loader = DataLoader(
        train_cal_dataset,
        batch_size=params["batch_size_joint"],
        shuffle=True,
        num_workers=params.get("num_workers", 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_cal_dataset,
        batch_size=params["batch_size_joint"],
        shuffle=False,
        num_workers=params.get("num_workers", 4),
        pin_memory=True
    )

    # Training loop with joint loss
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(params["epochs_joint"]):
        # Training phase
        joint_model.train()
        train_loss = 0
        print(f"\nEpoch {epoch + 1}/{params['epochs_joint']} - Training...")

        for batch_idx, (batch_X, batch_y, N_batch, M_batch) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass - get GARCH parameters
            garch_params_scaled = joint_model(batch_X)

            # Apply differentiable inverse transform
            garch_params = inverse_transform_diff(garch_params_scaled)

            # Calculate joint loss for batch
            batch_losses = []
            for i in range(len(batch_X)):
                # Extract individual parameters (actual values after inverse transform)
                # Note: order in data is [alpha, beta, omega, gamma, lambda]
                alpha_actual = garch_params[i, 0]
                beta_actual = garch_params[i, 1]
                omega_actual = garch_params[i, 2]
                gamma_actual = garch_params[i, 3]
                lambda_actual = garch_params[i, 4]

                # Get corresponding data
                N = N_batch[i].item()
                M = M_batch[i].item()

                # Get returns and risk-free rate for this sample
                idx = batch_idx * train_loader.batch_size + i
                if idx < len(train_cal_dataset.data):
                    returns = torch.tensor(
                        train_cal_dataset.data.iloc[idx]["returns"],
                        device=device,
                        dtype=torch.float32
                    )
                    r = torch.tensor(
                        train_cal_dataset.data.iloc[idx]["r"],
                        device=device,
                        dtype=torch.float32
                    )
                    implied_market = torch.tensor(
                        train_cal_dataset.data.iloc[idx]["sigma"],
                        device=device,
                        dtype=torch.float32
                    ).unsqueeze(0)
                    implied_model = torch.tensor(
                        train_cal_dataset.data.iloc[idx]["iv_model"],
                        device=device,
                        dtype=torch.float32
                    ).unsqueeze(0)

                    # Calculate joint loss using actual parameter values
                    try:
                        loss_i = joint_loss(
                            implied_market, implied_model, M, N,
                            omega_actual, alpha_actual, beta_actual, gamma_actual, lambda_actual,
                            returns, r
                        )
                        batch_losses.append(loss_i)
                    except Exception as e:
                        print(f"\nError in joint loss calculation:")
                        print(f"  omega: {omega_actual.item():.6f}")
                        print(f"  alpha: {alpha_actual.item():.6f}")
                        print(f"  beta: {beta_actual.item():.6f}")
                        print(f"  gamma: {gamma_actual.item():.6f}")
                        print(f"  lambda: {lambda_actual.item():.6f}")
                        print(f"  N: {N}, M: {M}")
                        print(f"  Error: {str(e)}")
                        raise

            if len(batch_losses) > 0:
                batch_loss = torch.stack(batch_losses).mean()
            else:
                print(f"Warning: No valid losses computed for batch {batch_idx}")
                continue
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(joint_model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += batch_loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {batch_loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        joint_model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (batch_X, batch_y, N_batch, M_batch) in enumerate(val_loader):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                garch_params_scaled = joint_model(batch_X)

                # Apply differentiable inverse transform
                garch_params = inverse_transform_diff(garch_params_scaled)

                # Calculate joint loss
                batch_losses = []
                for i in range(len(batch_X)):
                    # Extract individual parameters (actual values after inverse transform)
                    alpha_actual = garch_params[i, 0]
                    beta_actual = garch_params[i, 1]
                    omega_actual = garch_params[i, 2]
                    gamma_actual = garch_params[i, 3]
                    lambda_actual = garch_params[i, 4]

                    N = N_batch[i].item()
                    M = M_batch[i].item()

                    idx = batch_idx * val_loader.batch_size + i
                    if idx < len(val_cal_dataset.data):
                        returns = torch.tensor(
                            val_cal_dataset.data.iloc[idx]["returns"],
                            device=device,
                            dtype=torch.float32
                        )
                        r = torch.tensor(
                            val_cal_dataset.data.iloc[idx]["r"],
                            device=device,
                            dtype=torch.float32
                        )
                        implied_market = torch.tensor(
                            val_cal_dataset.data.iloc[idx]["sigma"],
                            device=device,
                            dtype=torch.float32
                        ).unsqueeze(0)
                        implied_model = torch.tensor(
                            val_cal_dataset.data.iloc[idx]["iv_model"],
                            device=device,
                            dtype=torch.float32
                        ).unsqueeze(0)

                        loss_i = joint_loss(
                            implied_market, implied_model, M, N,
                            omega_actual, alpha_actual, beta_actual, gamma_actual, lambda_actual,
                            returns, r
                        )

                        batch_losses.append(loss_i)

                batch_loss = torch.stack(batch_losses).mean() if batch_losses else torch.zeros(1, device=device)
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = joint_model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{params['epochs_joint']} Complete:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Best Val Loss: {best_val_loss:.6f}")

    # Load best model
    joint_model.load_state_dict(best_model_state)

    return joint_model, train_losses, val_losses


def calibrate_garch_with_joint_loss(data_path, params_path, save_path="garch_calibrator.pt"):
    """
    Main function to calibrate GARCH parameters using joint log-likelihood

    Args:
        data_path: Path to calibration dataset CSV
        params_path: Path to hyperparameters JSON
        save_path: Path to save trained models

    Returns:
        GARCHCalibrator instance
    """
    # Setup
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load data
    print("\nLoading calibration data...")
    data = pd.read_csv(data_path)

    # Create train/validation splits
    train_cal, train_iv, val_cal, val_iv = train_test_split2(
        data,
        test_size=params.get("val_size", 0.2),
        random_state=params.get("random_state", 42)
    )

    print(f"Training samples: {len(train_cal)}")
    print(f"Validation samples: {len(val_cal)}")

    # Step 1: Train IV model
    iv_model = train_iv_model(train_iv, val_iv, params, device)

    # Step 2: Add IV predictions to calibration datasets
    augment_data_with_iv_predictions(train_cal, iv_model, device)
    augment_data_with_iv_predictions(val_cal, iv_model, device)

    # Step 3: Train GARCH calibration model with joint loss
    joint_model, joint_train_losses, joint_val_losses = train_garch_calibration(
        train_cal, val_cal, params, device
    )

    # Create calibrator
    calibrator = GARCHCalibrator(iv_model, joint_model, device)

    # Save models
    print(f"\nSaving calibrator to {save_path}")
    torch.save({
        'iv_model_state_dict': iv_model.state_dict(),
        'joint_model_state_dict': joint_model.state_dict(),
        'joint_train_losses': joint_train_losses,
        'joint_val_losses': joint_val_losses,
        'target_scaler': train_cal.target_scaler,
        'params': params
    }, save_path)

    print("\n" + "="*60)
    print("Calibration Complete!")
    print("="*60)
    print(f"Final Joint Validation Loss: {joint_val_losses[-1]:.6f}")

    return calibrator


def load_garch_calibrator(model_path, device=None):
    """Load a saved GARCH calibrator"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    params = checkpoint['params']

    # Recreate models
    iv_model = IV(
        input_features=21,
        hidden_size=params.get("hidden_size", 200),
        dropout_rate=params.get("dropout_rate", 0.1),
        num_hidden_layers=params.get("num_hidden_layers", 6)
    ).to(device)

    joint_model = Joint(
        input_features=7,
        hidden_size=params.get("hidden_size", 200),
        dropout_rate=params.get("dropout_rate", 0.1),
        num_hidden_layers=params.get("num_hidden_layers", 6)
    ).to(device)

    # Load weights
    iv_model.load_state_dict(checkpoint['iv_model_state_dict'])
    joint_model.load_state_dict(checkpoint['joint_model_state_dict'])

    return GARCHCalibrator(iv_model, joint_model, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate GARCH with Joint Log-Likelihood")
    parser.add_argument("--data", type=str, required=True, help="Path to calibration dataset")
    parser.add_argument("--params", type=str, required=True, help="Path to parameters JSON")
    parser.add_argument("--output", type=str, default="garch_calibrator.pt", help="Output path")

    args = parser.parse_args()

    calibrator = calibrate_garch_with_joint_loss(args.data, args.params, args.output)
