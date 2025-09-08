#!/usr/bin/env python3
"""
Example script demonstrating how to use the calibration system for ANN with IV and joint loss.

This script shows:
1. How to prepare your calibration dataset
2. How to run the two-stage calibration process
3. How to use the calibrated model for predictions
4. How to evaluate the calibration performance
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calibration import calibrate_model, load_calibrated_model
from dataset import CalibrationDataset, IVDataset
import json


def prepare_calibration_data():
    """
    Example function showing the expected format for calibration data.

    Your calibration dataset should contain:
    - Option data: S0, m, r, T, corp, V (option value)
    - Market implied volatility: sigma
    - Historical returns: returns
    - GARCH parameters to calibrate: alpha, beta, omega, gamma, lambda
    """
    # Example data structure (replace with your actual data)
    data = {
        # Option characteristics
        'S0': np.random.uniform(90, 110, 1000),  # Spot price
        'm': np.random.uniform(0.8, 1.2, 1000),   # Moneyness (K/S0)
        'r': np.random.uniform(0.01, 0.05, 1000), # Risk-free rate
        'T': np.random.uniform(0.1, 2.0, 1000),   # Time to maturity
        'corp': np.random.choice([-1, 1], 1000),  # Put/Call indicator
        'V': np.random.uniform(1, 20, 1000),      # Option value

        # Market data
        'sigma': np.random.uniform(0.1, 0.5, 1000),  # Market implied volatility
        'returns': np.random.normal(0, 0.02, 1000),  # Historical returns

        # Initial GARCH parameter estimates (these will be calibrated)
        'alpha': np.random.uniform(0.05, 0.15, 1000),
        'beta': np.random.uniform(0.8, 0.95, 1000),
        'omega': np.random.uniform(0.0001, 0.001, 1000),
        'gamma': np.random.uniform(-0.5, 0.5, 1000),
        'lambda': np.random.uniform(0.01, 0.1, 1000),
    }

    df = pd.DataFrame(data)
    return df


def run_calibration_example():
    """
    Complete example of the calibration process
    """
    print("="*60)
    print("ANN Calibration Example with IV and Joint Loss")
    print("="*60)

    # Step 1: Prepare your data
    print("\n1. Preparing calibration data...")
    calibration_data = prepare_calibration_data()
    calibration_data.to_csv('calibration_data.csv', index=False)
    print(f"   Created dataset with {len(calibration_data)} samples")
    print(f"   Features: {list(calibration_data.columns)}")

    # Step 2: Set up calibration parameters
    print("\n2. Setting up calibration parameters...")
    params = {
        "lr": 0.001,              # Learning rate for IV model
        "lr_joint": 0.0005,       # Learning rate for joint model
        "weight_decay": 0.01,     # L2 regularization
        "batch_size": 256,        # Batch size
        "epochs_iv": 50,          # Epochs for IV model training
        "epochs_joint": 100,      #
