#!/usr/bin/env python3
"""
Script to check the actual fitted GARCH parameters from the dataset
"""

import torch
import numpy as np
import pandas as pd
from dataset2 import cal_dataset
from hn import HestonNandiGARCH

def main():
    print("🔍 Checking actual GARCH parameters from dataset...")

    # Load the same dataset used in calibration
    dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                          "joint_dataset/assetprices.csv")

    print(f"Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

    # Get the fitted parameters from the dataset
    fitted_params = dataset.target.numpy()
    param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

    print(f"\n📊 Actual fitted GARCH parameters:")
    print("-" * 50)
    for name, value in zip(param_names, fitted_params):
        print(f"{name:<8}: {value:.8f}")

    # Compare with the "true" parameters being used in calibration
    assumed_true = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2])

    print(f"\n🎯 Assumed 'true' parameters (used for comparison):")
    print("-" * 50)
    for name, value in zip(param_names, assumed_true):
        print(f"{name:<8}: {value:.8f}")

    # Calculate differences
    print(f"\n📏 Differences (Fitted - Assumed):")
    print("-" * 50)
    differences = fitted_params - assumed_true
    for name, diff in zip(param_names, differences):
        print(f"{name:<8}: {diff:.8f}")

    # Calculate relative differences
    print(f"\n📈 Relative differences (%):")
    print("-" * 50)
    rel_diffs = (differences / assumed_true) * 100
    for name, rel_diff in zip(param_names, rel_diffs):
        print(f"{name:<8}: {rel_diff:.2f}%")

    # Check stationarity conditions
    alpha, beta = fitted_params[1], fitted_params[2]
    persistence = alpha + beta
    omega_positive = fitted_params[0] > 0

    print(f"\n✅ Validation of fitted parameters:")
    print(f"   ω > 0: {'✅' if omega_positive else '❌'} ({fitted_params[0]:.8f})")
    print(f"   α + β < 1: {'✅' if persistence < 1 else '❌'} ({persistence:.6f})")

    if persistence < 1:
        unconditional_var = fitted_params[0] / (1 - persistence)
        empirical_var = dataset.returns.var().item()
        print(f"   Theoretical var: {unconditional_var:.8f}")
        print(f"   Empirical var: {empirical_var:.8f}")
        print(f"   Var ratio: {unconditional_var/empirical_var:.4f}")

    # Also manually fit to see if we get the same results
    print(f"\n🔧 Manual fitting check:")
    hn = HestonNandiGARCH(dataset.returns.numpy())
    hn.fit()
    manual_params = np.array([hn.omega, hn.alpha, hn.beta, hn.gamma, hn.lambda_])

    print("Manual fit results:")
    for name, value in zip(param_names, manual_params):
        print(f"{name:<8}: {value:.8f}")

    # Check if they match
    params_match = np.allclose(fitted_params, manual_params, rtol=1e-6)
    print(f"\nDataset params match manual fit: {'✅' if params_match else '❌'}")

    # L2 norm between fitted and assumed true
    l2_error = np.linalg.norm(fitted_params - assumed_true)
    print(f"\n🎯 L2 error between fitted and assumed true: {l2_error:.6f}")

    print(f"\n💡 Summary:")
    print(f"The calibration is trying to recover parameters {assumed_true}")
    print(f"But the actual fitted parameters from the data are {fitted_params}")
    print(f"This explains why the optimizer struggles - it's being asked to find")
    print(f"parameters that don't match the data it was trained on!")

if __name__ == "__main__":
    main()
