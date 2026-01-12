#!/usr/bin/env python3
"""
Test script for fitting Heston–Nandi GARCH(1,1) to dataset returns.

Usage:
    python jointcal/scripts/test_hn_fit.py
    python jointcal/scripts/test_hn_fit.py --dataset <path> --asset <path> --true-params <path>
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from dataset2 import cal_dataset
from hn import HestonNandiGARCH


def load_true_params(path: str) -> np.ndarray | None:
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return np.array([d["omega"], d["alpha"], d["beta"], d["gamma"], d["lambda"]])
    except Exception as e:
        print(f"Could not load true params from {path}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test HN GARCH fit to dataset returns")
    parser.add_argument(
        "--dataset",
        default="joint_dataset/scalable_hn_dataset_250x60.csv",
        help="CSV file with option dataset (default: scalable_hn_dataset_250x60.csv)",
    )
    parser.add_argument(
        "--asset",
        default="joint_dataset/assetprices.csv",
        help="CSV with asset prices for returns (default: assetprices.csv)",
    )
    parser.add_argument(
        "--true-params",
        default="true_params/1.json",
        help="JSON file with ground-truth parameters (default: true_params/1.json)",
    )

    args = parser.parse_args()

    print("Loading dataset...")
    dataset = cal_dataset(args.dataset, args.asset)
    print(f"Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

    print("\nFitting Heston–Nandi GARCH to returns...")
    hn = HestonNandiGARCH(dataset.returns.numpy())
    result = hn.fit()

    print("\nFit summary:")
    print("  Optimizer success:", getattr(result, "success", None))
    print("  Fitted params:", getattr(hn, "fitted_params", None))
    print("  Log-likelihood:", getattr(hn, "log_likelihood", None))

    true_params = load_true_params(args.true_params)
    if true_params is not None and hn.fitted_params is not None:
        l2_err = np.linalg.norm(hn.fitted_params - true_params)
        inf_err = np.linalg.norm(hn.fitted_params - true_params, ord=np.inf)
        print("\nComparison to true parameters:")
        print("  True params:      ", true_params.tolist())
        print("  Two-norm error:   ", l2_err)
        print("  Infinity-norm err:", inf_err)
    else:
        print("\nCould not compute error vs true params (missing ground truth or fit failed).")


if __name__ == "__main__":
    main()
