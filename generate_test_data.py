#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_test_data(num_samples=10, output_path="test_data.csv"):
    """
    Generate test data samples for model evaluation.

    Args:
        num_samples (int): Number of samples to generate
        output_path (str): Path to save the CSV file
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate sample data with reasonable ranges
    data = {
        "S0": np.random.uniform(80, 120, num_samples),  # Stock price
        "m": np.random.uniform(0.7, 1.3, num_samples),  # Moneyness
        "r": np.random.uniform(0.0001, 0.05, num_samples),  # Risk-free rate
        "T": np.random.uniform(0.1, 2.0, num_samples),  # Time to maturity
        "corp": np.random.choice([-1, 1], num_samples),  # Option type (-1: put, 1: call)
        "alpha": np.random.uniform(0.00001, 0.1, num_samples),  # GARCH alpha
        "beta": np.random.uniform(0.4, 0.9, num_samples),  # GARCH beta
        "omega": np.random.uniform(0.0000001, 0.0001, num_samples),  # GARCH omega
        "gamma": np.random.uniform(50, 300, num_samples),  # GARCH gamma
        "lambda": np.random.uniform(-0.4, 0.4, num_samples),  # GARCH lambda
    }

    # Calculate sigma (implied volatility) based on a simplified model for testing
    # This is a very simplified calculation and not theoretically accurate
    data["sigma"] = 0.2 + 0.1 * np.abs(1 - data["m"]) + 0.05 * np.sqrt(data["T"]) + 0.02 * data["alpha"] / data["beta"]

    # Calculate option value (V) based on a simplified model
    # Again, this is a very simplified calculation for testing only
    data["V"] = data["S0"] * data["m"] * data["sigma"] * np.sqrt(data["T"])

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} test samples and saved to {output_path}")

    return df

def main():
    parser = argparse.ArgumentParser(description="Generate test data for model evaluation")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="test_data.csv",
                        help="Path to save the output CSV file")

    args = parser.parse_args()

    # Generate test data
    generate_test_data(args.num_samples, args.output)

    print("\nSample of generated data:")
    df = pd.read_csv(args.output)
    print(df.head(3))

if __name__ == "__main__":
    main()
