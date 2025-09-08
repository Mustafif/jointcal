#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

from model import IV_GLU  # Importing only the IV_GLU model as specified


def load_model(model_path):
    """
    Load a saved PyTorch IV_GLU model.

    Args:
        model_path (str): Path to the model.pt file

    Returns:
        model: Loaded PyTorch model
    """
    # Create IV_GLU model with default parameters
    model = IV_GLU(input_features=30, hidden_size=350)

    try:
        # Try loading as a regular state dict
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError as e:
        if "Cannot use ``weights_only=True`` with TorchScript archives" in str(e):
            # Try loading as a TorchScript model with weights_only=False
            print("Detected TorchScript archive, loading with weights_only=False")
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            # If it's another error, re-raise it
            raise

    model.eval()  # Set model to evaluation mode
    print(f"Loaded IV_GLU model from {model_path}")
    return model


def prepare_data(data_path):
    """
    Prepare data from CSV file for IV_GLU model evaluation.

    Args:
        data_path (str): Path to the CSV data file

    Returns:
        tuple: (X, raw_data) - Prepared input tensor and raw dataframe
    """
    # Load the data
    data = pd.read_csv(data_path)
    print(f"Loaded data from {data_path} with {len(data)} rows")

    # For IV_GLU model, we need to prepare features similar to IVDataset
    # Creating derived features
    data["strike"] = data["S0"] * data["m"]
    data["returns"] = 0.0  # Placeholder
    data["log_moneyness"] = np.log(data["m"])
    data["moneyness_squared"] = data["m"] ** 2
    data["sqrt_T"] = np.sqrt(data["T"] + 1e-8)
    data["log_T"] = np.log(data["T"] + 1e-8)
    data["inv_T"] = 1 / (data["T"] + 1e-8)
    data["time_decay"] = np.exp(-data["T"])
    data["log_gamma"] = np.log(data["gamma"] + 1e-8)
    data["log_omega"] = np.log(data["omega"] + 1e-8)
    data["log_alpha"] = np.log(data["alpha"] + 1e-8)
    data["log_beta"] = np.log(data["beta"] + 1e-8)
    data["log_lambda"] = np.log(data["lambda"] + 1e-8)
    data["risk_free_T"] = data["r"] * data["T"]
    data["m_T_interaction"] = data["m"] * data["T"]
    data["value_ratio"] = data["V"] / (data["S0"] + 1e-8)
    data["log_value"] = np.log(data["V"] + 1e-8)
    data["is_call"] = (data["corp"] + 1) / 2
    data["corp_m_interaction"] = data["corp"] * data["m"]
    data["corp_T_interaction"] = data["corp"] * data["T"]

    # Base features for IV model
    base_features = ["S0", "m", "r", "T", "corp",
                    "alpha", "beta", "omega", "gamma", "lambda", "V"]

    # Engineered features for IV model
    engineered_features = [
        "strike", "returns", "log_moneyness", "moneyness_squared",
        "sqrt_T", "log_T", "inv_T", "time_decay",
        "log_gamma", "log_omega", "log_lambda", "log_alpha", "log_beta",
        "m_T_interaction", "value_ratio", "log_value",
        "is_call", "corp_m_interaction", "corp_T_interaction"
    ]

    # Extract features
    features = []
    for _, row in data.iterrows():
        base_feat = torch.tensor(row[base_features].values, dtype=torch.float32)
        eng_feat = torch.tensor(row[engineered_features].values, dtype=torch.float32)
        X = torch.cat([base_feat, eng_feat])
        features.append(X)

    X = torch.stack(features)

    return X, data


def evaluate_model(model, input_data, raw_data):
    """
    Run the IV_GLU model on input data and return predictions.

    Args:
        model: PyTorch model
        input_data (torch.Tensor): Input features
        raw_data (pd.DataFrame): Original data for reference

    Returns:
        pd.DataFrame: Results with predictions
    """
    with torch.no_grad():
        predictions = model(input_data)

    # Convert predictions to numpy array
    predictions_np = predictions.cpu().numpy()

    # Create DataFrame with predictions
    result_df = raw_data.copy()
    result_df["predicted_sigma"] = predictions_np

    # Calculate errors if we have ground truth
    if "sigma" in raw_data.columns:
        result_df["error"] = result_df["sigma"] - result_df["predicted_sigma"]
        result_df["percent_error"] = (result_df["error"] / result_df["sigma"]) * 100

        # Summary statistics
        mae = result_df["error"].abs().mean()
        mape = result_df["percent_error"].abs().mean()
        print(f"Mean Absolute Error: {mae:.6f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved IV_GLU model on input data")
    parser.add_argument("--model", type=str, default="saved_models/scalable_hn_dataset_250x60_20250820/model.pt",
                        help="Path to the saved model file (model.pt)")
    parser.add_argument("--data", type=str, default="random_data.csv",
                        help="Path to the input data CSV file")
    parser.add_argument("--output", type=str, default="model_predictions.csv",
                        help="Path to save the output results")

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Prepare the data
    inputs, raw_data = prepare_data(args.data)

    # Evaluate the model
    results = evaluate_model(model, inputs, raw_data)

    # Show a sample of the results
    print("\nSample of results:")
    print(results[["S0", "m", "T", "sigma", "predicted_sigma", "error", "percent_error"]].head())

    # Save results
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
