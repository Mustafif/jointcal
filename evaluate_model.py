#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import argparse
import os
import glob
from pathlib import Path

from model import IV_GLU
from dataset import CalibrationDataset
from sklearn.preprocessing import MinMaxScaler


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

    # Load the model state
    try:
        # First try loading as a regular state dict
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError as e:
        if "Cannot use ``weights_only=True`` with TorchScript archives" in str(e):
            # Try loading as a TorchScript model with weights_only=False
            print("Detected TorchScript archive, trying to load with weights_only=False")
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            # If it's another error, re-raise it
            raise
    model.eval()  # Set model to evaluation mode

    print(f"Loaded IV_GLU model from {model_path}")
    return model


def generate_test_data(num_samples=10):
    """
    Generate test data samples for model evaluation.

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        pd.DataFrame: DataFrame containing the generated test data
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

    print(f"Generated {num_samples} test samples")

    return df

def prepare_data(data):
    """
    Prepare data for IV_GLU model evaluation.

    Args:
        data (pd.DataFrame or str): DataFrame containing the data or path to the CSV file

    Returns:
        tuple: (X, raw_data) - Prepared input tensor and raw dataframe
    """
    # Load the data if it's a path
    if isinstance(data, str):
        data = pd.read_csv(data)
        print(f"Loaded data from {data} with {len(data)} rows")

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


def inverse_transform_outputs(outputs, data):
    """
    Apply inverse transformation to IV_GLU model outputs to get the original scale.

    Args:
        outputs (torch.Tensor): Model outputs
        data (pd.DataFrame): Original data with parameter columns for scaling

    Returns:
        pd.DataFrame: DataFrame with inverse transformed outputs
    """
    # For IV_GLU model - single output (implied volatility)
    scaler = MinMaxScaler()
    scaler.fit(data[["sigma"]].values)

    # Convert outputs to numpy and reshape
    outputs_np = outputs.detach().numpy()

    # Inverse transform
    outputs_original = scaler.inverse_transform(outputs_np)

    # Create DataFrame
    result_df = pd.DataFrame(outputs_original, columns=["predicted_sigma"])

    return result_df


def evaluate_model(model, input_data, raw_data):
    """
    Run the IV_GLU model on input data and return predictions.

    Args:
        model: PyTorch model
        input_data (torch.Tensor): Input features
        raw_data (pd.DataFrame): Original data for reference and scaling

    Returns:
        pd.DataFrame: Results with predictions
    """
    with torch.no_grad():
        predictions = model(input_data)

    # Apply inverse transformation to get predictions in original scale
    prediction_df = inverse_transform_outputs(predictions, raw_data)

    # Add the raw data for reference
    result_df = pd.concat([raw_data.reset_index(drop=True), prediction_df], axis=1)

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


def list_available_models(models_dir="saved_models"):
    """
    List all available model.pt files in the models directory.

    Args:
        models_dir (str): Directory containing model files

    Returns:
        list: List of model file paths
    """
    model_files = []

    # Check if directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return model_files

    # Find all model.pt files
    for model_file in glob.glob(f"{models_dir}/**/model.pt", recursive=True):
        model_files.append(model_file)

    return model_files

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved IV_GLU model on input data")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the saved model file (model.pt)")
    parser.add_argument("--model_dir", type=str, default="saved_models",
                        help="Directory containing model files")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the input data CSV file")
    parser.add_argument("--generate", type=int, default=0,
                        help="Generate N test samples instead of using a data file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the output results (optional)")
    parser.add_argument("--list_models", action="store_true",
                        help="List all available models and exit")

    args = parser.parse_args()

    # List available models if requested
    if args.list_models:
        models = list_available_models(args.model_dir)
        if models:
            print("Available models:")
            for i, model_path in enumerate(models):
                print(f"  {i+1}. {model_path}")
        else:
            print("No models found.")
        return

    # Find available models if no specific model is provided
    if args.model is None:
        models = list_available_models(args.model_dir)
        if not models:
            print("No models found. Please specify a model file with --model.")
            return

        # Select a model
        print("Available models:")
        for i, model_path in enumerate(models):
            print(f"  {i+1}. {model_path}")

        # Default to the first hn_dataset model if available
        default_model = next((m for m in models if "hn_dataset" in m), models[0])
        print(f"\nUsing default model: {default_model}")
        args.model = default_model

    # Generate data if requested
    if args.generate > 0:
        data = generate_test_data(args.generate)
        if args.output:
            # Save generated data
            data_output = f"{os.path.splitext(args.output)[0]}_input.csv"
            data.to_csv(data_output, index=False)
            print(f"Generated data saved to {data_output}")
    elif args.data is None:
        # Try to find random_data.csv if it exists
        if os.path.exists("random_data.csv"):
            args.data = "random_data.csv"
            print(f"Using default data file: {args.data}")
        else:
            # Generate a single sample as default
            print("No data file specified. Generating a single test sample.")
            data = generate_test_data(1)
    else:
        # Load data from file
        data = args.data

    # Set default output filename if not provided
    if args.output is None and isinstance(data, pd.DataFrame):
        args.output = "evaluation_results.csv"

    # Load the model
    model = load_model(args.model)

    # Prepare the data
    if isinstance(data, pd.DataFrame):
        inputs, raw_data = prepare_data(data)
    else:
        inputs, raw_data = prepare_data(args.data)

    # Evaluate the model
    results = evaluate_model(model, inputs, raw_data)

    # Show a sample of the results
    print("\nSample of results:")
    print(results.head())

    # Save results if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
