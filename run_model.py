#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime



def run_inference_on_file(model, device, data_path):
    """
    Runs model inference on a single data file.

    Args:
        model: The loaded model.
        device: The torch device.
        data_path (str): Path to the input data CSV file.

    Returns:
        pandas.DataFrame: DataFrame with predictions, or None on error.
    """
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        return None
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} samples from {data_path}")
    except Exception as e:
        print(f"Error loading data from {data_path}: {str(e)}")
        return None

    # Keep track of original file
    data['source_file'] = os.path.basename(data_path)

    # Create derived features
    data["strike"] = data["S0"] * data["m"]
    data["returns"] = 0.0
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

    # Define features
    base_features = ["S0", "m", "r", "T", "corp",
                    "alpha", "beta", "omega", "gamma", "lambda", "V"]

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
        base_feat = torch.tensor(row[base_features].tolist(), dtype=torch.float32)
        eng_feat = torch.tensor(row[engineered_features].tolist(), dtype=torch.float32)
        X = torch.cat([base_feat, eng_feat])
        features.append(X)

    if not features:
        print(f"No data to process in {data_path}")
        return None

    inputs = torch.stack(features).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model(inputs)

    # Add predictions to results
    results = data.copy()
    results["predicted_sigma"] = predictions.cpu().numpy()

    # Calculate errors if ground truth is available
    if "sigma" in results.columns:
        results["error"] = results["sigma"] - results["predicted_sigma"]
        results["percent_error"] = (results["error"] / results["sigma"]) * 100

    return results


def main(model_path, data_folder, output_folder):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()  # Set to evaluation mode

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    all_results = []

    if not os.path.isdir(data_folder):
        print(f"Error: Data folder {data_folder} not found or is not a directory!")
        return

    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            data_path = os.path.join(data_folder, filename)
            results_df = run_inference_on_file(model, device, data_path)
            if results_df is not None:
                all_results.append(results_df)

    if not all_results:
        print("No results to save.")
        return

    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)

    # Generate timestamp and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{timestamp}.csv"
    output_path = os.path.join(output_folder, output_filename)

    final_results.to_csv(output_path, index=False)
    print(f"\nAll results saved to {output_path}")

    # Print summary
    if "error" in final_results.columns:
        mae = final_results["error"].abs().mean()
        mape = final_results["percent_error"].abs().mean()
        print(f"\nOverall Mean Absolute Error: {mae:.6f}")
        print(f"Overall Mean Absolute Percentage Error: {mape:.2f}%")


if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
    DATA_FOLDER = "random_data_HN"
    OUTPUT_FOLDER = "inference_performance"
    # -------------------
    main(MODEL_PATH, DATA_FOLDER, OUTPUT_FOLDER)
