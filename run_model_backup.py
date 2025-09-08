#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import os

from model import IV_GLU

# Fixed paths - change these if needed
MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
DATA_PATH = "random_data_HN/random_data2.csv"
OUTPUT_PATH = "predictions.csv"

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
print(f"Loading model from {MODEL_PATH}")
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()  # Set to evaluation mode

# Load and prepare data
print(f"Loading data from {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"Error: Data file {DATA_PATH} not found!")
    exit(1)
try:
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(data)} samples")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

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
    base_feat = torch.tensor(row[base_features].values, dtype=torch.float32)
    eng_feat = torch.tensor(row[engineered_features].values, dtype=torch.float32)
    X = torch.cat([base_feat, eng_feat])
    features.append(X)

inputs = torch.stack(features).to(device)

# Run inference
print("Running model inference")
with torch.no_grad():
    predictions = model(inputs)

# Add predictions to results
results = data.copy()
results["predicted_sigma"] = predictions.cpu().numpy()

# Calculate errors if ground truth is available
if "sigma" in results.columns:
    results["error"] = results["sigma"] - results["predicted_sigma"]
    results["percent_error"] = (results["error"] / results["sigma"]) * 100

    # Print error statistics
    mae = results["error"].abs().mean()
    mape = results["percent_error"].abs().mean()
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Show the results
print("\nResults:")
if len(results) <= 5:
    print(results)
else:
    print(results.head(5))
    print("...")

# Save results
results.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")
