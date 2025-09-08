import torch
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import pickle

from model import IV_GLU
from dataset import CalibrationDataset
from sklearn.preprocessing import MinMaxScaler


def load_model(model_path):
    """
    Load a saved PyTorch IV_GLU model.
    """
    model = IV_GLU(input_features=30, hidden_size=350)
    try:
        # First try loading as a regular state dict
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except (RuntimeError, pickle.UnpicklingError) as e:
        if "weights_only" in str(e):
            # Try loading as a TorchScript model with weights_only=False
            print("Detected TorchScript archive, trying to load with weights_only=False")
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            # If it's another error, re-raise it
            raise
    model.eval()  # Set model to evaluation mode
    print(f"Loaded IV_GLU model from {model_path}")
    return model


def prepare_data(data):
    """
    Prepare data for IV_GLU model evaluation.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
        print(f"Loaded data from {data} with {len(data)} rows")

    data["strike"] = data["S0"] * data["m"]
    data["returns"] = 0.0
    data["log_moneyness"] = np.log(data["m"])
    data["moneyness_squared"] = data["m"] ** 2
    data["sqrt_T"] = np.sqrt(data["T"] + 1e-8)
    data["log_T"] = np.log(data["T"] + 1e-8)
    data["inv_T"] = 1 / (data["T"] + 1e-8)
    data["time_decay"] = np.exp(-data["T"])
    data["log_gamma"] = np.log(data["gamma"] + 1e-8)
    # Clip alpha and omega before taking the log
    data["log_omega"] = np.log(np.maximum(data["omega"], 1e-5))
    data["log_alpha"] = np.log(np.maximum(data["alpha"], 1e-5))
    data["log_beta"] = np.log(data["beta"] + 1e-8)
    data["log_lambda"] = np.log(data["lambda"] + 1e-8)
    data["risk_free_T"] = data["r"] * data["T"]
    data["m_T_interaction"] = data["m"] * data["T"]
    data["value_ratio"] = data["V"] / (data["S0"] + 1e-8)
    data["log_value"] = np.log(data["V"] + 1e-8)
    data["is_call"] = (data["corp"] + 1) / 2
    data["corp_m_interaction"] = data["corp"] * data["m"]
    data["corp_T_interaction"] = data["corp"] * data["T"]

    base_features = ["S0", "m", "r", "T", "corp",
                      "alpha", "beta", "omega", "gamma", "lambda", "V"]

    engineered_features = [
        "strike", "returns", "log_moneyness", "moneyness_squared",
        "sqrt_T", "log_T", "inv_T", "time_decay",
        "log_gamma", "log_omega", "log_lambda", "log_alpha", "log_beta",
        "m_T_interaction", "value_ratio", "log_value",
        "is_call", "corp_m_interaction", "corp_T_interaction"
    ]

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
    """
    scaler = MinMaxScaler()
    scaler.fit(data[["sigma"]].values)
    outputs_np = outputs.detach().numpy()
    outputs_original = scaler.inverse_transform(outputs_np)
    result_df = pd.DataFrame(outputs_original, columns=["predicted_sigma"])
    return result_df


def evaluate_model(model, input_data, raw_data):
    """
    Run the IV_GLU model on input data and return predictions.
    """
    with torch.no_grad():
        predictions = model(input_data)

    prediction_df = inverse_transform_outputs(predictions, raw_data)
    result_df = pd.concat([raw_data.reset_index(drop=True), prediction_df], axis=1)

    if "sigma" in raw_data.columns:
        result_df["error"] = result_df["sigma"] - result_df["predicted_sigma"]
        result_df["percent_error"] = (result_df["error"] / result_df["sigma"]) * 100

    return result_df


if __name__ == "__main__":
    model_path = "saved_models/duan_garch_dataset_250x60_12params_20250827/model.pt"
    data_folder = "random_data_HN"

    model = load_model(model_path)

    all_results = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            data_path = os.path.join(data_folder, filename)
            inputs, raw_data = prepare_data(data_path)
            results = evaluate_model(model, inputs, raw_data)
            all_results.append(results)

    final_results = pd.concat(all_results, ignore_index=True)

    mae = final_results["error"].abs().mean()
    mape = final_results["percent_error"].abs().mean()

    print("\n" + "="*20)
    print("Overall Performance")
    print("="*20)
    print(f"Overall Mean Absolute Error: {mae:.6f}")
    print(f"Overall Mean Absolute Percentage Error: {mape:.2f}%")
