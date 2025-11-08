import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from hn import HestonNandiGARCH
scaler = MinMaxScaler(feature_range=(0, 1))

# class CalibrationDataset(Dataset):
#     def __init__(self, dataframe, returns):
#         self.data = dataframe.copy()
#         self.base_features = ["S0", "m", "r", "T", "corp", "V"]
#         self.sigma = self.data["sigma"].values
#         self.returns = returns
#         hn = HestonNandiGARCH(self.returns)
#         hn.fit()
#         self.garch_params = ["omega", "alpha", "beta", "gamma", "lambda"]
#         self.target = torch.tensor(np.array([hn.omega, hn.alpha, hn.beta, hn.gamma, hn.lambda_]))
#         self.target_scaler = MinMaxScaler(feature_range=(0, 1))
#         self.N = len(self.returns)
#         self.M = len(self.data["S0"])
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         features = row[self.base_features].values
#         target = self.target
#         return_ = self.returns[idx]
#         return torch.tensor(features), torch.tensor(target), torch.tensor(return_), torch.tensor(self.sigma), self.N, self.M

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# class CalibrationDataset(Dataset):
#     def __init__(self, dataframe, returns, heston_class=HestonNandiGARCH):
#         """
#         dataframe: pd.DataFrame with option data (must include columns in base_features + 'sigma')
#         returns: np.ndarray or torch.Tensor of asset returns
#         heston_class: class implementing Heston-Nandi GARCH model with .fit()
#         """
#         super().__init__()
#         self.data = dataframe.copy()

#         # Base feature columns for model input
#         self.base_features = ["S0", "m", "r", "T", "corp", "V"]

#         self.data["strike"] = self.data["S0"] * self.data["m"]
#         self.data["returns"] = 0.0
#         self.data["log_moneyness"] = np.log(self.data["m"] + 1e-8)
#         self.data["moneyness_squared"] = self.data["m"] ** 2
#         self.data["sqrt_T"] = np.sqrt(self.data["T"] + 1e-8)
#         self.data["log_T"] = np.log(self.data["T"] + 1e-8)
#         self.data["inv_T"] = 1 / (self.data["T"] + 1e-8)
#         self.data["time_decay"] = np.exp(-self.data["T"])
#         self.data["log_gamma"] = np.log(self.data["gamma"] + 1e-8)
#         self.data["log_omega"] = np.log(self.data["omega"] + 1e-8)
#         self.data["log_alpha"] = np.log(self.data["alpha"] + 1e-8)
#         self.data["log_beta"] = np.log(self.data["beta"] + 1e-8)
#         self.data["log_lambda"] = np.log(self.data["lambda"] + 1e-8)
#         self.data["risk_free_T"] = self.data["r"] * self.data["T"]
#         self.data["m_T_interaction"] = self.data["m"] * self.data["T"]
#         self.data["value_ratio"] = self.data["V"] / (self.data["S0"] + 1e-8)
#         self.data["log_value"] = np.log(self.data["V"] + 1e-8)
#         self.data["is_call"] = (self.data["corp"] + 1) / 2
#         self.data["corp_m_interaction"] = self.data["corp"] * self.data["m"]
#         self.data["corp_T_interaction"] = self.data["corp"] * self.data["T"]

#         # ====== Features for model ======
#         self.base_features = ["S0", "m", "r", "T", "corp",
#                                 "alpha", "beta", "omega", "gamma", "lambda", "V"]

#         self.engineered_features = [
#             "strike", "returns", "log_moneyness", "moneyness_squared",
#             "sqrt_T", "log_T", "inv_T", "time_decay",
#             "log_gamma", "log_omega", "log_lambda", "log_alpha", "log_beta",
#             "m_T_interaction", "value_ratio", "log_value",
#             "is_call", "corp_m_interaction", "corp_T_interaction"
#         ]

#         # Extract sigma (observed implied volatility)
#         self.sigma = torch.tensor(self.data["sigma"].values, dtype=torch.float32)

#         # Store returns as torch tensor
#         self.returns = torch.tensor(returns, dtype=torch.float32)

#         # Fit Heston–Nandi GARCH model once (not per item)
#         hn = heston_class(self.returns.numpy())
#         hn.fit()

#         # GARCH parameter names and values
#         self.garch_params = ["omega", "alpha", "beta", "gamma", "lambda"]
#         param_values = np.array([hn.omega, hn.alpha, hn.beta, hn.gamma, hn.lambda_], dtype=np.float32)

#         # Scale parameters between 0 and 1
#         self.target_scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_target = self.target_scaler.fit_transform(param_values.reshape(-1, 1)).flatten()
#         self.target = torch.tensor(scaled_target, dtype=torch.float32)

#         # Store sizes
#         self.returns_len = len(self.returns)
#         self.sigma_len = len(self.sigma)

#     def __len__(self):
#         return self.sigma_len

#     def __getitem__(self, idx):
#         """
#         Returns:
#           features: torch.FloatTensor [len(base_features)]
#           target: torch.FloatTensor [5] (GARCH parameters)
#           return_: torch.FloatTensor scalar
#           sigma: torch.FloatTensor scalar
#           N, M: ints
#         """
#         row = self.data.iloc[idx]
#         features = torch.tensor(row[self.base_features].values, dtype=torch.float32)
#         sigma_i = self.sigma[idx]
#         return features, self.target, returns, sigma_i, self.returns_len, self.sigma_len
class CalibrationDataset(Dataset):
    def __init__(self, dataframe, returns, heston_class=HestonNandiGARCH):
        super().__init__()
        self.data = dataframe.copy()

        # Base feature columns for model input
        self.base_features = ["S0", "m", "r", "T", "corp", "V"]

        # Engineered features
        self.data["strike"] = self.data["S0"] * self.data["m"]
        self.data["returns"] = 0.0
        self.data["log_moneyness"] = np.log(self.data["m"] + 1e-8)
        self.data["moneyness_squared"] = self.data["m"] ** 2
        self.data["sqrt_T"] = np.sqrt(self.data["T"] + 1e-8)
        self.data["log_T"] = np.log(self.data["T"] + 1e-8)
        self.data["inv_T"] = 1 / (self.data["T"] + 1e-8)
        self.data["time_decay"] = np.exp(-self.data["T"])
        self.data["log_gamma"] = np.log(self.data["gamma"] + 1e-8)
        self.data["log_omega"] = np.log(self.data["omega"] + 1e-8)
        self.data["log_alpha"] = np.log(self.data["alpha"] + 1e-8)
        self.data["log_beta"] = np.log(self.data["beta"] + 1e-8)
        self.data["log_lambda"] = np.log(self.data["lambda"] + 1e-8)
        self.data["risk_free_T"] = self.data["r"] * self.data["T"]
        self.data["m_T_interaction"] = self.data["m"] * self.data["T"]
        self.data["value_ratio"] = self.data["V"] / (self.data["S0"] + 1e-8)
        self.data["log_value"] = np.log(self.data["V"] + 1e-8)
        self.data["is_call"] = (self.data["corp"] + 1) / 2
        self.data["corp_m_interaction"] = self.data["corp"] * self.data["m"]
        self.data["corp_T_interaction"] = self.data["corp"] * self.data["T"]

        # ====== Features for model ======
        self.base_features = ["S0", "m", "r", "T", "corp",
                              "alpha", "beta", "omega", "gamma", "lambda", "V"]

        self.engineered_features = [
            "strike", "returns", "log_moneyness", "moneyness_squared",
            "sqrt_T", "log_T", "inv_T", "time_decay",
            "log_gamma", "log_omega", "log_lambda", "log_alpha", "log_beta",
            "m_T_interaction", "value_ratio", "log_value",
            "is_call", "corp_m_interaction", "corp_T_interaction"
        ]

        # Observed implied volatility
        self.sigma = torch.tensor(self.data["sigma"].values, dtype=torch.float32)

        # Returns tensor
        self.returns = torch.tensor(returns, dtype=torch.float32)

        # Fit Heston–Nandi GARCH model once
        hn = heston_class(self.returns.numpy())
        hn.fit()

        # Store raw GARCH parameters as target
        param_values = torch.tensor([hn.omega, hn.alpha, hn.beta, hn.gamma, hn.lambda_],
                                    dtype=torch.float32)
        self.target = param_values

        # Precompute full feature tensor for vectorized inference
        self.X = torch.tensor(self.data[self.base_features + self.engineered_features].values,
                              dtype=torch.float32)

        # Sizes
        self.returns_len = len(self.returns)
        self.sigma_len = len(self.sigma)

    def __len__(self):
        return self.sigma_len

    def __getitem__(self, idx):
        row_features = self.X[idx]  # includes both base + engineered
        sigma_i = self.sigma[idx]
        return_ = self.returns
        return row_features, self.target, return_, sigma_i, self.returns_len, self.sigma_len


def dataset_file(filename):
    return pd.read_csv(filename)

def cleandataset(data):
    # Remove rows where implied volatility is close to zero
    initial_rows = len(data)
    data = data[data["sigma"] > 1e-4]
    final_rows = len(data)
    print(f"Cleaned dataset: Removed {initial_rows - final_rows} rows with sigma <= 1e-4.")
    return data

def cal_dataset(filename, asset_filename):
    from returns import daily_log_returns, read_data
    data = dataset_file(filename)
    data = cleandataset(data)
    asset = read_data(asset_filename)
    returns = daily_log_returns(asset)
    dataset = CalibrationDataset(data, returns)
    return dataset

dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv", "joint_dataset/assetprices.csv")
true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])

two_norm = np.linalg.norm(np.asarray(dataset.target.numpy()) - np.asarray(true_vals), ord=2)
norm = np.linalg.norm(np.asarray(true_vals), ord=2)

inf_norm = np.linalg.norm(np.asarray(dataset.target.numpy()) - np.asarray(true_vals), ord=np.inf)
print(f"Two-norm distance between predicted and true values: {two_norm}")
print(f"Percentage Two-norm distance between predicted and true values: {two_norm/norm * 100}%")
print(f"Infinity-norm distance between predicted and true values: {inf_norm}")
