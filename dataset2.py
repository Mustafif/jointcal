import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from hn import HestonNandiGARCH
scaler = MinMaxScaler(feature_range=(0, 1))

class CalibrationDataset(Dataset):
    def __init__(self, dataframe, returns):
        self.data = dataframe.copy()
        self.base_features = ["S0", "m", "r", "T", "corp", "V"]
        self.returns = returns
        hn = HestonNandiGARCH(self.returns)
        hn.fit()
        self.garch_params = ["omega", "alpha", "beta", "gamma", "lambda"]
        self.target = torch.tensor(np.array([hn.omega, hn.alpha, hn.beta, hn.gamma, hn.lambda_]))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.N = len(self.returns)
        self.M = len(self.data["S0"])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row[self.base_features].values
        target = self.target
        return_ = self.returns[idx]
        return torch.tensor(features), torch.tensor(target), torch.tensor(return_), self.N, self.M

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
