import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import sklearn.model_selection as sklearn
import numpy as np
scaler = MinMaxScaler(feature_range=(0, 1))

class CalibrationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.copy()  # Make a copy to avoid modifying the original
        self.base_features = ["S0", "m", "r", "T", "corp","sigma"]
        self.garch_params = ["alpha", "beta", "omega", "gamma", "lambda"]
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler.fit(self.data[self.garch_params])
        # Calculate returns from unique S0 values BEFORE creating IVDataset
        if "returns" not in self.data.columns:
            self._calculate_returns_from_s0()
        # Now create IVDataset after returns are calculated
        self.ivds = IVDataset(self.data.copy())  # Pass a copy to avoid overwriting returns
        self.data["iv_model"] = [0 for _ in range(len(dataframe))]
        # Use actual number of returns if calculated, otherwise default
        if hasattr(self, 'returns_series'):
            self.N = len(self.returns_series)
        else:
            self.N = 252  # Default: Number of daily returns in a year
        self.M = len(self.data["sigma"])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_features = torch.tensor(row[self.base_features].values, dtype=torch.float32)
        # Ensure iv_model is a 1D tensor
        iv_model_value = row["iv_model"]
        if isinstance(iv_model_value, (int, float)):
            engineered_features = torch.tensor([iv_model_value], dtype=torch.float32)
        else:
            engineered_features = torch.tensor(iv_model_value, dtype=torch.float32)
        X = torch.cat([base_features, engineered_features])
        # Extract GARCH parameters and reshape for scaler
        garch_params = row[self.garch_params].values
        # Create DataFrame with proper feature names to avoid sklearn warnings
        garch_df = pd.DataFrame([garch_params], columns=self.garch_params)
        scaled = self.target_scaler.transform(garch_df)
        Y = torch.tensor(scaled.flatten(), dtype=torch.float32)
        return X, Y, self.N, self.M

    def _calculate_returns_from_s0(self):
        """Calculate log returns from unique S0 values"""
        # Get unique S0 values in order of appearance
        unique_s0 = []
        seen = set()
        for s0 in self.data['S0'].values:  # Use .values for consistent float handling
            if s0 not in seen:
                unique_s0.append(s0)
                seen.add(s0)

        unique_s0 = np.array(unique_s0)

        # Calculate log returns between consecutive S0 values
        if len(unique_s0) > 1:
            # Calculate log returns: ln(S_t / S_{t-1})
            log_returns = np.log(unique_s0[1:] / unique_s0[:-1])

            # Create a mapping from S0 to return using rounded values to handle float precision
            s0_to_return = {}
            # Round to 10 decimal places to handle floating point comparison issues
            s0_to_return[round(unique_s0[0], 10)] = 0.0  # First period has no return

            for i in range(1, len(unique_s0)):
                s0_to_return[round(unique_s0[i], 10)] = log_returns[i-1]

            # Map returns to each row based on its S0 value (rounded)
            self.data['returns'] = self.data['S0'].apply(lambda x: s0_to_return.get(round(x, 10), 0.0))

            # Store the actual returns series
            self.returns_series = log_returns
            self.N = len(log_returns)  # Update N to actual number of returns

            print(f"Calculated {len(log_returns)} returns from {len(unique_s0)} unique S0 values")
            print(f"Returns range: [{log_returns.min():.6f}, {log_returns.max():.6f}]")
            print(f"Non-zero returns assigned: {(self.data['returns'] != 0).sum()}")
        else:
            # If only one unique S0, we can't calculate returns
            print("Warning: Only one unique S0 value found. Setting returns to 0.")
            self.data['returns'] = 0.0
            self.returns_series = np.array([])



class IVDataset(Dataset):
    def __init__(self, dataframe, is_train=False, target_scaler=None):
        self.data = dataframe
        self.is_train = is_train
        self.base_features = ["S0", "m", "r", "T", "corp",
                              "alpha", "beta", "omega", "gamma", "lambda", "V"]
        if target_scaler is None:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler.fit(self.data[["sigma"]])
        else:
            self.target_scaler = target_scaler

        # Precompute constant features for faster access later
        self.epsilon = 1e-8  # To avoid division by zero in calculations

        # Basic derived features
        self.data["strike"] = self.data["S0"] * self.data["m"]
        # IV dataset doesn't need returns data
        if "returns" not in self.data.columns:
            self.data["returns"] = 0.0  # Placeholder, not used in IV prediction

        # Advanced moneyness features
        self.data["log_moneyness"] = torch.log(torch.tensor(self.data["m"].values))
        self.data["moneyness_squared"] = torch.tensor(self.data["m"].values) ** 2
        # self.data["moneyness_centered"] = torch.tensor(self.data["m"].values) - 1.0
        # self.data["atm_indicator"] = torch.exp(-10 * (torch.tensor(self.data["m"].values) - 1.0) ** 2)

        # Time-related features
        self.data["sqrt_T"] = torch.sqrt(torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["log_T"] = torch.log(torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["inv_T"] = 1 / (torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["time_decay"] = torch.exp(-torch.tensor(self.data["T"].values))

        # GARCH-related features
        self.data["log_gamma"] = torch.log(torch.tensor(self.data["gamma"].values) + self.epsilon)
        self.data["log_omega"] = torch.log(torch.tensor(self.data["omega"].values) + self.epsilon)
        self.data["log_alpha"] = torch.log(torch.tensor(self.data["alpha"].values) + self.epsilon)
        self.data["log_beta"] = torch.log(torch.tensor(self.data["beta"].values) + self.epsilon)
        self.data["log_lambda"] = torch.log(torch.tensor(self.data["lambda"].values) + self.epsilon)
        # self.data["alpha_beta"] = torch.tensor(self.data["alpha"].values) * torch.tensor(self.data["beta"].values)
        # self.data["alpha_gamma"] = torch.tensor(self.data["alpha"].values) * torch.tensor(self.data["gamma"].values)
        # self.data["beta_squared"] = torch.tensor(self.data["beta"].values) ** 2

        # Volatility persistence and regime features
        # persistence = torch.tensor(self.data["alpha"].values) + torch.tensor(self.data["beta"].values)
        # self.data["persistence"] = persistence
        # self.data["mean_reversion"] = 1.0 - persistence
        # self.data["unconditional_vol"] = torch.sqrt(torch.tensor(self.data["omega"].values) /
        #                                            (1.0 - persistence + self.epsilon))

        # # Risk and return features
        self.data["risk_free_T"] = torch.tensor(self.data["r"].values) * torch.tensor(self.data["T"].values)
        # self.data["lambda_scaled"] = torch.tensor(self.data["lambda"].values) * torch.sqrt(torch.tensor(self.data["T"].values))

        # # Interaction features for volatility modeling
        self.data["m_T_interaction"] = torch.tensor(self.data["m"].values) * torch.tensor(self.data["T"].values)
        # self.data["vol_skew_proxy"] = torch.tensor(self.data["gamma"].values) * torch.tensor(self.data["lambda"].values)

        # # Option value relative features
        self.data["value_ratio"] = torch.tensor(self.data["V"].values) / (torch.tensor(self.data["S0"].values) + self.epsilon)
        self.data["log_value"] = torch.log(torch.tensor(self.data["V"].values) + self.epsilon)

        # Put-call indicator and its interactions
        corp_tensor = torch.tensor(self.data["corp"].values)
        self.data["is_call"] = (corp_tensor + 1) / 2  # Convert -1,1 to 0,1
        self.data["corp_m_interaction"] = corp_tensor * torch.tensor(self.data["m"].values)
        self.data["corp_T_interaction"] = corp_tensor * torch.tensor(self.data["T"].values)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Directly access precomputed features for this sample
        row = self.data.iloc[idx]
        base_features = torch.tensor(
            row[self.base_features].values, dtype=torch.float32)

        # Extract precomputed engineered features
        engineered_features = torch.tensor([
            row["strike"],
            row["returns"],
            row["log_moneyness"],
            row["moneyness_squared"],
            # row["moneyness_centered"],
            # row["atm_indicator"],
            row["sqrt_T"],
            row["log_T"],
            row["inv_T"],
            row["time_decay"],
             row["log_gamma"],
             row["log_omega"],
             row["log_lambda"],
             row["log_alpha"],
             row["log_beta"],
            # row["alpha_beta"],
            # row["alpha_gamma"],
            # row["beta_squared"],
            # row["persistence"],
            # row["mean_reversion"],
            # row["unconditional_vol"],
            # row["risk_free_T"],
            # row["lambda_scaled"],
            row["m_T_interaction"],
            # row["vol_skew_proxy"],
            row["value_ratio"],
            row["log_value"],
            row["is_call"],
            row["corp_m_interaction"],
            row["corp_T_interaction"]
        ], dtype=torch.float32)

        # Concatenate base features with engineered features
        X = torch.cat([base_features, engineered_features])
        # Scale target variable
        target_value = row["sigma"]
        # Create DataFrame with proper feature name to avoid sklearn warnings
        target_df = pd.DataFrame([[target_value]], columns=["sigma"])
        scaled_target = self.target_scaler.transform(target_df).flatten()
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y

def train_test_split2(data, test_size=0.3, random_state=42):
    # Split the data using sklearn's train_test_split
    train_data, val_data = sklearn.train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_cal = CalibrationDataset(train_data)
    train_iv = train_cal.ivds
    test_cal = CalibrationDataset(val_data)
    test_iv = test_cal.ivds

    return train_cal, train_iv, test_cal, test_iv



def train_test_split(data, test_size=0.3, random_state=42):
    """
    Split the dataset into training and validation sets.

    Args:
        data (pd.DataFrame): Input DataFrame containing the dataset
        test_size (float): Proportion of the dataset to include in the validation split (0 to 1)
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (train_dataset, val_dataset) containing OptionDataset objects
    """
    # Split the data using sklearn's train_test_split
    train_data, val_data = sklearn.train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Create OptionDataset objects
    train_dataset = IVDataset(train_data, is_train=True)
    val_dataset = IVDataset(val_data, is_train=False,
                                target_scaler=train_dataset.target_scaler)

    return train_dataset, val_dataset


def dataset_file(filename):
    return pd.read_csv(filename)


def cleandataset(data):
    # Remove rows where implied volatility is close to zero
    initial_rows = len(data)
    data = data[data["sigma"] > 1e-4]
    final_rows = len(data)
    print(f"Cleaned dataset: Removed {initial_rows - final_rows} rows with sigma <= 1e-4.")
    return data

class DS:
    def __init__(self, path, path2, name):
        self.path = path
        self.path2 = path2
        self.name = name

    def datasets(self):
        if self.path2 is None:
            # Load and clean the dataset
            df = cleandataset(dataset_file(self.path))

            # Split into train and test - train_test_split already returns OptionDataset objects
            ds_train, ds_test = train_test_split(df)

            return ds_train, ds_test
        else:
            # Load and clean both datasets
            train_df = cleandataset(dataset_file(self.path))
            test_df = cleandataset(dataset_file(self.path2))

            # Use train_test_split to create the training dataset
            # Setting test_size=0 to keep all data in the training set
            ds_train, _ = train_test_split(train_df, test_size=0)

            # Create test dataset using the same scaler from training
            ds_test = IVDataset(
                test_df, is_train=False, target_scaler=ds_train.target_scaler
            )

            return ds_train, ds_test
