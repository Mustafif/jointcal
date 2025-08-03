import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import sklearn.model_selection as sklearn
scaler = MinMaxScaler(feature_range=(0, 1))

class IVDataset(Dataset):
    def __init__(self, dataframe, is_train=False, target_scaler=None):
        self.data = dataframe
        self.is_train = is_train
        self.base_features = ["S0", "m", "r", "T", "corp",
                              "alpha", "beta", "omega", "gamma", "lambda", "V"]
        self.target_scaler = scaler
        self.target_scaler.fit(self.data[["sigma"]])

        # Precompute constant features for faster access later
        self.epsilon = 1e-6  # To avoid division by zero in calculations

        # Basic derived features
        self.data["strike"] = self.data["S0"] * self.data["m"]

        # Advanced moneyness features
        self.data["log_moneyness"] = torch.log(torch.tensor(self.data["m"].values))
        self.data["moneyness_squared"] = torch.tensor(self.data["m"].values) ** 2
        self.data["moneyness_centered"] = torch.tensor(self.data["m"].values) - 1.0
        self.data["atm_indicator"] = torch.exp(-10 * (torch.tensor(self.data["m"].values) - 1.0) ** 2)

        # Time-related features
        self.data["sqrt_T"] = torch.sqrt(torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["log_T"] = torch.log(torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["inv_T"] = 1 / (torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["time_decay"] = torch.exp(-torch.tensor(self.data["T"].values))

        # GARCH-related features
        self.data["log_gamma"] = torch.log(torch.tensor(self.data["gamma"].values) + self.epsilon)
        self.data["sqrt_omega"] = torch.sqrt(torch.tensor(self.data["omega"].values) + self.epsilon)
        self.data["log_omega"] = torch.log(torch.tensor(self.data["omega"].values) + self.epsilon)
        self.data["alpha_beta"] = torch.tensor(self.data["alpha"].values) * torch.tensor(self.data["beta"].values)
        self.data["alpha_gamma"] = torch.tensor(self.data["alpha"].values) * torch.tensor(self.data["gamma"].values)
        self.data["beta_squared"] = torch.tensor(self.data["beta"].values) ** 2

        # Volatility persistence and regime features
        persistence = torch.tensor(self.data["alpha"].values) + torch.tensor(self.data["beta"].values)
        self.data["persistence"] = persistence
        self.data["mean_reversion"] = 1.0 - persistence
        self.data["unconditional_vol"] = torch.sqrt(torch.tensor(self.data["omega"].values) /
                                                   (1.0 - persistence + self.epsilon))

        # Risk and return features
        self.data["risk_free_T"] = torch.tensor(self.data["r"].values) * torch.tensor(self.data["T"].values)
        self.data["lambda_scaled"] = torch.tensor(self.data["lambda"].values) * torch.sqrt(torch.tensor(self.data["T"].values))

        # Interaction features for volatility modeling
        self.data["m_T_interaction"] = torch.tensor(self.data["m"].values) * torch.tensor(self.data["T"].values)
        self.data["vol_skew_proxy"] = torch.tensor(self.data["gamma"].values) * torch.tensor(self.data["lambda"].values)

        # Option value relative features
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
            row["log_moneyness"],
            row["moneyness_squared"],
            row["moneyness_centered"],
            row["atm_indicator"],
            row["sqrt_T"],
            row["log_T"],
            row["inv_T"],
            row["time_decay"],
            row["log_gamma"],
            row["sqrt_omega"],
            row["log_omega"],
            row["alpha_beta"],
            row["alpha_gamma"],
            row["beta_squared"],
            row["persistence"],
            row["mean_reversion"],
            row["unconditional_vol"],
            row["risk_free_T"],
            row["lambda_scaled"],
            row["m_T_interaction"],
            row["vol_skew_proxy"],
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
        target_df = pd.DataFrame([[target_value]], columns=["sigma"])
        scaled_target = self.target_scaler.transform(target_df).flatten()
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y


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
