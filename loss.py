# Here is where we will define our loss function for joint calibration
import torch
import numpy as np
import pandas as pd


def calculate_loss(data_file, target_scaler):
    """
    Calculate various performance metrics between predictions and targets.

    Parameters:
    data_file (str): Path to the CSV file containing predictions and targets.

    Returns:
    dict: A dictionary containing detailed performance metrics.
    """
    # Load the CSV file
    data = pd.read_csv(data_file)

    # Extract predictions and targets as numpy arrays
    predictions = data["predictions"].to_numpy()
    targets = data["targets"].to_numpy()

    # Inverse transform the scaled targets using DataFrames to avoid warnings
    targets_df = pd.DataFrame(targets.reshape(-1, 1), columns=["sigma"])
    targets = target_scaler.inverse_transform(targets_df).flatten()
    predictions_df = pd.DataFrame(predictions.reshape(-1, 1), columns=["sigma"])
    predictions = target_scaler.inverse_transform(predictions_df).flatten()

    # Calculate various metrics
    absolute_errors = np.abs(predictions - targets)  # Absolute errors
    squared_errors = (predictions - targets) ** 2  # Squared errors

    # Mean Squared Error (MSE)
    mse = np.mean(squared_errors)

    # Mean Absolute Error (MAE)
    mae = np.mean(absolute_errors)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # R^2 score
    r2 = 1 - (np.sum(squared_errors) / np.sum((targets - np.mean(targets)) ** 2))

    # Mean Relative Error (MRE)
    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = 1e-8
    relative_errors = absolute_errors / (np.abs(targets) + epsilon)
    mre = np.mean(relative_errors)

    # Mean of predictions and targets
    mean_predicted = np.mean(predictions)
    mean_target = np.mean(targets)

    # Prepare detailed loss information
    loss_info = {
        "total_samples": len(predictions),
        "Mean of Predicted": mean_predicted,
        "Mean of Targets": mean_target,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2,
        "MRE": mre,  # Mean Relative Error
        "min_error": np.min(absolute_errors),
        "max_error": np.max(absolute_errors),
        "std_error": np.std(absolute_errors),
    }

    return loss_info


def HN_cond_var(N, omega, alpha, beta, gamma):
    device = omega.device if torch.is_tensor(omega) else 'cpu'

    # Validate GARCH parameters
    persistence = beta + alpha * gamma**2
    if persistence >= 1.0:
        # If non-stationary, use a small adjustment to ensure stability
        persistence = 0.99
        beta = persistence - alpha * gamma**2

    # Calculate initial variance
    h_0 = (omega + alpha) / (1 - persistence)
    # Ensure h_0 is positive
    h_0 = torch.abs(h_0) + 1e-8

    # Initialize variance list
    h_list = []
    # Ensure h_0 is a 1-element tensor, not a scalar
    if h_0.dim() == 0:
        h_0 = h_0.unsqueeze(0)
    h_list.append(h_0)

    h_t = h_0
    for t in range(N-1):
        z = torch.randn(1, device=device, dtype=torch.float32)
        # Calculate next variance ensuring it stays positive
        h_next = omega + beta*h_t + alpha*(z - gamma*torch.sqrt(torch.abs(h_t) + 1e-8))**2
        h_next = torch.abs(h_next) + 1e-8  # Ensure positivity
        # Ensure h_next is a 1-element tensor
        if h_next.dim() == 0:
            h_next = h_next.unsqueeze(0)
        h_list.append(h_next)
        h_t = h_next

    # Stack and squeeze to get 1D tensor
    return torch.stack(h_list).squeeze()

def returns_loss(N, omega, alpha, beta, gamma, lambda_, returns, r):
    # Ensure parameters are positive where needed
    omega = torch.abs(omega) + 1e-8
    alpha = torch.abs(alpha) + 1e-8
    beta = torch.abs(beta)

    # Calculate conditional variances
    h = HN_cond_var(N, omega, alpha, beta, gamma)

    # Ensure returns is a tensor with shape matching h
    if returns.dim() == 0:  # scalar
        returns = returns.expand(N)
    elif returns.shape[0] == 1:  # single value
        returns = returns.expand(N)
    elif returns.shape[0] != N:
        # If we have different length, use the mean return value
        returns = torch.full((N,), returns.mean(), device=returns.device)

    # Calculate log-likelihood
    eps = 1e-8  # Small value to prevent log(0)
    h_safe = h + eps
    # Ensure h is 1D
    if h_safe.dim() > 1:
        h_safe = h_safe.squeeze()

    # Calculate risk premium term
    risk_premium = lambda_ * torch.sqrt(h_safe)

    # Calculate standardized residuals
    residuals = returns - r - risk_premium + gamma * torch.sqrt(h_safe)

    # Log-likelihood calculation
    sum_ll = torch.sum(torch.log(h_safe) + (residuals**2) / h_safe)
    return -0.5 * sum_ll

def implied_loss(implied_market, implied_model):
    eps = 0.01
    # Ensure both are tensors on same device
    if not torch.is_tensor(implied_market):
        implied_market = torch.tensor(implied_market, dtype=torch.float32)
    if not torch.is_tensor(implied_model):
        implied_model = torch.tensor(implied_model, dtype=torch.float32)

    # Calculate squared error with epsilon for numerical stability
    squared_error = (implied_market - implied_model)**2
    return -0.5 * torch.sum(2 * torch.log(torch.tensor(eps)) + squared_error / (eps**2))


def joint_loss(implied_market, implied_model, M, N, omega, alpha, beta, gamma, lambda_, returns, r):
    # Ensure all parameters are tensors on the same device
    device = omega.device if torch.is_tensor(omega) else 'cpu'

    # Convert scalars to tensors if needed
    if not torch.is_tensor(omega):
        omega = torch.tensor(omega, device=device, dtype=torch.float32)
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, device=device, dtype=torch.float32)
    if not torch.is_tensor(lambda_):
        lambda_ = torch.tensor(lambda_, device=device, dtype=torch.float32)
    if not torch.is_tensor(returns):
        returns = torch.tensor(returns, device=device, dtype=torch.float32)
    if not torch.is_tensor(r):
        r = torch.tensor(r, device=device, dtype=torch.float32)

    rl = returns_loss(N, omega, alpha, beta, gamma, lambda_, returns, r)
    il = implied_loss(implied_market, implied_model)

    # Ensure N and M are floats for division
    N_float = float(N)
    M_float = float(M)

    # Calculate weighted joint loss
    total_weight = N_float + M_float
    return -((total_weight / (2 * N_float)) * rl + (total_weight / (2 * M_float)) * il)
