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

    # Inverse transform the scaled targets
    targets = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

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
    h = torch.zeros(N, device='cuda', dtype=torch.float32)
    h[0] = (omega + alpha) / (1 - beta - alpha*gamma**2)
    for t in range(N):
        z = torch.randn(1, device='cuda', dtype=torch.float32)
        h[t+1] = omega + beta*h[t] + alpha*(z - gamma*torch.sqrt(h[t]))**2
    return h

def returns_loss(N, omega, alpha, beta, gamma, lambda_, returns, r):
    h = HN_cond_var(N, omega, alpha, beta, gamma)
    sum = torch.sum(torch.log(h) + ((returns - r - lambda_*h + gamma)**2)/h)
    return -1/2 * sum

def implied_loss(implied_market, implied_model):
    eps = 0.01
    return -1/2*torch.sum(2*np.log(eps) + (implied_market - implied_model)**2 / eps**2)


def joint_loss(implied_market, implied_model,M,N, omega, alpha, beta, gamma, lambda_, returns, r):
    rl = returns_loss(N, omega, alpha, beta, gamma, lambda_, returns, r)
    il = implied_loss(implied_market, implied_model)
    return -(((N+M)/2*N)*rl + ((N+M)/2*M)*il)
