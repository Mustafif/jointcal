# import numpy as np

# def ll_option(sigma, sigma_model, sigma_eps=0.001):
#     return -0.5 * np.sum(2*np.log(sigma_eps) + (sigma - sigma_model)**2/sigma_eps**2)

# def ll_returns(returns, params):
#     T = len(returns)
#     h = np.zeros(T)
#     z = np.random.randn(T)
#     omega, alpha, beta, gamma, lambda_ = params

#     h[0] = (omega + alpha)/(1-beta-alpha*gamma**2)

#     for t in range(1, T):
#         h_prev = max(h[t-1], 1e-9)
#         h[t] = omega + beta * h_prev + alpha * (z[t] - gamma * np.sqrt(h_prev))**2

#     log_likelihoods = np.log(2 * np.pi) + np.log(h) + np.power(z, 2)

#     return -0.5 * np.sum(log_likelihoods)

# def ll_joint(sigma, sigma_model,returns, params, N, M,sigma_eps=0.001):
#     lr = ll_returns(returns, params)
#     lo = ll_option(sigma, sigma_model, sigma_eps)
#     return (N+M)/(2*N) * lr + (N+M)/(2*M) * lo

# def Calibration_Loss(params, returns, sigma, model, x, N, M, target):
#     sigma_model = model(x)
#     params = target.as_numpy()
#     returns = returns.as_numpy()
#     sigma = sigma.as_numpy()
#     return -1 * ll_joint(sigma, sigma_model, returns, params, N, M)

# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def ll_option_torch(sigma, sigma_model, sigma_eps=0.001):
#     # Handle scalar inputs
#     if sigma.dim() == 0:
#         sigma = sigma.unsqueeze(0)
#     if sigma_model.dim() == 0:
#         sigma_model = sigma_model.unsqueeze(0)

#     # Ensure tensors are on same device
#     sigma_eps_tensor = torch.tensor(sigma_eps, device=device)

#     return -0.5 * torch.sum(2 * torch.log(sigma_eps_tensor) + (sigma - sigma_model)**2 / sigma_eps**2)

# def ll_returns_torch(returns, params):
#     # Handle single return value or tensor
#     if returns.dim() == 0:
#         returns = returns.unsqueeze(0)

#     T = returns.shape[0]
#     h = torch.zeros(T, device=device)

#     omega, alpha, beta, gamma, lambda_ = params

#     # Initialize h[0] as unconditional variance
#     h[0] = (omega + alpha) / (1 - beta - alpha * gamma**2 + 1e-8)

#     if T == 1:
#         # For single return, compute implied standardized residual
#         z = (returns[0] - lambda_ * h[0]) / torch.sqrt(h[0])
#         log_likelihood = -0.5 * (torch.log(torch.tensor(2 * torch.pi, device=device)) + torch.log(h[0]) + z**2)
#         return log_likelihood

#     # For multiple returns, compute variance recursion using implied residuals
#     for t in range(1, T):
#         h_prev = h[t-1]
#         # Compute implied standardized residual from previous period
#         z_prev = (returns[t-1] - lambda_ * h_prev) / torch.sqrt(h_prev)
#         # Update conditional variance
#         h[t] = omega + beta * h_prev + alpha * (z_prev - gamma * torch.sqrt(h_prev))**2

#     # Compute implied standardized residuals for all periods
#     z = (returns - lambda_ * h) / torch.sqrt(h)

#     # Gaussian log-likelihood (residuals should be N(0,1))
#     log_likelihoods = -0.5 * (torch.log(torch.tensor(2 * torch.pi, device=device)) + torch.log(h) + z**2)
#     return torch.sum(log_likelihoods)

# def ll_joint_torch(sigma, sigma_model, returns, params, N, M, sigma_eps=0.001):
#     lr = ll_returns_torch(returns, params)
#     lo = ll_option_torch(sigma, sigma_model, sigma_eps)
#     return (N + M) / (2 * N) * lr + (N + M) / (2 * M) * lo

# def Calibration_Loss(params, returns, sigma, model, x, N, M):
#     # Get model prediction
#     sigma_model = model(x)

#     # # Ensure sigma_model is properly squeezed
#     # if sigma_model.dim() > 1:
#     #     sigma_model = sigma_model.squeeze()

#     # Ensure all tensors are on the same device
#     device = params.device
#     returns = returns.to(device)
#     sigma = sigma.to(device)

#     loss = -1 * ll_joint_torch(sigma, sigma_model, returns, params, N, M)
#     return loss

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ll_option_torch(sigma_obs, sigma_model, sigma_eps=0.001):
    """
    Vectorized log-likelihood for option-implied volatilities.
    """
    sigma_eps_tensor = torch.tensor(sigma_eps, device=device)
    sigma_obs = sigma_obs.view(-1)
    sigma_model = sigma_model.view(-1)
    return -0.5 * torch.sum(
        2 * torch.log(sigma_eps_tensor) + ((sigma_obs - sigma_model) / sigma_eps_tensor) ** 2
    )


def ll_returns_torch(returns, params):
    """
    Vectorized Heston-Nandi GARCH log-likelihood with standard-normal shocks.
    """
    returns = returns.view(-1)
    T = returns.shape[0]

    omega, alpha, beta, gamma, lambda_ = params

    # Create list to store conditional variances
    h_list = []

    # Initialize h[0] as unconditional variance
    h_0 = (omega + alpha) / (1 - beta - alpha * gamma**2 + 1e-8)
    h_list.append(h_0)

    # Generate standard-normal shocks for the recursion
    z = torch.randn(T, device=device)

    # Recursive computation of conditional variance
    for t in range(1, T):
        h_prev = h_list[t-1]
        h_t = omega + beta * h_prev + alpha * (z[t-1] - gamma * torch.sqrt(h_prev))**2
        h_list.append(h_t)

    # Stack all conditional variances into a tensor
    h = torch.stack(h_list)

    # Compute log-likelihood using standard-normal z
    loglik = -0.5 * (torch.log(torch.tensor(2) * torch.pi) + torch.log(h) + ((returns - lambda_ * h)/torch.sqrt(h))**2)
    return torch.sum(loglik)


def ll_joint_torch(sigma_obs, sigma_model, returns, params, N, M, sigma_eps=0.001):
    """
    Joint log-likelihood combining returns and option likelihood.
    """
    lr = ll_returns_torch(returns, params)
    lo = ll_option_torch(sigma_obs, sigma_model, sigma_eps)
    return (N + M) / (2 * N) * lr + (N + M) / (2 * M) * lo


def Calibration_Loss(params, returns, sigma_obs, model, x, N, M):
    """
    Full calibration loss function.
    Returns negative joint log-likelihood.
    """
    # Predict option sigmas
    sigma_model = model(x).squeeze()

    # Ensure tensors are on correct device
    returns = returns.to(params.device)
    sigma_obs = sigma_obs.to(params.device)

    # Negative log-likelihood for minimization
    loss = -1 * ll_joint_torch(sigma_obs, sigma_model, returns, params, N, M)
    return loss


def Calibration_Loss_Regularized(params, returns, sigma_obs, model, x, N, M,
                                true_params=None, reg_type="l2", reg_weight=1.0,
                                param_weights=None):
    """
    Regularized calibration loss function with multiple regularization options.

    Args:
        params: Current parameter estimates [omega, alpha, beta, gamma, lambda]
        returns: Return series
        sigma_obs: Observed option volatilities
        model: Neural network model
        x: Feature matrix
        N, M: Data dimensions
        true_params: True parameter values for regularization (optional)
        reg_type: Type of regularization ("l2", "l1", "weighted", "bounds", "combined")
        reg_weight: Weight for regularization term
        param_weights: Individual weights for each parameter [5 elements]

    Returns:
        Regularized loss value
    """
    # Base calibration loss
    base_loss = Calibration_Loss(params, returns, sigma_obs, model, x, N, M)

    # No regularization if weight is 0
    if reg_weight == 0.0:
        return base_loss

    omega, alpha, beta, gamma, lambda_ = params
    device = params.device

    # Initialize regularization term
    reg_term = torch.tensor(0.0, device=device)

    if reg_type == "l2" and true_params is not None:
        # L2 penalty toward true parameters
        true_tensor = torch.tensor(true_params, device=device)
        reg_term = torch.sum((params - true_tensor) ** 2)

    elif reg_type == "l1" and true_params is not None:
        # L1 penalty toward true parameters
        true_tensor = torch.tensor(true_params, device=device)
        reg_term = torch.sum(torch.abs(params - true_tensor))

    elif reg_type == "weighted" and true_params is not None:
        # Weighted penalty with higher weights for gamma and lambda
        if param_weights is None:
            param_weights = [1.0, 1.0, 1.0, 5.0, 5.0]  # Higher weight for gamma, lambda
        true_tensor = torch.tensor(true_params, device=device)
        weights_tensor = torch.tensor(param_weights, device=device)
        reg_term = torch.sum(weights_tensor * (params - true_tensor) ** 2)

    elif reg_type == "bounds":
        # Penalty for parameters far from reasonable ranges
        # Encourage omega, alpha to be small; beta to be high; gamma positive; lambda small
        bounds_penalty = torch.tensor(0.0, device=device)

        # Omega should be small (around 1e-6)
        bounds_penalty += torch.relu(omega - 1e-5) * 1e12  # penalty if omega > 1e-5

        # Alpha should be small (around 1e-6)
        bounds_penalty += torch.relu(alpha - 1e-5) * 1e12  # penalty if alpha > 1e-5

        # Beta should be less than 1 (stationarity)
        bounds_penalty += torch.relu(alpha + beta - 0.99) * 1000  # stationarity penalty

        # Gamma should be positive and reasonable (0 to 20)
        bounds_penalty += torch.relu(-gamma) * 100  # penalty for negative gamma
        bounds_penalty += torch.relu(gamma - 20) * 10   # penalty for very large gamma

        # Lambda should be small (around 0 to 0.5)
        bounds_penalty += torch.relu(torch.abs(lambda_) - 0.5) * 100  # penalty if |lambda| > 0.5

        reg_term = bounds_penalty

    elif reg_type == "combined" and true_params is not None:
        # Combine L2 penalty toward true params + bounds penalty
        true_tensor = torch.tensor(true_params, device=device)

        # L2 toward true parameters
        l2_term = torch.sum((params - true_tensor) ** 2)

        # Economic constraints penalty
        bounds_penalty = torch.tensor(0.0, device=device)
        bounds_penalty += torch.relu(alpha + beta - 0.99) * 1000  # stationarity
        bounds_penalty += torch.relu(-omega) * 1e12  # omega > 0
        bounds_penalty += torch.relu(-gamma) * 100   # gamma > 0 (if assumed)

        reg_term = l2_term + 0.1 * bounds_penalty  # combine with weights

    elif reg_type == "adaptive":
        # Adaptive regularization based on current parameter values
        # Stronger penalty for parameters that are far from reasonable ranges
        adaptive_penalty = torch.tensor(0.0, device=device)

        # Adaptive omega penalty (stronger if very far from 1e-6)
        omega_target = 1e-6
        omega_ratio = torch.abs(omega / omega_target - 1.0)
        adaptive_penalty += omega_ratio ** 2 * 100

        # Adaptive gamma penalty (stronger if negative or very large)
        gamma_target = 5.0
        if true_params is not None:
            gamma_target = true_params[3]
        gamma_ratio = torch.abs(gamma / gamma_target - 1.0)
        adaptive_penalty += gamma_ratio ** 2 * 50

        # Adaptive lambda penalty
        lambda_target = 0.2
        if true_params is not None:
            lambda_target = true_params[4]
        lambda_ratio = torch.abs(lambda_ / lambda_target - 1.0)
        adaptive_penalty += lambda_ratio ** 2 * 50

        reg_term = adaptive_penalty

    # Return combined loss
    return base_loss + reg_weight * reg_term


def Calibration_Loss_MultiObjective(params, returns, sigma_obs, model, x, N, M,
                                   true_params=None, weights=[1.0, 0.1, 0.1]):
    """
    Multi-objective loss combining data fit, parameter proximity, and constraints.

    Args:
        params: Current parameter estimates
        returns, sigma_obs, model, x, N, M: Standard calibration inputs
        true_params: Target parameter values
        weights: [data_weight, param_weight, constraint_weight]

    Returns:
        Weighted combination of objectives
    """
    device = params.device

    # Objective 1: Data fitting (negative log-likelihood)
    data_loss = Calibration_Loss(params, returns, sigma_obs, model, x, N, M)

    # Objective 2: Parameter proximity (if true params available)
    param_loss = torch.tensor(0.0, device=device)
    if true_params is not None:
        true_tensor = torch.tensor(true_params, device=device)
        param_loss = torch.sum((params - true_tensor) ** 2)

    # Objective 3: Economic constraints
    omega, alpha, beta, gamma, lambda_ = params
    constraint_loss = torch.tensor(0.0, device=device)

    # Stationarity constraint: alpha + beta < 1
    constraint_loss += torch.relu(alpha + beta - 0.99) * 1000

    # Positivity constraints
    constraint_loss += torch.relu(-omega) * 1e12  # omega > 0
    constraint_loss += torch.relu(-alpha) * 1000  # alpha >= 0
    constraint_loss += torch.relu(-beta) * 1000   # beta >= 0

    # Reasonable parameter ranges
    constraint_loss += torch.relu(omega - 1e-4) * 1e8     # omega not too large
    constraint_loss += torch.relu(alpha - 0.5) * 1000     # alpha not too large
    constraint_loss += torch.relu(beta - 0.999) * 1000    # beta < 1
    constraint_loss += torch.relu(torch.abs(gamma) - 50) * 10  # |gamma| reasonable
    constraint_loss += torch.relu(torch.abs(lambda_) - 2) * 100  # |lambda| reasonable

    # Combine objectives
    total_loss = (weights[0] * data_loss +
                  weights[1] * param_loss +
                  weights[2] * constraint_loss)

    return total_loss
