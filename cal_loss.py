import math

import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps:0"
    if torch.backends.mps.is_available()
    else "cpu"
)


def ll_option_torch(sigma_obs, sigma_model, sigma_eps=0.01):
    """
    Vectorized log-likelihood for option-implied volatilities.
    """
    device = sigma_obs.device
    sigma_eps_tensor = torch.tensor(sigma_eps, device=device)
    sigma_obs = sigma_obs.view(-1)
    sigma_model = sigma_model.view(-1)
    return -0.5 * torch.sum(
        2 * torch.log(sigma_eps_tensor)
        + ((sigma_obs - sigma_model) / sigma_eps_tensor) ** 2
    )


def ll_returns_torch(returns, params, r_daily=0.0):
    """
    Vectorized Heston-Nandi GARCH log-likelihood using observed returns.

    Model:
    R_t = r + lambda * h_t + sqrt(h_t) * z_t
    h_{t+1} = omega + beta * h_t + alpha * (z_t - gamma * sqrt(h_t))^2

    where z_t ~ N(0,1)
    """
    returns = returns.view(-1)
    T = returns.shape[0]

    omega, alpha, beta, gamma, lambda_ = params

    # Initialize lists to store values for likelihood computation
    h_list = []
    z_list = []

    # Initialize h[0] as unconditional variance
    # h_unc = (omega + alpha) / (1 - beta - alpha * gamma**2)

    # Check stationarity for initialization
    persistence = beta + alpha * gamma**2
    if persistence < 0.999:
        h_curr = (omega + alpha) / (1 - persistence + 1e-8)
    else:
        # Fallback to sample variance if non-stationary parameters
        h_curr = torch.var(returns)

    # Ensure positive variance
    if h_curr < 1e-8 or torch.isnan(h_curr):
        h_curr = torch.tensor(1e-6, device=returns.device)

    # Loop through time steps
    # Note: This loop is necessary because h_t depends on h_{t-1}
    for t in range(T):
        # Stability check inside loop to prevent NaNs
        if torch.isnan(h_curr) or torch.isinf(h_curr) or h_curr > 1e6:
            # If variance explodes or becomes NaN, return very large negative log-likelihood (penalty)
            return torch.tensor(-1e9, device=returns.device)

        # Clamp variance to prevent division by zero
        h_curr = torch.clamp(h_curr, min=1e-9)

        # Store current variance
        h_list.append(h_curr)

        # Calculate standardized residual z_t
        # z_t = (R_t - r - lambda * h_t) / sqrt(h_t)
        z_t = (returns[t] - r_daily - lambda_ * h_curr) / (torch.sqrt(h_curr) + 1e-8)
        z_list.append(z_t)

        # Update variance for next step
        # h_{t+1} = omega + beta * h_t + alpha * (z_t - gamma * sqrt(h_t))^2
        h_curr = (
            omega + (beta * h_curr) + alpha * (z_t - gamma * torch.sqrt(h_curr)) ** 2
        )

    # Stack lists into tensors
    h_vec = torch.stack(h_list)
    z_vec = torch.stack(z_list)

    # Compute log-likelihood
    # LL = -0.5 * sum( log(2*pi) + log(h_t) + z_t^2 )
    logik = -0.5 * torch.sum(
        torch.log(torch.tensor(2 * math.pi, device=returns.device))
        + torch.log(h_vec + 1e-8)
        + z_vec**2
    )

    return logik


def ll_joint_torch(
    sigma_obs, sigma_model, returns, params, N, M, r_daily=0.0, sigma_eps=0.01
):
    """
    Joint log-likelihood combining returns and option likelihood.
    """
    lr = ll_returns_torch(returns, params, r_daily)
    lo = ll_option_torch(sigma_obs, sigma_model, sigma_eps)

    # Use Mean Log-Likelihoods to make them scale-invariant to dataset size
    mean_lr = lr / N
    mean_lo = lo / M

    # Weighting scheme to balance the two objectives
    # Returns likelihood is typically much larger in magnitude per point than options
    # Previously we strongly downweighted returns (w_ret=0.05) to prioritize options.
    # For testing, increase returns weight so returns influence calibration more.
    # TODO: expose these weights as configurable hyperparameters.
    w_ret = 0.5  # increased for testing — make configurable if desired
    w_opt = 1.0

    return w_ret * mean_lr + w_opt * mean_lo


def Calibration_Loss(params, returns, sigma_obs, model, x, N, M):
    """
    Full calibration loss function.
    Returns negative joint log-likelihood.

    Injects current optimization parameters into the model input features
    to ensure gradients flow from the option pricing model to the parameters.
    """
    # Ensure tensors are on correct device
    returns = returns.to(params.device)
    sigma_obs = sigma_obs.to(params.device)
    x = x.to(params.device)

    # Inject current parameters into model input features x
    # We need to clone x to avoid modifying the original tensor in the dataset/batch
    x_in = x.clone()

    # Unpack parameters
    omega, alpha, beta, gamma, lambda_ = params

    # Indices based on dataset2.py:
    # base_features: ["S0", "m", "r", "T", "corp", "alpha", "beta", "omega", "gamma", "lambda", "V"]
    # Indices:
    # 0: S0, 1: m, 2: r, 3: T, 4: corp
    # 5: alpha, 6: beta, 7: omega, 8: gamma, 9: lambda
    # 10: V

    # engineered_features starts at index 11
    # "log_gamma" (19), "log_omega" (20), "log_lambda" (21), "log_alpha" (22), "log_beta" (23)

    epsilon = 1e-8

    # Handle both batched (2D) and single (1D) input
    if x_in.dim() == 1:
        # Extract risk-free rate (index 2)
        r_daily = x_in[2]

        # Update base parameters
        x_in[5] = alpha
        x_in[6] = beta
        x_in[7] = omega
        x_in[8] = gamma
        x_in[9] = lambda_

        # Update log parameters
        x_in[19] = torch.log(gamma + epsilon)
        x_in[20] = torch.log(omega + epsilon)
        x_in[21] = torch.log(lambda_ + epsilon)
        x_in[22] = torch.log(alpha + epsilon)
        x_in[23] = torch.log(beta + epsilon)
    else:
        # Extract risk-free rate (index 2) - take mean of batch
        r_daily = x_in[:, 2].mean()

        # Update base parameters
        x_in[:, 5] = alpha
        x_in[:, 6] = beta
        x_in[:, 7] = omega
        x_in[:, 8] = gamma
        x_in[:, 9] = lambda_

        # Update log parameters
        x_in[:, 19] = torch.log(gamma + epsilon)
        x_in[:, 20] = torch.log(omega + epsilon)
        x_in[:, 21] = torch.log(lambda_ + epsilon)
        x_in[:, 22] = torch.log(alpha + epsilon)
        x_in[:, 23] = torch.log(beta + epsilon)

    # Predict option sigmas using updated features
    sigma_model = model(x_in).squeeze()

    # Negative log-likelihood for minimization
    # We want to maximize Joint Likelihood, so we minimize Negative Joint Likelihood

    joint_loss = ll_joint_torch(
        sigma_obs, sigma_model, returns, params, N, M, r_daily, sigma_eps=0.01
    )

    # Total loss to minimize
    # Note: joint_loss is LogLikelihood (higher is better), so we negate it
    loss = -joint_loss

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(1e9, device=params.device)

    return loss
