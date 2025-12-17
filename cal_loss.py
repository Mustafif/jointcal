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
        2 * torch.log(sigma_eps_tensor)
        + ((sigma_obs - sigma_model) / sigma_eps_tensor) ** 2
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
        h_prev = h_list[t - 1]
        h_t = (
            omega
            + (beta * h_prev)
            + (alpha * (z[t - 1] - gamma * torch.sqrt(h_prev))) ** 2
        )
        h_list.append(h_t)

    # Stack all conditional variances into a tensor
    h = torch.stack(h_list)

    # Compute log-likelihood using standard-normal z
    logik = -0.5 * torch.sum(
        torch.log(torch.tensor(2) * torch.pi) + torch.log(h) + z**2
    )
    return logik


def ll_joint_torch(sigma_obs, sigma_model, returns, params, N, M, sigma_eps=0.0001):
    """
    Joint log-likelihood combining returns and option likelihood.
    """
    lr = ll_returns_torch(returns, params)
    lo = ll_option_torch(sigma_obs, sigma_model, sigma_eps)
    return ((N + M) / (2 * N)) * lr + ((N + M) / (2 * M)) * lo


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
    w1 = 0.5
    w2 = 1 - w1
    joint_loss = ll_joint_torch(sigma_obs, sigma_model, returns, params, N, M)
    sigma_loss = torch.nn.MSELoss()(sigma_obs, sigma_model)
    loss = -((w1 * joint_loss) - (w2 * sigma_loss))
    return loss
