import torch
import numpy as np
from dataset2 import cal_dataset
from cal_loss import ll_returns_torch
import matplotlib.pyplot as plt
from scipy import stats

def test_standardized_residuals():
    """Test if standardized residuals from GARCH model are actually N(0,1)"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                         "joint_dataset/assetprices.csv")

    # Get returns data
    returns = dataset.returns.to(device)
    print(f"Returns shape: {returns.shape}")
    print(f"Returns stats: mean={returns.mean():.6f}, std={returns.std():.6f}")

    # Test with different parameter sets
    param_sets = [
        [1e-5, 0.05, 0.9, 0.1, 0.0],     # Simple GARCH
        [1e-4, 0.02, 0.8, 0.0, 0.0],     # No leverage
        [5e-5, 0.1, 0.85, 0.2, -0.1],    # With leverage and risk premium
        [6.32e-05, 0.0172, 0.7954, 0.0147, -0.3867]  # Calibrated params
    ]

    for i, param_list in enumerate(param_sets):
        print(f"\n=== Parameter Set {i+1} ===")
        params = torch.tensor(param_list, dtype=torch.float32, device=device, requires_grad=False)
        print(f"ω={params[0]:.2e}, α={params[1]:.4f}, β={params[2]:.4f}, γ={params[3]:.4f}, λ={params[4]:.4f}")

        # Use a subset of returns for testing
        test_returns = returns[:100]  # Use first 100 returns

        # Compute GARCH likelihood and extract residuals
        omega, alpha, beta, gamma, lambda_ = params
        T = test_returns.shape[0]
        h = torch.zeros(T, device=device)

        # Initialize h[0] as unconditional variance
        h[0] = (omega + alpha) / (1 - beta - alpha * gamma**2 + 1e-8)
        h[0] = torch.clamp(h[0], min=1e-9)

        # Compute variance recursion
        for t in range(1, T):
            h_prev = torch.clamp(h[t-1], min=1e-9)
            # Use previous return to compute previous standardized residual
            z_prev = (test_returns[t-1] - lambda_ * h_prev) / torch.sqrt(h_prev)
            # Update conditional variance
            h[t] = omega + beta * h_prev + alpha * (z_prev - gamma * torch.sqrt(h_prev))**2
            h[t] = torch.clamp(h[t], min=1e-9)

        # Compute standardized residuals
        z = (test_returns - lambda_ * h) / torch.sqrt(h)

        # Convert to numpy for analysis
        z_np = z.detach().cpu().numpy()

        print(f"Standardized residuals stats:")
        print(f"  Mean: {np.mean(z_np):.6f} (should be ~0)")
        print(f"  Std:  {np.std(z_np, ddof=1):.6f} (should be ~1)")
        print(f"  Min:  {np.min(z_np):.3f}")
        print(f"  Max:  {np.max(z_np):.3f}")

        # Test normality
        _, p_value = stats.normaltest(z_np)
        print(f"  Normality test p-value: {p_value:.4f} (>0.05 suggests normal)")

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(z_np)
        print(f"  Jarque-Bera p-value: {jb_p:.4f} (>0.05 suggests normal)")

        # Check for obvious patterns
        print(f"  Skewness: {stats.skew(z_np):.4f} (should be ~0)")
        print(f"  Kurtosis: {stats.kurtosis(z_np):.4f} (should be ~0 for normal)")

def test_garch_properties():
    """Test basic GARCH properties"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                         "joint_dataset/assetprices.csv")

    returns = dataset.returns[:500].to(device)  # Use first 500 returns

    # Test with calibrated parameters
    params = torch.tensor([6.32e-05, 0.0172, 0.7954, 0.0147, -0.3867],
                         dtype=torch.float32, device=device)

    omega, alpha, beta, gamma, lambda_ = params
    T = returns.shape[0]
    h = torch.zeros(T, device=device)

    # Initialize h[0]
    h[0] = (omega + alpha) / (1 - beta - alpha * gamma**2 + 1e-8)
    h[0] = torch.clamp(h[0], min=1e-9)

    print(f"\n=== GARCH Properties Test ===")
    print(f"Unconditional variance h[0]: {h[0].item():.8f}")
    print(f"Returns variance: {torch.var(returns).item():.8f}")

    # Check stationarity condition
    persistence = alpha + beta
    print(f"Persistence (α+β): {persistence.item():.6f} (should be < 1)")

    # Theoretical unconditional variance
    if persistence < 1:
        theory_var = omega / (1 - alpha - beta)
        print(f"Theoretical unconditional var: {theory_var.item():.8f}")

    # Compute full GARCH process
    z_all = torch.zeros(T, device=device)

    for t in range(1, T):
        h_prev = torch.clamp(h[t-1], min=1e-9)
        z_prev = (returns[t-1] - lambda_ * h_prev) / torch.sqrt(h_prev)
        z_all[t-1] = z_prev
        h[t] = omega + beta * h_prev + alpha * (z_prev - gamma * torch.sqrt(h_prev))**2
        h[t] = torch.clamp(h[t], min=1e-9)

    # Last residual
    z_all[T-1] = (returns[T-1] - lambda_ * h[T-1]) / torch.sqrt(h[T-1])

    h_np = h.detach().cpu().numpy()
    z_np = z_all.detach().cpu().numpy()

    print(f"Conditional variance stats:")
    print(f"  Mean: {np.mean(h_np):.8f}")
    print(f"  Std:  {np.std(h_np):.8f}")
    print(f"  Min:  {np.min(h_np):.8f}")
    print(f"  Max:  {np.max(h_np):.8f}")

    print(f"Final standardized residuals:")
    print(f"  Mean: {np.mean(z_np):.6f}")
    print(f"  Std:  {np.std(z_np, ddof=1):.6f}")

if __name__ == "__main__":
    print("Testing Standardized Residuals in GARCH Model")
    print("=" * 50)

    test_standardized_residuals()
    test_garch_properties()

    print("\n" + "=" * 50)
    print("Testing complete!")
