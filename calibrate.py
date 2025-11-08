import torch
from dataset2 import cal_dataset
from cal_loss import Calibration_Loss
import json

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calibrate(model, dataset, lr=1e-3, steps=100):
    """
    Calibrate GARCH parameters using neural network model and market data.
    Iterates through each option observation (row) in the dataset.
    """
    print(f"Starting calibration with {len(dataset)} option observations")
    print(f"Using full returns series of length {len(dataset.returns)}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False  # freeze trained network

    # Initialize GARCH parameters (ω, α, β, γ, λ)
    params = torch.tensor([1e-6, 0.01, 0.85, 0.1, 0.0],
                         dtype=torch.float32,
                         requires_grad=True,
                         device=device)

    optimizer = torch.optim.Adam([params], lr=lr)

    # Get full returns series and dimensions
    all_returns = dataset.returns.to(device)
    N = len(all_returns)  # Number of returns observations
    M = len(dataset.sigma)  # Number of option observations

    print(f"N (returns): {N}, M (options): {M}")

    # Use subset of option observations for efficiency
    max_options = len(dataset)

    for epoch in range(steps):
        total_loss = 0.0
        num_valid = 0

        # Process option observations in small batches
        batch_size = 20
        for start_idx in range(0, max_options, batch_size):
            batch_losses = []

            for i in range(start_idx, min(start_idx + batch_size, max_options)):
                try:
                    # Get option features for this observation
                    row = dataset.data.iloc[i]
                    features = torch.tensor(row[dataset.base_features].values,
                                          dtype=torch.float32, device=device)

                    # Pad to 30 features for model input
                    if len(features) < 30:
                        padding = torch.zeros(30 - len(features), device=device)
                        features = torch.cat([features, padding])
                    else:
                        features = features[:30]

                    features = features.unsqueeze(0)  # Add batch dimension

                    # Get observed implied volatility for this option
                    sigma_obs = dataset.sigma[i].to(device)

                    # Compute loss using full returns series and this option observation
                    # Calibration_Loss expects: (params, returns, sigma, model, x, N, M)
                    loss = Calibration_Loss(params, all_returns, sigma_obs, model, features, N, M)

                    if torch.isfinite(loss):
                        batch_losses.append(loss)

                except Exception as e:
                    # Skip problematic observations
                    continue

            if batch_losses:
                # Compute batch loss and backpropagate
                batch_loss = torch.stack(batch_losses).mean()

                optimizer.zero_grad()
                batch_loss.backward()

                # Clip gradients for stability
                # torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)

                optimizer.step()

                # Parameter constraints removed to avoid inplace operation conflicts
                # Constraints will be checked after calibration is complete

                total_loss += batch_loss.item()
                num_valid += len(batch_losses)

        # Print progress
        if epoch % 20 == 0 or epoch == steps - 1:
            avg_loss = total_loss / max(num_valid, 1)
            print(f"Epoch {epoch+1:3d}/{steps} | Loss: {avg_loss:.6f} | Valid: {num_valid}")
            print(f"  ω={params[0].item():.2e}, α={params[1].item():.4f}, β={params[2].item():.4f}, γ={params[3].item():.4f}, λ={params[4].item():.4f}")

    print("\n✅ Calibration Complete")
    final_params = params.detach().cpu().numpy()

    print(f"Final Parameters:")
    print(f"  ω (omega): {final_params[0]:.2e}")
    print(f"  α (alpha): {final_params[1]:.4f}")
    print(f"  β (beta):  {final_params[2]:.4f}")
    print(f"  γ (gamma): {final_params[3]:.4f}")
    print(f"  λ (lambda): {final_params[4]:.4f}")

    return final_params

def main():
    """Main calibration routine"""
    try:
        # Load model
        print("Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print(f"Model loaded successfully on {device}")

        # Load dataset
        print("Loading dataset...")
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                             "joint_dataset/assetprices.csv")
        print(f"Dataset loaded:")
        print(f"  Option observations: {len(dataset)}")
        print(f"  Returns length: {len(dataset.returns)}")
        print(f"  Sigma observations: {len(dataset.sigma)}")

        # Run calibration
        calibrated_params = calibrate(model, dataset, lr=5e-4, steps=100)

        # Save results
        results = {
            'omega': float(calibrated_params[0]),
            'alpha': float(calibrated_params[1]),
            'beta': float(calibrated_params[2]),
            'gamma': float(calibrated_params[3]),
            'lambda': float(calibrated_params[4])
        }

        with open('calibrated_params.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to calibrated_params.json")

        # Validation checks
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta
        print(f"\nValidation:")
        print(f"  Persistence (α + β): {persistence:.6f}")
        if persistence < 1.0:
            print("  ✅ Stationarity condition satisfied (α + β < 1)")
        else:
            print("  ⚠️  Warning: Stationarity condition violated (α + β ≥ 1)")

        if calibrated_params[0] > 0:
            print("  ✅ Omega is positive")
        else:
            print("  ⚠️  Warning: Omega is not positive")

        # Theoretical unconditional variance
        if persistence < 1.0:
            unconditional_var = calibrated_params[0] / (1 - persistence)
            print(f"  Theoretical unconditional variance: {unconditional_var:.8f}")
            empirical_var = dataset.returns.var().item()
            print(f"  Empirical returns variance: {empirical_var:.8f}")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
