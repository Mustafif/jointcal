import torch
from dataset2 import cal_dataset
from cal_loss import Calibration_Loss
import json
import numpy as np
MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def project_parameters(params):
#     """Project parameters to valid domain without in-place operations"""
#     omega = torch.clamp(params[0], min=1e-8)
#     alpha = torch.clamp(params[1], min=0.0, max=1.0)
#     beta = torch.clamp(params[2], min=0.0, max=1.0)
#     gamma = params[3]
#     lambda_param = params[4]  # No constraint on lambda

#     return torch.stack([omega, alpha, beta, gamma, lambda_param])


# def calibrate(model, dataset, lr=5e-4, steps=100, batch_size=64):
#     """
#     Fully vectorized calibration routine.
#     Uses batch processing for option observations and keeps all gradients intact.
#     """

#     model.eval().to(device)
#     for p in model.parameters():
#         p.requires_grad = False

#     # Initialize GARCH parameters with requires_grad=True
#     params = torch.tensor([1e-6, 0.01, 0.85, 0.1, 0.0],
#                           dtype=torch.float32,
#                           requires_grad=True,
#                           device=device)

#     optimizer = torch.optim.Adam([params], lr=lr)

#     all_returns = dataset.returns.to(device)
#     N = len(all_returns)
#     M = len(dataset)

#     print(f"Starting calibration: {M} options, {N} returns")

#     # Prepare full batch of option features and sigmas
#     X_all = []
#     sigma_all = []
#     for i in range(M):
#         row = dataset.data.iloc[i]
#         features = torch.tensor(row[dataset.base_features].values, dtype=torch.float32, device=device)
#         engineered_features = torch.tensor(row[dataset.engineered_features].values, dtype=torch.float32, device=device)
#         features = torch.cat([features, engineered_features])
#         # if len(features) < 30:
#         #     features = torch.cat([features, torch.zeros(30 - len(features), device=device)])
#         # else:
#         #     features = features[:30]
#         X_all.append(features.unsqueeze(0))
#         sigma_all.append(dataset.sigma[i].to(device).unsqueeze(0))

#     X_all = torch.cat(X_all, dim=0)       # M x 30
#     sigma_all = torch.cat(sigma_all, dim=0)  # M

#     # Batch calibration loop
#     for epoch in range(steps):
#         optimizer.zero_grad()

#         # Predict sigmas for all options
#         sigma_model = model(X_all).squeeze()  # M

#         # Compute joint loss (returns + options)
#         loss = -1 * Calibration_Loss(params, all_returns, sigma_all, model, X_all, N, M)
#         loss.backward()
#         optimizer.step()

#         # Project parameters to valid domain
#         with torch.no_grad():
#             projected_params = project_parameters(params)
#             params.data.copy_(projected_params.data)

#         if epoch % 20 == 0 or epoch == steps - 1:
#             print(f"Epoch {epoch+1:3d}/{steps} | Loss: {loss.item():.6f} | Params: {params.detach().cpu().numpy()}")

#     print("\n✅ Calibration Complete")
#     return params.detach().cpu().numpy()

import torch
from cal_loss import Calibration_Loss

def project_parameters(params):
    """Project parameters to valid domain without in-place ops"""
    omega = torch.clamp(params[0], min=1e-8)
    alpha = torch.clamp(params[1], min=0.0, max=1.0)
    beta = torch.clamp(params[2], min=0.0, max=1.0)
    gamma = params[3]        # can be negative
    lambda_param = params[4] # no constraint
    return torch.stack([omega, alpha, beta, gamma, lambda_param])


# def calibrate(model, dataset, lr=5e-4, steps=500, device=None):
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model.eval().to(device)
#     for p in model.parameters():
#         p.requires_grad = False  # freeze network

#     # Initialize Heston-Nandi parameters
#     params = torch.tensor([1e-6, 0.01, 0.85, 0.1, 0.0],
#                           dtype=torch.float32, requires_grad=True, device=device)

#     optimizer = torch.optim.Adam([params], lr=lr)

#     # Vectorized tensors
#     X_all = dataset.X.to(device)        # M x num_features
#     sigma_all = dataset.sigma.to(device) # M
#     all_returns = dataset.returns.to(device)
#     N = len(all_returns)
#     M = len(dataset)

#     print(f"Starting calibration: {M} options, {N} returns")

#     for epoch in range(steps):
#         optimizer.zero_grad()

#         # Predict option implied vols
#         sigma_model = model(X_all).squeeze()  # M

#         # Compute joint calibration loss
#         loss = -1 * Calibration_Loss(params, all_returns, sigma_all, model, X_all, N, M)
#         loss.backward()
#         optimizer.step()

#         # Project parameters into valid domain
#         with torch.no_grad():
#             params.data.copy_(project_parameters(params).data)

#         if epoch % 50 == 0 or epoch == steps - 1:
#             print(f"Epoch {epoch+1:4d}/{steps} | Loss: {loss.item():.6f} | Params: {params.detach().cpu().numpy()}")

#     print("\n✅ Calibration complete")
#     return params.detach().cpu().numpy()

def calibrate(model, dataset, lr=5e-4, steps=500, batch_size=64, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False  # freeze network

    # 1️⃣ Initialize parameters using Heston-Nandi GARCH estimates
    params = dataset.target.clone().to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([params], lr=lr)
    # 2️⃣ Adaptive learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Precomputed tensors
    X_all = dataset.X.to(device)        # M x num_features
    sigma_all = dataset.sigma.to(device) # M
    all_returns = dataset.returns.to(device)
    N = len(all_returns)
    M = len(dataset)

    print(f"Starting calibration: {M} options, {N} returns")

    for epoch in range(steps):
        optimizer.zero_grad()

        loss = 0.0
        # 3️⃣ Mini-batch processing
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            X_batch = X_all[start:end]
            sigma_batch = sigma_all[start:end]

            loss_batch = -1 * Calibration_Loss(params, all_returns, sigma_batch, model, X_batch, N, len(X_batch))
            loss += loss_batch

        loss.backward()
        optimizer.step()
        scheduler.step()  # update learning rate

        # Project parameters into valid domain
        with torch.no_grad():
            omega = torch.clamp(params[0], min=1e-8)
            alpha = torch.clamp(params[1], min=0.0, max=1.0)
            beta  = torch.clamp(params[2], min=0.0, max=1.0)
            gamma = params[3]
            lambda_param = params[4]
            params.data.copy_(torch.stack([omega, alpha, beta, gamma, lambda_param]).data)

        if epoch % 50 == 0 or epoch == steps - 1:
            print(f"Epoch {epoch+1:4d}/{steps} | Loss: {loss.item():.6f} | Params: {params.detach().cpu().numpy()}")

    print("\n✅ Calibration complete")
    return params.detach().cpu().numpy()


def main():
    try:
        # Anomaly detection disabled for better performance
        # torch.autograd.set_detect_anomaly(True)

        print("Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print("Model loaded successfully.")

        print("Loading dataset...")
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                              "joint_dataset/assetprices.csv")
        print(f"Dataset: {len(dataset)} options, returns length {len(dataset.returns)}")

        calibrated_params = calibrate(model, dataset, lr=5e-4, steps=100)

        true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        pred_vals = calibrated_params

        two_norm = np.linalg.norm(pred_vals - true_vals, ord=2)

        print(f"Two-norm error: {two_norm:.6f}")

        results = dict(zip(['omega', 'alpha', 'beta', 'gamma', 'lambda'], calibrated_params.tolist()))
        with open('calibrated_params.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n📁 Results saved to calibrated_params.json")

        # Validation
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta
        print(f"\nValidation: α+β = {persistence:.6f}")
        if persistence < 1.0:
            print("✅ Stationarity satisfied")
        else:
            print("⚠️  Stationarity violated")

        if calibrated_params[0] > 0:
            print("✅ Omega positive")
        else:
            print("⚠️  Omega not positive")

        if persistence < 1.0:
            unconditional_var = calibrated_params[0] / (1 - persistence)
            print(f"Theoretical unconditional variance: {unconditional_var:.8f}")
            empirical_var = dataset.returns.var().item()
            print(f"Empirical returns variance: {empirical_var:.8f}")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
