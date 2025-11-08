import torch
import numpy as np
from dataset2 import cal_dataset
from cal_loss import Calibration_Loss
from torch.utils.data import DataLoader

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_dataset():
    """Debug the dataset to understand its structure"""
    print("🔍 Debugging dataset structure...")

    dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv", "joint_dataset/assetprices.csv")
    print(f"Dataset length: {len(dataset)}")

    # Test single item
    try:
        item = dataset[0]
        print(f"Single item type: {type(item)}")
        print(f"Single item length: {len(item) if hasattr(item, '__len__') else 'No length'}")

        if isinstance(item, (list, tuple)):
            for i, element in enumerate(item):
                print(f"  Element {i}: type={type(element)}, shape={element.shape if hasattr(element, 'shape') else element}")

    except Exception as e:
        print(f"Error accessing single item: {e}")
        return None

    # Test dataloader
    try:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        print(f"\nBatch type: {type(batch)}")
        print(f"Batch length: {len(batch) if hasattr(batch, '__len__') else 'No length'}")

        if isinstance(batch, (list, tuple)):
            for i, element in enumerate(batch):
                print(f"  Batch element {i}: type={type(element)}, shape={element.shape if hasattr(element, 'shape') else element}")

    except Exception as e:
        print(f"Error with dataloader: {e}")
        return None

    return dataset

def load_and_debug_model():
    """Load model and check its structure"""
    print("\n🔍 Debugging model structure...")

    try:
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")

        # Check input size
        if hasattr(model, 'stem') and hasattr(model.stem, '0'):
            first_layer = model.stem[0]
            if hasattr(first_layer, 'in_features'):
                print(f"Expected input features: {first_layer.in_features}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def calibrate_fixed(model, dataset, lr=1e-3, steps=100):
    """
    Fixed calibration function with proper error handling
    """
    print("\n🚀 Starting calibration...")

    # Use batch size of 1 to avoid batching issues
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Initialize GARCH parameters (ω, α, β, γ, λ)
    params = torch.tensor([1e-5, 0.05, 0.9, 0.1, 0.0],
                         dtype=torch.float32,
                         requires_grad=True,
                         device=device)

    optimizer = torch.optim.Adam([params], lr=lr)

    successful_batches = 0
    total_loss = 0.0

    for epoch in range(steps):
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            try:
                # Unpack batch data carefully
                if len(batch_data) != 6:
                    print(f"Warning: Expected 6 elements, got {len(batch_data)}")
                    continue

                x, target, returns_item, sigma_item, N, M = batch_data

                # Move to device and handle dimensions
                x = x.to(device).float()
                returns_item = returns_item.to(device).float()
                sigma_item = sigma_item.to(device).float()

                # Handle batch dimensions
                if x.dim() == 3:  # [batch, seq, features]
                    x = x.squeeze(0)  # Remove batch dimension if it's 1
                elif x.dim() == 1:  # [features] - add batch dimension
                    x = x.unsqueeze(0)

                # Pad input to match model expectations (30 features)
                if x.shape[-1] < 30:
                    padding_size = 30 - x.shape[-1]
                    padding = torch.zeros(*x.shape[:-1], padding_size, device=device)
                    x = torch.cat([x, padding], dim=-1)
                elif x.shape[-1] > 30:
                    x = x[..., :30]  # Truncate if too many features

                # Extract scalar values
                if returns_item.dim() > 0:
                    returns_val = returns_item.squeeze()
                else:
                    returns_val = returns_item

                if sigma_item.dim() > 0:
                    sigma_val = sigma_item.squeeze()
                else:
                    sigma_val = sigma_item

                # Get N and M as integers
                N_val = int(N.item()) if hasattr(N, 'item') else int(N)
                M_val = int(M.item()) if hasattr(M, 'item') else int(M)

                optimizer.zero_grad()

                # Compute loss
                loss = Calibration_Loss(params, returns_val, sigma_val, model, x, N_val, M_val)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at batch {batch_idx}")
                    continue

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                successful_batches += 1

                # Break after processing a few batches to avoid memory issues
                if batch_idx >= 10:
                    break

            except Exception as e:
                print(f"Warning: Error in batch {batch_idx}: {e}")
                continue

        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
            total_loss = avg_loss

        # Print progress
        if epoch % 20 == 0 or epoch == steps - 1:
            print(f"Epoch {epoch+1:3d}/{steps} | Avg Loss: {total_loss:.6f} | Success: {successful_batches}")
            print(f"  Params: ω={params[0].item():.2e}, α={params[1].item():.4f}, β={params[2].item():.4f}, γ={params[3].item():.4f}, λ={params[4].item():.4f}")

    print(f"\n✅ Calibration Complete - {successful_batches} successful batches")
    final_params = params.detach().cpu().numpy()

    print(f"Final Parameters:")
    print(f"  ω (omega): {final_params[0]:.2e}")
    print(f"  α (alpha): {final_params[1]:.4f}")
    print(f"  β (beta):  {final_params[2]:.4f}")
    print(f"  γ (gamma): {final_params[3]:.4f}")
    print(f"  λ (lambda): {final_params[4]:.4f}")

    return final_params

if __name__ == "__main__":
    # Debug dataset
    dataset = debug_dataset()
    if dataset is None:
        print("❌ Dataset debugging failed")
        exit(1)

    # Debug model
    model = load_and_debug_model()
    if model is None:
        print("❌ Model loading failed")
        exit(1)

    # Run calibration
    try:
        calibrated_params = calibrate_fixed(model, dataset, lr=1e-3, steps=50)

        # Save results
        results = {
            'omega': float(calibrated_params[0]),
            'alpha': float(calibrated_params[1]),
            'beta': float(calibrated_params[2]),
            'gamma': float(calibrated_params[3]),
            'lambda': float(calibrated_params[4])
        }

        import json
        with open('debug_calibrated_params.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to debug_calibrated_params.json")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
