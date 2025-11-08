import torch
import numpy as np
from dataset2 import cal_dataset
from cal_loss import Calibration_Loss

def test_single_calibration_step():
    """Test a single calibration step to isolate issues"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
    try:
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval().to(device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load dataset
    try:
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                             "joint_dataset/assetprices.csv")
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print(f"   Returns shape: {dataset.returns.shape}")
        print(f"   Sigma shape: {dataset.sigma.shape}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Test model input/output
    try:
        # Create test input (30 features as expected by model)
        test_features = torch.randn(1, 30, device=device)
        with torch.no_grad():
            model_output = model(test_features)
        print(f"✅ Model inference works - output shape: {model_output.shape}")
        print(f"   Sample output: {model_output.squeeze()[:5].cpu().numpy()}")
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return

    # Test with real data
    try:
        # Get first data point
        idx = 0
        features = torch.tensor(dataset.data.iloc[idx][dataset.base_features].values,
                              dtype=torch.float32, device=device)

        # Pad to 30 features
        if features.shape[0] < 30:
            padding = torch.zeros(30 - features.shape[0], device=device)
            features = torch.cat([features, padding])
        features = features.unsqueeze(0)  # Add batch dimension

        # Get returns and sigma values
        return_val = dataset.returns[0].to(device)
        sigma_val = dataset.sigma[0].to(device)
        N = len(dataset.returns)
        M = len(dataset.sigma)

        print(f"✅ Data extracted successfully")
        print(f"   Features shape: {features.shape}")
        print(f"   Return value: {return_val.item()}")
        print(f"   Sigma value: {sigma_val.item()}")
        print(f"   N (returns): {N}, M (sigma): {M}")

    except Exception as e:
        print(f"❌ Data extraction failed: {e}")
        return

    # Test model prediction with real data
    try:
        with torch.no_grad():
            sigma_pred = model(features)
        print(f"✅ Model prediction: {sigma_pred.item():.6f}")
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return

    # Test GARCH parameters
    try:
        params = torch.tensor([1e-5, 0.05, 0.9, 0.1, 0.0],
                            dtype=torch.float32,
                            requires_grad=True,
                            device=device)
        print(f"✅ GARCH params initialized: {params.detach().cpu().numpy()}")
    except Exception as e:
        print(f"❌ GARCH params failed: {e}")
        return

    # Test loss computation
    try:
        print("\n🔍 Testing Calibration_Loss function...")
        loss = Calibration_Loss(params, return_val, sigma_val, model, features, N, M)
        print(f"✅ Loss computed: {loss.item():.6f}")
        print(f"   Loss requires grad: {loss.requires_grad}")
        print(f"   Loss is finite: {torch.isfinite(loss).item()}")

        # Test backward pass
        loss.backward()
        print(f"✅ Backward pass successful")
        print(f"   Param gradients: {params.grad}")

    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 All tests passed! The calibration components are working.")

def test_batch_processing():
    """Test processing multiple samples"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Load components
        MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval().to(device)

        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                             "joint_dataset/assetprices.csv")

        params = torch.tensor([1e-5, 0.05, 0.9, 0.1, 0.0],
                            dtype=torch.float32,
                            requires_grad=True,
                            device=device)

        N = len(dataset.returns)
        M = len(dataset.sigma)

        print(f"\n🔍 Testing batch processing with {min(10, len(dataset))} samples...")

        total_loss = 0.0
        valid_samples = 0

        for idx in range(min(10, len(dataset))):
            try:
                # Get features
                features = torch.tensor(dataset.data.iloc[idx][dataset.base_features].values,
                                      dtype=torch.float32, device=device)

                # Pad to 30 features
                if features.shape[0] < 30:
                    padding = torch.zeros(30 - features.shape[0], device=device)
                    features = torch.cat([features, padding])
                features = features.unsqueeze(0)

                # Get return and sigma values safely
                return_val = dataset.returns[idx % len(dataset.returns)].to(device)
                sigma_val = dataset.sigma[idx % len(dataset.sigma)].to(device)

                # Compute loss
                loss = Calibration_Loss(params, return_val, sigma_val, model, features, N, M)

                if torch.isfinite(loss):
                    total_loss += loss.item()
                    valid_samples += 1
                    print(f"   Sample {idx}: loss = {loss.item():.6f}")
                else:
                    print(f"   Sample {idx}: invalid loss")

            except Exception as e:
                print(f"   Sample {idx}: error - {e}")
                continue

        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
            print(f"\n✅ Batch processing successful!")
            print(f"   Valid samples: {valid_samples}/10")
            print(f"   Average loss: {avg_loss:.6f}")
        else:
            print(f"\n❌ No valid samples processed")

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("CALIBRATION TEST SUITE")
    print("=" * 60)

    # Test single step
    test_single_calibration_step()

    # Test batch processing
    test_batch_processing()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
