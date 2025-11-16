import numpy as np
import torch

# Option 2: SciPy DE implementation (recommended - professional grade)
from calibrate_scipy_de import calibrate_scipy_de
from dataset2 import cal_dataset

# Load model and dataset
MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 GARCH Parameter Calibration with Differential Evolution")
print("=" * 60)

print("Loading model...")
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
print("✅ Model loaded successfully")

print("Loading dataset...")
dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                      "joint_dataset/assetprices.csv")
print(f"✅ Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

# Run SciPy DE calibration (recommended - best balance of speed and accuracy)
print(f"\n🔬 Running SciPy Differential Evolution...")
params_scipy, history_scipy = calibrate_scipy_de(
    model=model,
    dataset=dataset,
    popsize=40,            # Population size multiplier (20 * 5 params = 100)
    maxiter=400,           # Maximum iterations
    strategy='rand1bin',   # DE strategy (fast and reliable)
    mutation=(0.65, 1.0),   # Adaptive mutation range
    recombination=0.7,     # Crossover probability
    seed=42,              # For reproducibility
    polish=False,          # Use L-BFGS-B for final refinement
    atol=1e-6             # Convergence tolerance
)

true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
l2_error = np.linalg.norm(params_scipy - true_vals, ord=2)

# Use SciPy DE results by default (recommended for production)
# - Professional implementation with excellent reliability
# - Good balance of speed and accuracy
# - Well-tested and maintained
params = params_scipy
history = history_scipy

print(f"\n✅ Calibration completed!")
print(f"📊 Final calibrated parameters:")

omega, alpha, beta, gamma, lam = params

print(f"  Omega: {omega:.8f}")
print(f"  Alpha: {alpha:.8f}")
print(f"  Beta: {beta:.8f}")
print(f"  Gamma: {gamma:.8f}")
print(f"  Lambda: {lam:.8f}")

print(f"L2 error: {l2_error:.8f}")
# Validate results
alpha, beta = params[1], params[2]
persistence = alpha + beta
print(f"\n📋 Validation:")
print(f"  Persistence (α+β): {persistence:.6f}")
print(f"  Stationary: {'✅' if persistence < 1.0 else '❌'}")
print(f"  Omega > 0: {'✅' if params[0] > 0 else '❌'}")
