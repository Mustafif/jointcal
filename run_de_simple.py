#!/usr/bin/env python3
"""
Simple script to run Differential Evolution GARCH calibration with default settings.
Just run: python run_de_simple.py
"""

import torch
import numpy as np
import time
import json
from pathlib import Path

# Import our modules
from dataset2 import cal_dataset
from calibrate2_de import calibrate_de

def main():
    """Run DE calibration with sensible defaults"""

    print("🧬 Quick Differential Evolution GARCH Calibration")
    print("=" * 55)

    # Default configuration
    config = {
        'model_path': 'saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt',
        'options_data': 'joint_dataset/scalable_hn_dataset_250x60.csv',
        'returns_data': 'joint_dataset/assetprices.csv',
        'popsize': 50,
        'max_iter': 300,
        'mutation': 0.8,
        'crossover': 0.7,
        'seed': 42
    }

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Print config
    print(f"\n📋 Configuration:")
    print(f"   Population size: {config['popsize']}")
    print(f"   Max iterations: {config['max_iter']}")
    print(f"   Mutation: {config['mutation']}")
    print(f"   Crossover: {config['crossover']}")

    try:
        # Check files exist
        model_file = Path(config['model_path'])
        options_file = Path(config['options_data'])
        returns_file = Path(config['returns_data'])

        if not model_file.exists():
            print(f"❌ Model file not found: {config['model_path']}")
            print("   Please update the model_path in the script")
            return 1

        if not options_file.exists():
            print(f"❌ Options data not found: {config['options_data']}")
            print("   Please update the options_data path in the script")
            return 1

        if not returns_file.exists():
            print(f"❌ Returns data not found: {config['returns_data']}")
            print("   Please update the returns_data path in the script")
            return 1

        # Load model
        print(f"\n📦 Loading model...")
        model = torch.load(config['model_path'], map_location=device, weights_only=False)
        model.eval()
        print("✅ Model loaded")

        # Load dataset
        print(f"📊 Loading dataset...")
        dataset = cal_dataset(config['options_data'], config['returns_data'])
        print(f"✅ Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

        # True parameters for comparison
        true_params = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        # Run calibration
        print(f"\n🚀 Starting calibration (this may take a few minutes)...")
        start_time = time.time()

        calibrated_params, history = calibrate_de(
            model=model,
            dataset=dataset,
            popsize=config['popsize'],
            max_iter=config['max_iter'],
            mutation=config['mutation'],
            crossover=config['crossover'],
            seed=config['seed'],
            device=device
        )

        elapsed = time.time() - start_time

        # Results
        print(f"\n✅ Completed in {elapsed:.1f} seconds")

        l2_error = np.linalg.norm(calibrated_params - true_params, ord=2)

        print(f"\n📊 Results:")
        print(f"{'Param':<8} {'True':<10} {'Found':<10} {'Error':<10}")
        print("-" * 40)

        for i, name in enumerate(param_names):
            error = abs(calibrated_params[i] - true_params[i])
            print(f"{name:<8} {true_params[i]:<10.6f} {calibrated_params[i]:<10.6f} {error:<10.6f}")

        print(f"\nTotal L2 Error: {l2_error:.6f}")

        # Check constraints
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta

        print(f"\n🔍 Validation:")
        print(f"   ω > 0: {'✅' if calibrated_params[0] > 0 else '❌'}")
        print(f"   α + β < 1: {'✅' if persistence < 1.0 else '❌'} ({persistence:.4f})")

        # Save results
        results = {
            'calibrated_parameters': dict(zip(param_names, calibrated_params.tolist())),
            'true_parameters': dict(zip(param_names, true_params.tolist())),
            'l2_error': float(l2_error),
            'time_seconds': elapsed,
            'convergence_history': history,
            'validation': {
                'omega_positive': bool(calibrated_params[0] > 0),
                'stationary': bool(persistence < 1.0),
                'persistence': float(persistence)
            }
        }

        output_file = 'de_quick_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {output_file}")

        # Quick plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(history)
            plt.title('DE Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.savefig('de_quick_convergence.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"📈 Convergence plot: de_quick_convergence.png")

        except ImportError:
            pass

        print(f"\n🎉 Done! L2 error: {l2_error:.6f}")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
