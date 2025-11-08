#!/usr/bin/env python3
"""
Simple script to run Differential Evolution GARCH calibration

This script provides an easy way to run DE calibration with configurable parameters.
Usage: python run_de_calibration.py
"""

import torch
import numpy as np
import time
import json
import argparse
from pathlib import Path

# Import our modules
from dataset2 import cal_dataset
from calibrate2_de import calibrate_de
from cal_loss import Calibration_Loss

def main():
    """Main function to run DE calibration"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Differential Evolution GARCH Calibration')
    parser.add_argument('--model-path', type=str,
                       default='saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt',
                       help='Path to the trained model')
    parser.add_argument('--options-data', type=str,
                       default='joint_dataset/scalable_hn_dataset_250x60.csv',
                       help='Path to options dataset')
    parser.add_argument('--returns-data', type=str,
                       default='joint_dataset/assetprices.csv',
                       help='Path to returns dataset')
    parser.add_argument('--popsize', type=int, default=50,
                       help='DE population size (default: 50)')
    parser.add_argument('--max-iter', type=int, default=500,
                       help='Maximum iterations (default: 500)')
    parser.add_argument('--mutation', type=str, default='0.5,1.0',
                       help='DE mutation factor or range (default: 0.5,1.0)')
    parser.add_argument('--crossover', type=float, default=0.7,
                       help='DE crossover probability (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='de_calibration_results.json',
                       help='Output file for results (default: de_calibration_results.json)')

    # Regularization options
    parser.add_argument('--reg-type', type=str, default=None,
                       choices=['l2', 'l1', 'weighted', 'bounds', 'combined', 'adaptive', 'multi'],
                       help='Regularization type (default: None)')
    parser.add_argument('--reg-weight', type=float, default=0.1,
                       help='Regularization weight (default: 0.1)')
    parser.add_argument('--reg-param-weights', type=str, default='1.0,1.0,1.0,5.0,5.0',
                       help='Parameter weights for weighted regularization (default: 1.0,1.0,1.0,5.0,5.0)')
    parser.add_argument('--reg-multi-weights', type=str, default='1.0,0.1,0.1',
                       help='Multi-objective weights [data,param,constraint] (default: 1.0,0.1,0.1)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output (default: False)')

    args = parser.parse_args()

    # Device setup
    if args.cpu:
        device = torch.device("cpu")
        print("🖥️  Using CPU (forced)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {device}")

    print("🧬 Differential Evolution GARCH Calibration")
    print("=" * 60)

    # Print configuration
    print(f"📋 Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Options data: {args.options_data}")
    print(f"   Returns data: {args.returns_data}")
    print(f"   Population size: {args.popsize}")
    print(f"   Max iterations: {args.max_iter}")
    # Parse mutation parameter
    if ',' in args.mutation:
        mutation_range = tuple(map(float, args.mutation.split(',')))
        mutation_str = f"range {mutation_range}"
    else:
        mutation_range = float(args.mutation)
        mutation_str = f"fixed {mutation_range}"

    print(f"   Mutation factor: {mutation_str}")
    print(f"   Crossover prob: {args.crossover}")
    print(f"   Random seed: {args.seed}")
    print(f"   Output file: {args.output}")

    # Parse regularization options
    regularization = None
    if args.reg_type is not None:
        param_weights = list(map(float, args.reg_param_weights.split(',')))
        multi_weights = list(map(float, args.reg_multi_weights.split(',')))

        regularization = {
            'type': args.reg_type,
            'weight': args.reg_weight,
            'true_params': [1e-6, 1.33e-6, 0.8, 5.0, 0.2],
            'param_weights': param_weights,
            'multi_weights': multi_weights
        }

        print(f"   Regularization: {args.reg_type} (weight: {args.reg_weight})")
        if args.reg_type == 'weighted':
            print(f"   Parameter weights: {param_weights}")
        elif args.reg_type == 'multi':
            print(f"   Multi-objective weights: {multi_weights}")
    else:
        print(f"   Regularization: None")

    try:
        # Check if files exist
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        if not Path(args.options_data).exists():
            raise FileNotFoundError(f"Options data file not found: {args.options_data}")
        if not Path(args.returns_data).exists():
            raise FileNotFoundError(f"Returns data file not found: {args.returns_data}")

        # Load model
        print(f"\n📦 Loading model from {args.model_path}...")
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        model.eval()
        print("✅ Model loaded successfully")

        # Load dataset
        print(f"\n📊 Loading dataset...")
        dataset = cal_dataset(args.options_data, args.returns_data)
        print(f"✅ Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

        # Validate dataset
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        if len(dataset.returns) == 0:
            raise ValueError("No returns data found")

        # Print dataset info
        print(f"   Options data shape: {len(dataset)} observations")
        print(f"   Returns data length: {len(dataset.returns)}")
        if hasattr(dataset, 'X'):
            print(f"   Feature dimensions: {dataset.X.shape}")

        # True parameter values for comparison (if known)
        true_params = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        print(f"\n🎯 Target parameters:")
        for name, val in zip(param_names, true_params):
            print(f"   {name}: {val}")

        # Run DE calibration
        print(f"\n🚀 Starting Differential Evolution calibration...")
        print(f"   This may take several minutes depending on configuration...")

        start_time = time.time()

        calibrated_params, convergence_history = calibrate_de(
            model=model,
            dataset=dataset,
            popsize=args.popsize,
            max_iter=args.max_iter,
            mutation=mutation_range,
            crossover=args.crossover,
            seed=args.seed,
            device=device,
            regularization=regularization
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\n✅ Calibration completed in {elapsed_time:.2f} seconds")

        # Calculate errors
        l2_error = np.linalg.norm(calibrated_params - true_params, ord=2)
        abs_errors = np.abs(calibrated_params - true_params)
        rel_errors = abs_errors / np.abs(true_params) * 100

        # Print results
        print(f"\n📊 Results:")
        print(f"{'Parameter':<8} {'True':<12} {'Calibrated':<12} {'Abs Error':<12} {'Rel Error %':<12}")
        print("-" * 60)

        for i, name in enumerate(param_names):
            print(f"{name:<8} {true_params[i]:<12.6f} {calibrated_params[i]:<12.6f} "
                  f"{abs_errors[i]:<12.6f} {rel_errors[i]:<12.2f}")

        print(f"\nOverall L2 Error: {l2_error:.6f}")

        # Validate results
        alpha, beta = calibrated_params[1], calibrated_params[2]
        persistence = alpha + beta
        omega_positive = calibrated_params[0] > 0
        stationarity = persistence < 1.0

        print(f"\n🔍 Validation:")
        print(f"   ω > 0: {'✅' if omega_positive else '❌'} ({calibrated_params[0]:.8f})")
        print(f"   α + β < 1: {'✅' if stationarity else '❌'} ({persistence:.6f})")

        if stationarity:
            unconditional_var = calibrated_params[0] / (1 - persistence)
            empirical_var = dataset.returns.var().item()
            print(f"   Theoretical var: {unconditional_var:.8f}")
            print(f"   Empirical var: {empirical_var:.8f}")
            print(f"   Var ratio: {unconditional_var/empirical_var:.4f}")

        # Save results
        results = {
            'configuration': {
                'popsize': args.popsize,
                'max_iter': args.max_iter,
                'mutation': mutation_range,
                'crossover': args.crossover,
                'seed': args.seed,
                'device': str(device),
                'regularization': regularization
            },
            'dataset_info': {
                'num_options': len(dataset),
                'num_returns': len(dataset.returns),
                'options_file': args.options_data,
                'returns_file': args.returns_data
            },
            'true_parameters': dict(zip(param_names, true_params.tolist())),
            'calibrated_parameters': dict(zip(param_names, calibrated_params.tolist())),
            'errors': {
                'l2_error': float(l2_error),
                'absolute_errors': dict(zip(param_names, abs_errors.tolist())),
                'relative_errors_percent': dict(zip(param_names, rel_errors.tolist()))
            },
            'validation': {
                'omega_positive': bool(omega_positive),
                'stationarity': bool(stationarity),
                'persistence': float(persistence),
                'unconditional_variance': float(unconditional_var) if stationarity else None,
                'empirical_variance': float(empirical_var)
            },
            'convergence_history': convergence_history,
            'timing': {
                'total_seconds': elapsed_time,
                'iterations_completed': len(convergence_history)
            },
            'final_objective_value': convergence_history[-1] if convergence_history else None
        }

        # Save to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {args.output}")

        # Create convergence plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(convergence_history, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Function Value')
            plt.title('Differential Evolution Convergence')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            plot_file = args.output.replace('.json', '_convergence.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📈 Convergence plot saved to: {plot_file}")

        except ImportError:
            print("📈 Matplotlib not available, skipping convergence plot")
        except Exception as e:
            print(f"⚠️  Could not create convergence plot: {e}")

        # Summary
        print(f"\n🎉 Calibration Summary:")
        print(f"   ✅ Successfully calibrated {len(param_names)} parameters")
        print(f"   ⏱️  Completed in {elapsed_time:.1f} seconds")
        print(f"   🎯 L2 error: {l2_error:.6f}")
        print(f"   🔍 Constraints: {'All satisfied' if omega_positive and stationarity else 'Some violated'}")

        return 0

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"   Please check that all data files exist in the specified paths")
        return 1

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
