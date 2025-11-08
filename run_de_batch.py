#!/usr/bin/env python3
"""
Batch script to run multiple Differential Evolution configurations for GARCH calibration.
This script tests different hyperparameter combinations to find optimal settings.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from itertools import product

# Import our modules
from dataset2 import cal_dataset
from calibrate2_de import calibrate_de

def run_batch_calibration():
    """Run multiple DE configurations and compare results"""

    print("🧪 Batch Differential Evolution GARCH Calibration")
    print("=" * 60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Configuration grid
    configurations = {
        'popsize': [30, 50, 100],
        'mutation': [0.5, 0.8, 1.2],
        'crossover': [0.5, 0.7, 0.9],
        'max_iter': [200, 500]
    }

    # Fixed parameters
    fixed_params = {
        'model_path': 'saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt',
        'options_data': 'joint_dataset/scalable_hn_dataset_250x60.csv',
        'returns_data': 'joint_dataset/assetprices.csv',
        'seed': 42
    }

    # Generate all combinations
    param_names = ['popsize', 'mutation', 'crossover', 'max_iter']
    param_values = [configurations[name] for name in param_names]
    all_combinations = list(product(*param_values))

    total_runs = len(all_combinations)
    print(f"📊 Testing {total_runs} configurations")
    print(f"   Population sizes: {configurations['popsize']}")
    print(f"   Mutation factors: {configurations['mutation']}")
    print(f"   Crossover probs: {configurations['crossover']}")
    print(f"   Max iterations: {configurations['max_iter']}")

    try:
        # Check files exist
        for file_path in [fixed_params['model_path'], fixed_params['options_data'], fixed_params['returns_data']]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load model and dataset once
        print(f"\n📦 Loading model and dataset...")
        model = torch.load(fixed_params['model_path'], map_location=device, weights_only=False)
        model.eval()

        dataset = cal_dataset(fixed_params['options_data'], fixed_params['returns_data'])
        print(f"✅ Loaded: {len(dataset)} options, {len(dataset.returns)} returns")

        # True parameters for comparison
        true_params = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        param_labels = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        # Results storage
        results = {
            'configurations': [],
            'summary': {
                'total_runs': total_runs,
                'successful_runs': 0,
                'failed_runs': 0,
                'best_config': None,
                'best_error': float('inf')
            },
            'true_parameters': dict(zip(param_labels, true_params.tolist()))
        }

        # Run all configurations
        for i, combo in enumerate(all_combinations):
            config = dict(zip(param_names, combo))

            print(f"\n🚀 Run {i+1}/{total_runs}: {config}")
            print("-" * 40)

            run_result = {
                'run_id': i + 1,
                'configuration': config,
                'success': False,
                'error_message': None,
                'calibrated_params': None,
                'l2_error': None,
                'time_seconds': None,
                'convergence_history': None,
                'validation': None
            }

            try:
                # Run calibration
                start_time = time.time()

                calibrated_params, history = calibrate_de(
                    model=model,
                    dataset=dataset,
                    popsize=config['popsize'],
                    max_iter=config['max_iter'],
                    mutation=config['mutation'],
                    crossover=config['crossover'],
                    seed=fixed_params['seed'],
                    device=device
                )

                elapsed = time.time() - start_time

                # Calculate error
                l2_error = np.linalg.norm(calibrated_params - true_params, ord=2)

                # Validation
                alpha, beta = calibrated_params[1], calibrated_params[2]
                persistence = alpha + beta
                omega_positive = calibrated_params[0] > 0
                stationary = persistence < 1.0

                # Store results
                run_result.update({
                    'success': True,
                    'calibrated_params': dict(zip(param_labels, calibrated_params.tolist())),
                    'l2_error': float(l2_error),
                    'time_seconds': elapsed,
                    'convergence_history': history,
                    'validation': {
                        'omega_positive': bool(omega_positive),
                        'stationary': bool(stationary),
                        'persistence': float(persistence)
                    }
                })

                # Update summary
                results['summary']['successful_runs'] += 1
                if l2_error < results['summary']['best_error']:
                    results['summary']['best_error'] = float(l2_error)
                    results['summary']['best_config'] = config.copy()

                print(f"✅ Success - L2 error: {l2_error:.6f}, Time: {elapsed:.1f}s")
                print(f"   Constraints: {'✅' if omega_positive and stationary else '❌'}")

            except Exception as e:
                run_result.update({
                    'success': False,
                    'error_message': str(e)
                })
                results['summary']['failed_runs'] += 1
                print(f"❌ Failed: {e}")

            results['configurations'].append(run_result)

        # Analysis
        print(f"\n📈 Batch Analysis")
        print("=" * 50)

        successful_runs = [r for r in results['configurations'] if r['success']]

        if successful_runs:
            # Sort by L2 error
            successful_runs.sort(key=lambda x: x['l2_error'])

            print(f"✅ Successful runs: {len(successful_runs)}/{total_runs}")
            print(f"❌ Failed runs: {results['summary']['failed_runs']}")

            # Top 5 configurations
            print(f"\n🏆 Top 5 Configurations:")
            print(f"{'Rank':<4} {'PopSize':<8} {'Mutation':<8} {'Crossover':<9} {'MaxIter':<8} {'L2 Error':<10} {'Time(s)':<8}")
            print("-" * 70)

            for rank, run in enumerate(successful_runs[:5], 1):
                config = run['configuration']
                print(f"{rank:<4} {config['popsize']:<8} {config['mutation']:<8} "
                      f"{config['crossover']:<9} {config['max_iter']:<8} "
                      f"{run['l2_error']:<10.6f} {run['time_seconds']:<8.1f}")

            # Parameter analysis
            print(f"\n📊 Parameter Analysis (Best 3 configs):")
            print(f"{'Rank':<4} {'Omega':<12} {'Alpha':<12} {'Beta':<12} {'Gamma':<12} {'Lambda':<12}")
            print("-" * 70)

            for rank, run in enumerate(successful_runs[:3], 1):
                params = run['calibrated_params']
                print(f"{rank:<4} {params['omega']:<12.6f} {params['alpha']:<12.6f} "
                      f"{params['beta']:<12.6f} {params['gamma']:<12.6f} {params['lambda']:<12.6f}")

            # Speed analysis
            times = [r['time_seconds'] for r in successful_runs]
            errors = [r['l2_error'] for r in successful_runs]

            print(f"\n⏱️  Timing Analysis:")
            print(f"   Fastest: {min(times):.1f}s")
            print(f"   Slowest: {max(times):.1f}s")
            print(f"   Average: {np.mean(times):.1f}s")

            print(f"\n🎯 Accuracy Analysis:")
            print(f"   Best error: {min(errors):.6f}")
            print(f"   Worst error: {max(errors):.6f}")
            print(f"   Average error: {np.mean(errors):.6f}")

            # Hyperparameter insights
            print(f"\n🔍 Hyperparameter Insights:")

            # Best popsize
            popsize_errors = {}
            for popsize in configurations['popsize']:
                runs_with_popsize = [r for r in successful_runs if r['configuration']['popsize'] == popsize]
                if runs_with_popsize:
                    avg_error = np.mean([r['l2_error'] for r in runs_with_popsize])
                    popsize_errors[popsize] = avg_error

            if popsize_errors:
                best_popsize = min(popsize_errors.items(), key=lambda x: x[1])
                print(f"   Best population size: {best_popsize[0]} (avg error: {best_popsize[1]:.6f})")

            # Best mutation
            mutation_errors = {}
            for mutation in configurations['mutation']:
                runs_with_mutation = [r for r in successful_runs if r['configuration']['mutation'] == mutation]
                if runs_with_mutation:
                    avg_error = np.mean([r['l2_error'] for r in runs_with_mutation])
                    mutation_errors[mutation] = avg_error

            if mutation_errors:
                best_mutation = min(mutation_errors.items(), key=lambda x: x[1])
                print(f"   Best mutation factor: {best_mutation[0]} (avg error: {best_mutation[1]:.6f})")

        else:
            print("❌ No successful runs!")

        # Save results
        output_file = 'de_batch_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Complete results saved to: {output_file}")

        # Create summary plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            if successful_runs:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

                # Error vs population size
                popsizes = [r['configuration']['popsize'] for r in successful_runs]
                errors = [r['l2_error'] for r in successful_runs]
                ax1.scatter(popsizes, errors, alpha=0.6)
                ax1.set_xlabel('Population Size')
                ax1.set_ylabel('L2 Error')
                ax1.set_title('Error vs Population Size')
                ax1.grid(True, alpha=0.3)

                # Error vs mutation
                mutations = [r['configuration']['mutation'] for r in successful_runs]
                ax2.scatter(mutations, errors, alpha=0.6, color='red')
                ax2.set_xlabel('Mutation Factor')
                ax2.set_ylabel('L2 Error')
                ax2.set_title('Error vs Mutation Factor')
                ax2.grid(True, alpha=0.3)

                # Time vs error
                times = [r['time_seconds'] for r in successful_runs]
                ax3.scatter(times, errors, alpha=0.6, color='green')
                ax3.set_xlabel('Time (seconds)')
                ax3.set_ylabel('L2 Error')
                ax3.set_title('Speed vs Accuracy Trade-off')
                ax3.grid(True, alpha=0.3)

                # Best convergence
                best_run = successful_runs[0]
                ax4.plot(best_run['convergence_history'])
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Objective Value')
                ax4.set_title(f'Best Convergence (Error: {best_run["l2_error"]:.6f})')
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('de_batch_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

                print(f"📊 Analysis plots saved to: de_batch_analysis.png")

        except ImportError:
            print("📊 Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")

        # Final summary
        if successful_runs:
            best_config = results['summary']['best_config']
            best_error = results['summary']['best_error']

            print(f"\n🎉 Batch Complete!")
            print(f"   Best configuration: {best_config}")
            print(f"   Best L2 error: {best_error:.6f}")
            print(f"   Success rate: {len(successful_runs)/total_runs*100:.1f}%")
        else:
            print(f"\n😞 Batch failed - no successful runs")

        return 0

    except Exception as e:
        print(f"\n❌ Batch failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(run_batch_calibration())
