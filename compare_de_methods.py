import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from calibrate2_de import calibrate_de
from calibrate_metade import calibrate_metade
from dataset2 import cal_dataset

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_de_methods():
    """Compare custom Differential Evolution vs MetaDE for GARCH calibration"""

    print("=" * 80)
    print("DIFFERENTIAL EVOLUTION METHODS COMPARISON")
    print("Custom DE Implementation vs MetaDE Library")
    print("=" * 80)

    try:
        # Load model and dataset
        print("Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print("Model loaded successfully.")

        print("Loading dataset...")
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                              "joint_dataset/assetprices.csv")
        print(f"Dataset: {len(dataset)} options, returns length {len(dataset.returns)}")

        # True parameter values for comparison
        true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        # Common parameters for fair comparison
        common_params = {
            'popsize': 100,
            'max_iter': 500,
            'seed': 42
        }

        print(f"True parameters: {dict(zip(param_names, true_vals))}")
        print(f"Common settings: {common_params}")

        # Method 1: Custom Differential Evolution
        print("\n" + "=" * 60)
        print("METHOD 1: CUSTOM DIFFERENTIAL EVOLUTION")
        print("=" * 60)

        custom_success = False
        custom_params = None
        custom_time = None
        custom_error = None
        custom_history = None

        try:
            start_time = time.time()
            custom_params, custom_history = calibrate_de(
                model=model,
                dataset=dataset,
                popsize=common_params['popsize'],
                max_iter=common_params['max_iter'],
                mutation=0.8,
                crossover=0.7,
                seed=common_params['seed']
            )
            custom_time = time.time() - start_time
            custom_error = np.linalg.norm(custom_params - true_vals, ord=2)
            custom_success = True

            print(f"✅ Custom DE completed successfully")
            print(f"⏱️  Time: {custom_time:.2f} seconds")
            print(f"📊 L2 Error: {custom_error:.6f}")
            print(f"📈 Final loss: {custom_history[-1]:.6f}")

        except Exception as e:
            print(f"❌ Custom DE failed: {e}")
            custom_time = float('inf')
            custom_error = float('inf')

        # Method 2: MetaDE
        print("\n" + "=" * 60)
        print("METHOD 2: METADE LIBRARY")
        print("=" * 60)

        metade_success = False
        metade_params = None
        metade_time = None
        metade_error = None
        metade_history = None

        try:
            start_time = time.time()
            metade_params, metade_history = calibrate_metade(
                model=model,
                dataset=dataset,
                popsize=common_params['popsize'],
                max_iter=common_params['max_iter'],
                strategy='best1bin',
                F=0.8,
                CR=0.7,
                seed=common_params['seed']
            )
            metade_time = time.time() - start_time
            metade_error = np.linalg.norm(metade_params - true_vals, ord=2)
            metade_success = True

            print(f"✅ MetaDE completed successfully")
            print(f"⏱️  Time: {metade_time:.2f} seconds")
            print(f"📊 L2 Error: {metade_error:.6f}")
            print(f"📈 Final loss: {metade_history[-1]:.6f}")

        except Exception as e:
            print(f"❌ MetaDE failed: {e}")
            metade_time = float('inf')
            metade_error = float('inf')

        # Comparison Summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        if custom_success and metade_success:
            # Parameter comparison
            print("\nParameter Comparison:")
            print(f"{'Parameter':<10} {'True':<12} {'Custom DE':<12} {'MetaDE':<12} {'Custom Err':<12} {'MetaDE Err':<12}")
            print("-" * 80)
            for i, name in enumerate(param_names):
                custom_err = abs(custom_params[i] - true_vals[i])
                metade_err = abs(metade_params[i] - true_vals[i])
                print(f"{name:<10} {true_vals[i]:<12.6f} {custom_params[i]:<12.6f} {metade_params[i]:<12.6f} "
                      f"{custom_err:<12.6f} {metade_err:<12.6f}")

            # Overall metrics
            print(f"\nOverall Metrics:")
            print(f"{'Metric':<25} {'Custom DE':<15} {'MetaDE':<15} {'Winner':<10}")
            print("-" * 70)
            print(f"{'L2 Error':<25} {custom_error:<15.6f} {metade_error:<15.6f} "
                  f"{'Custom' if custom_error < metade_error else 'MetaDE':<10}")
            print(f"{'Time (seconds)':<25} {custom_time:<15.2f} {metade_time:<15.2f} "
                  f"{'Custom' if custom_time < metade_time else 'MetaDE':<10}")
            print(f"{'Final Loss':<25} {custom_history[-1]:<15.6f} {metade_history[-1]:<15.6f} "
                  f"{'Custom' if custom_history[-1] < metade_history[-1] else 'MetaDE':<10}")

            # Calculate improvement percentages
            accuracy_improvement = abs(custom_error - metade_error) / max(custom_error, metade_error) * 100
            speed_ratio = max(custom_time, metade_time) / min(custom_time, metade_time)

            print(f"\nImprovement Analysis:")
            print(f"Accuracy improvement: {accuracy_improvement:.1f}%")
            print(f"Speed ratio: {speed_ratio:.1f}x")

            # Check parameter validation
            print(f"\nParameter Validation:")
            for method_name, params in [("Custom DE", custom_params), ("MetaDE", metade_params)]:
                alpha, beta = params[1], params[2]
                persistence = alpha + beta
                stationary = persistence < 1.0
                omega_positive = params[0] > 0
                print(f"{method_name:<10}: Persistence={persistence:.4f}, Stationary={stationary}, Omega>0={omega_positive}")

        # Save comparison results
        results = {
            'comparison_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'true_parameters': dict(zip(param_names, true_vals.tolist())),
            'common_settings': common_params,
            'custom_de': {
                'success': custom_success,
                'parameters': dict(zip(param_names, custom_params.tolist())) if custom_success else None,
                'time_seconds': custom_time if custom_success else None,
                'l2_error': custom_error if custom_success else None,
                'final_loss': custom_history[-1] if custom_success and custom_history else None,
                'convergence_history': custom_history if custom_success else None,
                'validation': {
                    'persistence': float(custom_params[1] + custom_params[2]) if custom_success else None,
                    'stationary': bool((custom_params[1] + custom_params[2]) < 1.0) if custom_success else None,
                    'omega_positive': bool(custom_params[0] > 0) if custom_success else None
                }
            },
            'metade': {
                'success': metade_success,
                'parameters': dict(zip(param_names, metade_params.tolist())) if metade_success else None,
                'time_seconds': metade_time if metade_success else None,
                'l2_error': metade_error if metade_success else None,
                'final_loss': metade_history[-1] if metade_success and metade_history else None,
                'convergence_history': metade_history if metade_success else None,
                'validation': {
                    'persistence': float(metade_params[1] + metade_params[2]) if metade_success else None,
                    'stationary': bool((metade_params[1] + metade_params[2]) < 1.0) if metade_success else None,
                    'omega_positive': bool(metade_params[0] > 0) if metade_success else None
                }
            }
        }

        if custom_success and metade_success:
            results['comparison'] = {
                'accuracy_winner': 'custom_de' if custom_error < metade_error else 'metade',
                'speed_winner': 'custom_de' if custom_time < metade_time else 'metade',
                'accuracy_improvement_percent': float(accuracy_improvement),
                'speed_ratio': float(speed_ratio),
                'custom_better_accuracy': custom_error < metade_error,
                'custom_better_speed': custom_time < metade_time
            }

        # Save results to file
        with open('de_methods_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Detailed results saved to de_methods_comparison.json")

        # Create visualization
        create_comparison_plots(results)

        return results

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_plots(results):
    """Create visualization plots for the comparison"""

    if not (results['custom_de']['success'] and results['metade']['success']):
        print("⚠️  Cannot create plots - one or both methods failed")
        return

    # Set up the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Differential Evolution Methods Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Parameter comparison
    param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
    true_vals = list(results['true_parameters'].values())
    custom_vals = list(results['custom_de']['parameters'].values())
    metade_vals = list(results['metade']['parameters'].values())

    x = np.arange(len(param_names))
    width = 0.25

    ax1.bar(x - width, true_vals, width, label='True', color='green', alpha=0.7)
    ax1.bar(x, custom_vals, width, label='Custom DE', color='blue', alpha=0.7)
    ax1.bar(x + width, metade_vals, width, label='MetaDE', color='red', alpha=0.7)

    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Parameter Values')
    ax1.set_title('Parameter Values Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale due to wide range of parameter values

    # Plot 2: Error comparison
    methods = ['Custom DE', 'MetaDE']
    errors = [results['custom_de']['l2_error'], results['metade']['l2_error']]
    colors = ['blue', 'red']

    bars = ax2.bar(methods, errors, color=colors, alpha=0.7)
    ax2.set_ylabel('L2 Error')
    ax2.set_title('Calibration Error Comparison')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.6f}', ha='center', va='bottom')

    # Plot 3: Time comparison
    times = [results['custom_de']['time_seconds'], results['metade']['time_seconds']]

    bars = ax3.bar(methods, times, color=colors, alpha=0.7)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computation Time Comparison')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom')

    # Plot 4: Convergence comparison
    custom_history = results['custom_de']['convergence_history']
    metade_history = results['metade']['convergence_history']

    if custom_history and metade_history:
        # Normalize lengths for comparison
        max_len = max(len(custom_history), len(metade_history))
        custom_x = np.linspace(0, max_len, len(custom_history))
        metade_x = np.linspace(0, max_len, len(metade_history))

        ax4.plot(custom_x, custom_history, 'b-', label='Custom DE', linewidth=2)
        ax4.plot(metade_x, metade_history, 'r-', label='MetaDE', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Best Fitness')
        ax4.set_title('Convergence History')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig('de_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison plots saved to de_methods_comparison.png")


def analyze_metade_advantages():
    """Analyze the theoretical advantages of MetaDE over custom implementation"""

    print("\n" + "=" * 80)
    print("METADE ADVANTAGES ANALYSIS")
    print("=" * 80)

    advantages = {
        'Implementation Quality': [
            'Professionally developed and tested library',
            'Extensive testing and validation',
            'Bug fixes and optimizations from community',
            'Follows established DE best practices'
        ],
        'Algorithm Features': [
            'Multiple DE strategies (best1bin, rand1bin, currenttobest1bin, etc.)',
            'Adaptive parameter control',
            'Better population initialization strategies',
            'Advanced constraint handling'
        ],
        'Performance': [
            'Optimized C/Fortran implementations under the hood',
            'Better numerical stability',
            'More efficient memory usage',
            'Vectorized operations'
        ],
        'Maintainability': [
            'Regular updates and improvements',
            'Community support and documentation',
            'Integration with scientific Python ecosystem',
            'Consistent API design'
        ],
        'Reliability': [
            'Extensive testing on various optimization problems',
            'Proven convergence properties',
            'Robust handling of edge cases',
            'Better numerical precision'
        ]
    }

    for category, items in advantages.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

    print(f"\nRecommendation: Switch to MetaDE for production use")


if __name__ == "__main__":
    results = compare_de_methods()
    analyze_metade_advantages()
