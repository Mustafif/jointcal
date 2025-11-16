import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from calibrate2_de import calibrate_de
from calibrate_scipy_de import calibrate_scipy_de
from dataset2 import cal_dataset

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_de_methods():
    """Compare custom Differential Evolution vs SciPy DE for GARCH calibration"""

    print("=" * 80)
    print("DIFFERENTIAL EVOLUTION METHODS COMPARISON")
    print("Custom DE Implementation vs SciPy DE")
    print("=" * 80)

    try:
        # Load model and dataset
        print("Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print("✅ Model loaded successfully.")

        print("Loading dataset...")
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                              "joint_dataset/assetprices.csv")
        print(f"✅ Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

        # True parameter values for comparison
        true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        # Common parameters for fair comparison
        common_params = {
            'popsize': 80,
            'maxiter': 300,
            'seed': 42
        }

        print(f"\nTrue parameters: {dict(zip(param_names, true_vals))}")
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
                max_iter=common_params['maxiter'],
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

        # Method 2: SciPy DE
        print("\n" + "=" * 60)
        print("METHOD 2: SCIPY DIFFERENTIAL EVOLUTION")
        print("=" * 60)

        scipy_success = False
        scipy_params = None
        scipy_time = None
        scipy_error = None
        scipy_history = None

        try:
            start_time = time.time()
            scipy_params, scipy_history = calibrate_scipy_de(
                model=model,
                dataset=dataset,
                popsize=20,  # SciPy uses multiplier (20 * 5 params = 100)
                maxiter=common_params['maxiter'],
                strategy='best1bin',
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=common_params['seed'],
                polish=False  # Disable for fair comparison
            )
            scipy_time = time.time() - start_time
            scipy_error = np.linalg.norm(scipy_params - true_vals, ord=2)
            scipy_success = True

            print(f"✅ SciPy DE completed successfully")
            print(f"⏱️  Time: {scipy_time:.2f} seconds")
            print(f"📊 L2 Error: {scipy_error:.6f}")
            print(f"📈 Final loss: {scipy_history[-1]:.6f}")

        except Exception as e:
            print(f"❌ SciPy DE failed: {e}")
            scipy_time = float('inf')
            scipy_error = float('inf')

        # Comparison Summary
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print("=" * 80)

        if custom_success and scipy_success:
            # Parameter comparison
            print(f"\nParameter Comparison:")
            print(f"{'Parameter':<10} {'True':<12} {'Custom DE':<12} {'SciPy DE':<12} {'Custom Err':<12} {'SciPy Err':<12}")
            print("-" * 80)
            for i, name in enumerate(param_names):
                custom_err = abs(custom_params[i] - true_vals[i])
                scipy_err = abs(scipy_params[i] - true_vals[i])
                print(f"{name:<10} {true_vals[i]:<12.6f} {custom_params[i]:<12.6f} {scipy_params[i]:<12.6f} "
                      f"{custom_err:<12.6f} {scipy_err:<12.6f}")

            # Overall metrics
            print(f"\nOverall Metrics:")
            print(f"{'Metric':<25} {'Custom DE':<15} {'SciPy DE':<15} {'Winner':<10}")
            print("-" * 70)
            print(f"{'L2 Error':<25} {custom_error:<15.6f} {scipy_error:<15.6f} "
                  f"{'Custom' if custom_error < scipy_error else 'SciPy':<10}")
            print(f"{'Time (seconds)':<25} {custom_time:<15.2f} {scipy_time:<15.2f} "
                  f"{'Custom' if custom_time < scipy_time else 'SciPy':<10}")
            print(f"{'Final Loss':<25} {custom_history[-1]:<15.6f} {scipy_history[-1]:<15.6f} "
                  f"{'Custom' if custom_history[-1] < scipy_history[-1] else 'SciPy':<10}")

            # Calculate improvement percentages
            accuracy_improvement = abs(custom_error - scipy_error) / max(custom_error, scipy_error) * 100
            speed_ratio = max(custom_time, scipy_time) / min(custom_time, scipy_time)

            print(f"\nImprovement Analysis:")
            print(f"Accuracy improvement: {accuracy_improvement:.1f}%")
            print(f"Speed ratio: {speed_ratio:.1f}x")

            # Check parameter validation
            print(f"\nParameter Validation:")
            for method_name, params in [("Custom DE", custom_params), ("SciPy DE", scipy_params)]:
                alpha, beta = params[1], params[2]
                persistence = alpha + beta
                stationary = persistence < 1.0
                omega_positive = params[0] > 0
                print(f"{method_name:<10}: Persistence={persistence:.4f}, Stationary={stationary}, Omega>0={omega_positive}")

            # Convergence analysis
            print(f"\nConvergence Analysis:")
            if len(custom_history) > 10 and len(scipy_history) > 10:
                custom_final_improvement = custom_history[0] - custom_history[-1]
                scipy_final_improvement = scipy_history[0] - scipy_history[-1]
                print(f"Custom DE improvement: {custom_final_improvement:.6f}")
                print(f"SciPy DE improvement: {scipy_final_improvement:.6f}")

        elif custom_success:
            print("✅ Only Custom DE completed successfully")
            print("❌ SciPy DE failed")
        elif scipy_success:
            print("❌ Custom DE failed")
            print("✅ Only SciPy DE completed successfully")
        else:
            print("❌ Both methods failed")

        # Save comparison results
        results = {
            'comparison_timestamp': datetime.now().isoformat(),
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
            'scipy_de': {
                'success': scipy_success,
                'parameters': dict(zip(param_names, scipy_params.tolist())) if scipy_success else None,
                'time_seconds': scipy_time if scipy_success else None,
                'l2_error': scipy_error if scipy_success else None,
                'final_loss': scipy_history[-1] if scipy_success and scipy_history else None,
                'convergence_history': scipy_history if scipy_success else None,
                'validation': {
                    'persistence': float(scipy_params[1] + scipy_params[2]) if scipy_success else None,
                    'stationary': bool((scipy_params[1] + scipy_params[2]) < 1.0) if scipy_success else None,
                    'omega_positive': bool(scipy_params[0] > 0) if scipy_success else None
                }
            }
        }

        if custom_success and scipy_success:
            results['comparison'] = {
                'accuracy_winner': 'custom_de' if custom_error < scipy_error else 'scipy_de',
                'speed_winner': 'custom_de' if custom_time < scipy_time else 'scipy_de',
                'accuracy_improvement_percent': float(accuracy_improvement),
                'speed_ratio': float(speed_ratio),
                'custom_better_accuracy': custom_error < scipy_error,
                'custom_better_speed': custom_time < scipy_time
            }

        # Save results to file
        with open('de_methods_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Detailed results saved to de_methods_comparison.json")

        # Create visualization
        create_comparison_plots(results)

        # Recommendations
        print(f"\n💡 Recommendations:")
        if custom_success and scipy_success:
            if scipy_error < custom_error and scipy_time < custom_time:
                print("🏆 Use SciPy DE - better accuracy AND faster")
            elif scipy_error < custom_error:
                print("🎯 Use SciPy DE - better accuracy")
            elif scipy_time < custom_time:
                print("⚡ Use SciPy DE - faster execution")
            else:
                print("🔧 Custom DE has advantages but SciPy DE is more mature")
        elif scipy_success:
            print("✅ Use SciPy DE - only working method")
        elif custom_success:
            print("🔧 Use Custom DE - only working method")
        else:
            print("❌ Neither method worked - check implementation")

        print("📚 SciPy DE benefits: mature, tested, optimized, well-documented")
        print("🔧 Custom DE benefits: full control, customizable, research flexibility")

        return results

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_plots(results):
    """Create visualization plots for the comparison"""

    if not (results['custom_de']['success'] and results['scipy_de']['success']):
        print("⚠️  Cannot create plots - one or both methods failed")
        return

    # Set up the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Custom DE vs SciPy DE Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Parameter comparison
    param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
    true_vals = list(results['true_parameters'].values())
    custom_vals = list(results['custom_de']['parameters'].values())
    scipy_vals = list(results['scipy_de']['parameters'].values())

    x = np.arange(len(param_names))
    width = 0.25

    ax1.bar(x - width, true_vals, width, label='True', color='green', alpha=0.7)
    ax1.bar(x, custom_vals, width, label='Custom DE', color='blue', alpha=0.7)
    ax1.bar(x + width, scipy_vals, width, label='SciPy DE', color='orange', alpha=0.7)

    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Parameter Values')
    ax1.set_title('Parameter Values Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale due to wide range of parameter values

    # Plot 2: Error comparison
    methods = ['Custom DE', 'SciPy DE']
    errors = [results['custom_de']['l2_error'], results['scipy_de']['l2_error']]
    colors = ['blue', 'orange']

    bars = ax2.bar(methods, errors, color=colors, alpha=0.7)
    ax2.set_ylabel('L2 Error')
    ax2.set_title('Calibration Error Comparison')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.4f}', ha='center', va='bottom')

    # Plot 3: Time comparison
    times = [results['custom_de']['time_seconds'], results['scipy_de']['time_seconds']]

    bars = ax3.bar(methods, times, color=colors, alpha=0.7)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computation Time Comparison')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom')

    # Plot 4: Convergence comparison
    custom_history = results['custom_de']['convergence_history']
    scipy_history = results['scipy_de']['convergence_history']

    if custom_history and scipy_history:
        # Normalize lengths for comparison
        max_len = max(len(custom_history), len(scipy_history))
        custom_x = np.linspace(0, max_len, len(custom_history))
        scipy_x = np.linspace(0, max_len, len(scipy_history))

        ax4.plot(custom_x, custom_history, 'b-', label='Custom DE', linewidth=2)
        ax4.plot(scipy_x, scipy_history, 'orange', linestyle='-', label='SciPy DE', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Best Fitness')
        ax4.set_title('Convergence History')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig('custom_vs_scipy_de_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison plots saved to custom_vs_scipy_de_comparison.png")


def analyze_scipy_advantages():
    """Analyze the advantages of SciPy DE over custom implementation"""

    print("\n" + "=" * 80)
    print("SCIPY DE ADVANTAGES ANALYSIS")
    print("=" * 80)

    advantages = {
        'Implementation Quality': [
            'Mature, battle-tested implementation',
            'Extensive peer review and validation',
            'Optimized C/Fortran backend for performance',
            'Comprehensive testing across diverse problems'
        ],
        'Algorithm Features': [
            'Multiple DE strategies (best1bin, best1exp, rand1exp, etc.)',
            'Adaptive parameter control mechanisms',
            'Built-in constraint handling',
            'Optional local optimization polishing'
        ],
        'Robustness': [
            'Robust numerical implementation',
            'Proper handling of edge cases',
            'Convergence criteria and tolerances',
            'Automatic parameter validation'
        ],
        'Usability': [
            'Clean, well-documented API',
            'Extensive documentation and examples',
            'Integration with SciPy ecosystem',
            'Standard scientific Python interface'
        ],
        'Maintenance': [
            'Regular updates and bug fixes',
            'Large community support',
            'Long-term stability and support',
            'No custom code maintenance burden'
        ]
    }

    for category, items in advantages.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

    print(f"\n🎯 Recommendation: Migrate to SciPy DE for production use")
    print(f"📚 SciPy DE provides professional-grade optimization with minimal maintenance")

    # Performance characteristics
    print(f"\n📈 Performance Characteristics:")
    print(f"  • Population size: Automatically optimized")
    print(f"  • Memory usage: Efficient C backend")
    print(f"  • Convergence: Multiple stopping criteria")
    print(f"  • Reliability: Extensively tested")


if __name__ == "__main__":
    results = compare_de_methods()
    if results:
        analyze_scipy_advantages()

    print(f"\n🎉 Comparison completed!")
    print(f"📁 Results saved to: de_methods_comparison.json")
    print(f"📊 Plots saved to: custom_vs_scipy_de_comparison.png")
