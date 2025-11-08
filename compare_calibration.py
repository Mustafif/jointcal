import torch
import numpy as np
import time
import json
from dataset2 import cal_dataset
from calibrate2 import calibrate as calibrate_gradient
from calibrate2_de import calibrate_de
import matplotlib.pyplot as plt

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_calibration_methods():
    """Compare gradient-based optimization vs differential evolution for GARCH calibration"""

    print("=" * 80)
    print("CALIBRATION METHOD COMPARISON")
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

        results = {}

        # Method 1: Gradient-based optimization (Adam)
        print("\n" + "=" * 50)
        print("METHOD 1: GRADIENT-BASED OPTIMIZATION (ADAM)")
        print("=" * 50)

        start_time = time.time()
        try:
            gradient_params = calibrate_gradient(
                model, dataset,
                lr=5e-4,
                steps=500,
                batch_size=64,
                device=device
            )
            gradient_time = time.time() - start_time
            gradient_success = True
            gradient_error = np.linalg.norm(gradient_params - true_vals, ord=2)

            print(f"✅ Gradient method completed in {gradient_time:.2f} seconds")
            print(f"Two-norm error: {gradient_error:.6f}")

        except Exception as e:
            print(f"❌ Gradient method failed: {e}")
            gradient_params = None
            gradient_time = None
            gradient_success = False
            gradient_error = None

        # Method 2: Differential Evolution
        print("\n" + "=" * 50)
        print("METHOD 2: DIFFERENTIAL EVOLUTION")
        print("=" * 50)

        start_time = time.time()
        try:
            de_params, de_history = calibrate_de(
                model, dataset,
                popsize=50,
                max_iter=500,
                mutation=0.8,
                crossover=0.7,
                seed=42,
                device=device
            )
            de_time = time.time() - start_time
            de_success = True
            de_error = np.linalg.norm(de_params - true_vals, ord=2)

            print(f"✅ DE method completed in {de_time:.2f} seconds")
            print(f"Two-norm error: {de_error:.6f}")

        except Exception as e:
            print(f"❌ DE method failed: {e}")
            de_params = None
            de_time = None
            de_success = False
            de_error = None
            de_history = None

        # Results comparison
        print("\n" + "=" * 80)
        print("RESULTS COMPARISON")
        print("=" * 80)

        if gradient_success and de_success:
            # Parameter comparison table
            print(f"{'Parameter':<10} {'True Value':<12} {'Gradient':<12} {'DE':<12} {'Grad Error':<12} {'DE Error':<12}")
            print("-" * 80)

            for i, name in enumerate(param_names):
                true_val = true_vals[i]
                grad_val = gradient_params[i] if gradient_success else np.nan
                de_val = de_params[i] if de_success else np.nan
                grad_err = abs(grad_val - true_val) if gradient_success else np.nan
                de_err = abs(de_val - true_val) if de_success else np.nan

                print(f"{name:<10} {true_val:<12.6f} {grad_val:<12.6f} {de_val:<12.6f} {grad_err:<12.6f} {de_err:<12.6f}")

            print(f"\n{'Metric':<20} {'Gradient':<15} {'DE':<15} {'Winner':<10}")
            print("-" * 60)
            print(f"{'Total Error (L2)':<20} {gradient_error:<15.6f} {de_error:<15.6f} {'Gradient' if gradient_error < de_error else 'DE':<10}")
            print(f"{'Time (seconds)':<20} {gradient_time:<15.2f} {de_time:<15.2f} {'Gradient' if gradient_time < de_time else 'DE':<10}")

            # Stationarity check
            def check_stationarity(params):
                if params is not None:
                    alpha, beta = params[1], params[2]
                    return alpha + beta < 1.0
                return False

            grad_stationary = check_stationarity(gradient_params)
            de_stationary = check_stationarity(de_params)

            print(f"{'Stationarity':<20} {'✅' if grad_stationary else '❌':<15} {'✅' if de_stationary else '❌':<15}")

        # Save detailed results
        results = {
            'true_parameters': dict(zip(param_names, true_vals.tolist())),
            'gradient_method': {
                'success': gradient_success,
                'parameters': dict(zip(param_names, gradient_params.tolist())) if gradient_success else None,
                'time_seconds': gradient_time,
                'l2_error': gradient_error,
                'stationary': check_stationarity(gradient_params) if gradient_success else None
            },
            'differential_evolution': {
                'success': de_success,
                'parameters': dict(zip(param_names, de_params.tolist())) if de_success else None,
                'time_seconds': de_time,
                'l2_error': de_error,
                'convergence_history': de_history,
                'stationary': check_stationarity(de_params) if de_success else None
            }
        }

        # Add winner analysis
        if gradient_success and de_success:
            results['comparison'] = {
                'accuracy_winner': 'gradient' if gradient_error < de_error else 'differential_evolution',
                'speed_winner': 'gradient' if gradient_time < de_time else 'differential_evolution',
                'accuracy_improvement': abs(gradient_error - de_error) / max(gradient_error, de_error),
                'speed_ratio': max(gradient_time, de_time) / min(gradient_time, de_time)
            }

        # Save results
        with open('calibration_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Detailed results saved to calibration_comparison.json")

        # Generate comparison plots
        try:
            create_comparison_plots(results, gradient_params, de_params, true_vals, param_names, de_history)
        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


def create_comparison_plots(results, gradient_params, de_params, true_vals, param_names, de_history):
    """Create comparison plots for the calibration methods"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Parameter comparison bar chart
    if gradient_params is not None and de_params is not None:
        x = np.arange(len(param_names))
        width = 0.25

        ax1.bar(x - width, true_vals, width, label='True', alpha=0.8, color='green')
        ax1.bar(x, gradient_params, width, label='Gradient', alpha=0.8, color='blue')
        ax1.bar(x + width, de_params, width, label='DE', alpha=0.8, color='red')

        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Values')
        ax1.set_title('Parameter Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(param_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    if results['gradient_method']['success'] and results['differential_evolution']['success']:
        methods = ['Gradient', 'DE']
        errors = [results['gradient_method']['l2_error'], results['differential_evolution']['l2_error']]
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
    if results['gradient_method']['success'] and results['differential_evolution']['success']:
        times = [results['gradient_method']['time_seconds'], results['differential_evolution']['time_seconds']]

        bars = ax3.bar(methods, times, color=colors, alpha=0.7)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Computation Time Comparison')
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f}s', ha='center', va='bottom')

    # Plot 4: DE convergence history
    if de_history is not None:
        ax4.plot(de_history, 'r-', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Objective Function Value')
        ax4.set_title('DE Convergence History')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison plots saved to calibration_comparison.png")

    # Additional detailed parameter error plot
    if gradient_params is not None and de_params is not None:
        fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

        grad_errors = np.abs(gradient_params - true_vals)
        de_errors = np.abs(de_params - true_vals)

        x = np.arange(len(param_names))
        width = 0.35

        ax.bar(x - width/2, grad_errors, width, label='Gradient Error', alpha=0.8, color='blue')
        ax.bar(x + width/2, de_errors, width, label='DE Error', alpha=0.8, color='red')

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Parameter-wise Absolute Errors')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('parameter_errors.png', dpi=300, bbox_inches='tight')
        print("📊 Parameter error plot saved to parameter_errors.png")


if __name__ == "__main__":
    compare_calibration_methods()
