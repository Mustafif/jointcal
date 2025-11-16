import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from calibrate2_de import calibrate_de
from calibrate_gpu_de import calibrate_gpu_de
from calibrate_scipy_de import calibrate_scipy_de
from dataset2 import cal_dataset

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"


def get_system_info():
    """Get system information for benchmarking"""
    info = {
        'cpu': {
            'model': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
        },
        'memory': {
            'total_gb': psutil.virtual_memory().total / 1e9,
            'available_gb': psutil.virtual_memory().available / 1e9
        },
        'gpu': {}
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            info['gpu'][f'gpu_{i}'] = {
                'name': gpu_props.name,
                'memory_gb': gpu_props.total_memory / 1e9,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            }
    else:
        info['gpu']['available'] = False

    return info


def benchmark_method(method_name, calibration_func, *args, **kwargs):
    """Benchmark a calibration method"""
    print(f"\n{'='*60}")
    print(f"🔬 Benchmarking: {method_name}")
    print(f"{'='*60}")

    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1e9

    initial_cpu_memory = psutil.Process().memory_info().rss / 1e9

    # Run calibration
    start_time = time.time()
    try:
        result = calibration_func(*args, **kwargs)
        params, history = result if isinstance(result, tuple) else (result, [])
        success = True
        error_msg = None
    except Exception as e:
        params, history = None, []
        success = False
        error_msg = str(e)
        print(f"❌ {method_name} failed: {e}")

    end_time = time.time()

    # Memory after
    final_cpu_memory = psutil.Process().memory_info().rss / 1e9

    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated() / 1e9
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1e9
    else:
        final_gpu_memory = 0
        peak_gpu_memory = 0

    # Calculate metrics
    execution_time = end_time - start_time
    cpu_memory_used = final_cpu_memory - initial_cpu_memory
    gpu_memory_used = final_gpu_memory - initial_gpu_memory

    # Accuracy metrics
    if success and params is not None:
        true_vals = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        l2_error = np.linalg.norm(params - true_vals)
        final_loss = history[-1] if history else float('inf')

        # Parameter validation
        alpha, beta = params[1], params[2]
        persistence = alpha + beta
        stationary = persistence < 1.0
        omega_positive = params[0] > 0
    else:
        l2_error = float('inf')
        final_loss = float('inf')
        stationary = False
        omega_positive = False
        persistence = float('inf')

    benchmark_result = {
        'method': method_name,
        'success': success,
        'error_message': error_msg,
        'performance': {
            'execution_time_seconds': execution_time,
            'cpu_memory_used_gb': cpu_memory_used,
            'gpu_memory_used_gb': gpu_memory_used,
            'peak_gpu_memory_gb': peak_gpu_memory
        },
        'accuracy': {
            'l2_error': l2_error,
            'final_loss': final_loss,
            'calibrated_params': params.tolist() if params is not None else None
        },
        'validation': {
            'persistence': persistence,
            'stationary': stationary,
            'omega_positive': omega_positive
        }
    }

    # Print summary
    if success:
        print(f"✅ Success in {execution_time:.2f}s")
        print(f"📊 L2 Error: {l2_error:.6f}")
        print(f"🎯 Final Loss: {final_loss:.6f}")
        print(f"💾 CPU Memory: {cpu_memory_used:.2f} GB")
        if torch.cuda.is_available():
            print(f"🚀 GPU Memory: {gpu_memory_used:.2f} GB (Peak: {peak_gpu_memory:.2f} GB)")
        print(f"✅ Parameters valid: Stationary={stationary}, Ω>0={omega_positive}")
    else:
        print(f"❌ Failed after {execution_time:.2f}s")

    return benchmark_result


def run_comprehensive_benchmark():
    """Run comprehensive benchmark of all DE methods"""
    print("🏁 Comprehensive DE Methods Benchmark")
    print("=" * 80)

    # System info
    system_info = get_system_info()
    print(f"🖥️  System Information:")
    print(f"   CPU: {system_info['cpu']['cores_logical']} cores")
    print(f"   RAM: {system_info['memory']['total_gb']:.1f} GB")

    if torch.cuda.is_available():
        for gpu_name, gpu_info in system_info['gpu'].items():
            if gpu_name != 'available':
                print(f"   GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
    else:
        print("   GPU: Not available")

    # Load model and dataset
    print(f"\n📂 Loading model and dataset...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        dataset = cal_dataset("joint_dataset/scalable_hn_dataset_250x60.csv",
                              "joint_dataset/assetprices.csv")
        print(f"✅ Loaded: {len(dataset)} options, {len(dataset.returns)} returns")
    except Exception as e:
        print(f"❌ Failed to load model/dataset: {e}")
        return None

    # Benchmark parameters
    common_params = {
        'popsize': 50,
        'maxiter': 200,  # Reduced for faster benchmarking
        'seed': 42
    }

    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'common_params': common_params,
        'methods': {}
    }

    # Method 1: Custom DE
    results['methods']['custom_de'] = benchmark_method(
        'Custom DE (PyTorch)',
        calibrate_de,
        model, dataset,
        popsize=common_params['popsize'],
        max_iter=common_params['maxiter'],
        mutation=0.8,
        crossover=0.7,
        seed=common_params['seed']
    )

    # Method 2: SciPy DE
    results['methods']['scipy_de'] = benchmark_method(
        'SciPy DE (CPU)',
        calibrate_scipy_de,
        model, dataset,
        popsize=15,  # SciPy uses multiplier
        maxiter=common_params['maxiter'],
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=common_params['seed'],
        polish=False  # Disable for fair comparison
    )

    # Method 3: GPU DE (if available)
    if torch.cuda.is_available():
        results['methods']['gpu_de'] = benchmark_method(
            'GPU DE (CUDA)',
            calibrate_gpu_de,
            model, dataset,
            popsize=common_params['popsize'],
            maxiter=common_params['maxiter'],
            strategy='best1bin',
            mutation=(0.5, 1.0),
            crossover=0.7,
            seed=common_params['seed']
        )
    else:
        print("⚠️ Skipping GPU DE - CUDA not available")

    # Analysis and comparison
    print(f"\n{'='*80}")
    print("📊 BENCHMARK ANALYSIS")
    print("=" * 80)

    successful_methods = {k: v for k, v in results['methods'].items() if v['success']}

    if not successful_methods:
        print("❌ No methods completed successfully")
        return results

    # Performance comparison
    print(f"\n⏱️  Performance Comparison:")
    print(f"{'Method':<20} {'Time (s)':<12} {'CPU Mem (GB)':<15} {'GPU Mem (GB)':<15} {'L2 Error':<12}")
    print("-" * 80)

    fastest_time = min(m['performance']['execution_time_seconds'] for m in successful_methods.values())
    most_accurate = min(m['accuracy']['l2_error'] for m in successful_methods.values())

    for method_name, result in successful_methods.items():
        perf = result['performance']
        acc = result['accuracy']

        time_marker = "🏆" if perf['execution_time_seconds'] == fastest_time else "  "
        acc_marker = "🎯" if acc['l2_error'] == most_accurate else "  "

        print(f"{method_name:<20} {perf['execution_time_seconds']:<12.2f} "
              f"{perf['cpu_memory_used_gb']:<15.2f} {perf['gpu_memory_used_gb']:<15.2f} "
              f"{acc['l2_error']:<12.6f} {time_marker}{acc_marker}")

    # Speed comparison
    if len(successful_methods) > 1:
        print(f"\n🚀 Speed Comparison:")
        times = {k: v['performance']['execution_time_seconds'] for k, v in successful_methods.items()}
        sorted_times = sorted(times.items(), key=lambda x: x[1])

        for i, (method, time_val) in enumerate(sorted_times):
            if i == 0:
                print(f"   1. {method}: {time_val:.2f}s (baseline)")
            else:
                speedup = sorted_times[0][1] / time_val
                print(f"   {i+1}. {method}: {time_val:.2f}s ({speedup:.1f}x slower)")

    # Accuracy comparison
    print(f"\n🎯 Accuracy Comparison:")
    accuracies = {k: v['accuracy']['l2_error'] for k, v in successful_methods.items()}
    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1])

    for i, (method, error) in enumerate(sorted_accuracies):
        if i == 0:
            print(f"   1. {method}: {error:.6f} (most accurate)")
        else:
            relative_error = error / sorted_accuracies[0][1]
            print(f"   {i+1}. {method}: {error:.6f} ({relative_error:.1f}x worse)")

    # Resource usage
    print(f"\n💾 Resource Usage:")
    for method_name, result in successful_methods.items():
        perf = result['performance']
        print(f"   {method_name}:")
        print(f"     CPU Memory: {perf['cpu_memory_used_gb']:.2f} GB")
        if torch.cuda.is_available() and perf['gpu_memory_used_gb'] > 0:
            print(f"     GPU Memory: {perf['gpu_memory_used_gb']:.2f} GB (Peak: {perf['peak_gpu_memory_gb']:.2f} GB)")

    # Recommendations
    print(f"\n💡 Recommendations:")

    if torch.cuda.is_available() and 'gpu_de' in successful_methods:
        gpu_time = successful_methods['gpu_de']['performance']['execution_time_seconds']
        cpu_methods = {k: v for k, v in successful_methods.items() if k != 'gpu_de'}

        if cpu_methods:
            fastest_cpu_time = min(m['performance']['execution_time_seconds'] for m in cpu_methods.values())
            if gpu_time < fastest_cpu_time:
                print("   🚀 Use GPU DE for best performance")
            else:
                print("   💻 GPU DE not significantly faster - CPU methods may be sufficient")

        gpu_accuracy = successful_methods['gpu_de']['accuracy']['l2_error']
        if gpu_accuracy == most_accurate:
            print("   🎯 GPU DE also provides best accuracy")
    else:
        print("   💻 Use SciPy DE for CPU-only systems (most reliable)")
        print("   🔧 Custom DE provides more control but may be less optimized")

    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Detailed results saved to benchmark_results.json")

    # Create visualization
    create_benchmark_plots(results)

    return results


def create_benchmark_plots(results):
    """Create visualization plots for benchmark results"""
    successful_methods = {k: v for k, v in results['methods'].items() if v['success']}

    if len(successful_methods) < 2:
        print("⚠️ Need at least 2 successful methods for plotting")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DE Methods Benchmark Results', fontsize=16, fontweight='bold')

    methods = list(successful_methods.keys())

    # Plot 1: Execution time
    times = [successful_methods[m]['performance']['execution_time_seconds'] for m in methods]
    colors = ['blue', 'orange', 'green', 'red'][:len(methods)]

    bars1 = ax1.bar(methods, times, color=colors, alpha=0.7)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Performance Comparison')
    ax1.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom')

    # Plot 2: Accuracy
    errors = [successful_methods[m]['accuracy']['l2_error'] for m in methods]

    bars2 = ax2.bar(methods, errors, color=colors, alpha=0.7)
    ax2.set_ylabel('L2 Error')
    ax2.set_title('Accuracy Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')

    for bar, error in zip(bars2, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.2e}', ha='center', va='bottom')

    # Plot 3: Memory usage
    cpu_memory = [successful_methods[m]['performance']['cpu_memory_used_gb'] for m in methods]
    gpu_memory = [successful_methods[m]['performance']['gpu_memory_used_gb'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    ax3.bar(x - width/2, cpu_memory, width, label='CPU Memory', alpha=0.7)
    ax3.bar(x + width/2, gpu_memory, width, label='GPU Memory', alpha=0.7)
    ax3.set_ylabel('Memory Usage (GB)')
    ax3.set_title('Memory Usage Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45)
    ax3.legend()

    # Plot 4: Speed vs Accuracy scatter
    ax4.scatter(times, errors, c=colors[:len(methods)], s=100, alpha=0.7)
    ax4.set_xlabel('Execution Time (seconds)')
    ax4.set_ylabel('L2 Error')
    ax4.set_title('Speed vs Accuracy Trade-off')
    ax4.set_yscale('log')

    for i, method in enumerate(methods):
        ax4.annotate(method, (times[i], errors[i]),
                    xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Benchmark plots saved to benchmark_comparison.png")


if __name__ == "__main__":
    try:
        results = run_comprehensive_benchmark()
        print("\n🎉 Benchmark completed successfully!")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
