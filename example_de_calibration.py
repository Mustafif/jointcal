import torch
import numpy as np
from dataset2 import cal_dataset
from calibrate2_de import DifferentialEvolution, calibrate_de
from cal_loss import Calibration_Loss
import json

def simple_de_example():
    """Simple example of using differential evolution for GARCH calibration"""

    print("🔬 Differential Evolution GARCH Calibration Example")
    print("=" * 60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model path (adjust as needed)
    MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"

    try:
        # Load pre-trained model
        print("\n📦 Loading model...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        print("✅ Model loaded successfully")

        # Load calibration dataset
        print("\n📊 Loading dataset...")
        dataset = cal_dataset(
            "joint_dataset/scalable_hn_dataset_250x60.csv",
            "joint_dataset/assetprices.csv"
        )
        print(f"✅ Dataset loaded: {len(dataset)} options, {len(dataset.returns)} returns")

        # Run DE calibration with different configurations
        configurations = [
            {"popsize": 30, "max_iter": 200, "mutation": 0.5, "crossover": 0.5},
            {"popsize": 50, "max_iter": 300, "mutation": 0.8, "crossover": 0.7},
            {"popsize": 100, "max_iter": 500, "mutation": 1.0, "crossover": 0.9}
        ]

        results = {}
        true_params = np.array([1e-6, 1.33e-6, 0.8, 5, 0.2])
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']

        for i, config in enumerate(configurations):
            print(f"\n🚀 Configuration {i+1}: {config}")
            print("-" * 40)

            try:
                # Run calibration
                calibrated_params, history = calibrate_de(
                    model=model,
                    dataset=dataset,
                    seed=42 + i,  # Different seed for each run
                    device=device,
                    **config
                )

                # Calculate error
                error = np.linalg.norm(calibrated_params - true_params, ord=2)

                # Store results
                results[f"config_{i+1}"] = {
                    "configuration": config,
                    "parameters": dict(zip(param_names, calibrated_params.tolist())),
                    "l2_error": error,
                    "convergence_history": history,
                    "final_loss": history[-1] if history else None
                }

                print(f"✅ Configuration {i+1} completed")
                print(f"   L2 Error: {error:.6f}")
                print(f"   Final Loss: {history[-1]:.6f}" if history else "   No history available")

                # Check parameter constraints
                alpha, beta = calibrated_params[1], calibrated_params[2]
                stationarity = alpha + beta < 1.0
                omega_positive = calibrated_params[0] > 0

                print(f"   Stationarity (α+β<1): {'✅' if stationarity else '❌'} ({alpha+beta:.4f})")
                print(f"   Omega positive: {'✅' if omega_positive else '❌'} ({calibrated_params[0]:.8f})")

            except Exception as e:
                print(f"❌ Configuration {i+1} failed: {e}")
                results[f"config_{i+1}"] = {"error": str(e)}

        # Find best configuration
        successful_configs = {k: v for k, v in results.items() if "error" not in v}

        if successful_configs:
            best_config = min(successful_configs.items(), key=lambda x: x[1]["l2_error"])
            print(f"\n🏆 Best Configuration: {best_config[0]}")
            print(f"   L2 Error: {best_config[1]['l2_error']:.6f}")
            print(f"   Parameters: {best_config[1]['parameters']}")

            # Save all results
            with open('de_calibration_example_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("\n📁 Results saved to 'de_calibration_example_results.json'")

        else:
            print("\n❌ No configurations succeeded")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Please ensure the model and dataset files exist in the correct paths")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


def custom_de_example():
    """Example showing how to use the DifferentialEvolution class directly"""

    print("\n🛠️  Custom DE Implementation Example")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a simple test function (Rosenbrock function)
    def rosenbrock(x):
        """Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2"""
        a, b = 1.0, 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

    # Set up bounds for 2D Rosenbrock
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]  # x and y bounds

    # Initialize DE optimizer
    de = DifferentialEvolution(
        bounds=bounds,
        popsize=30,
        mutation=(0.5, 1.0),  # Use mutation range for better exploration
        crossover=0.7,
        seed=42,
        device=device
    )

    print("🎯 Optimizing 2D Rosenbrock function")
    print("   Global minimum: f(1,1) = 0")

    # Run optimization
    best_params, best_fitness, history = de.optimize(
        objective_func=rosenbrock,
        max_iter=100,
        tolerance=1e-6,
        verbose=True
    )

    print(f"\n✅ Optimization complete")
    print(f"   Best parameters: [{best_params[0]:.6f}, {best_params[1]:.6f}]")
    print(f"   Best fitness: {best_fitness:.6f}")
    print(f"   Expected: [1.000000, 1.000000] with fitness 0.000000")

    error = torch.norm(best_params - torch.tensor([1.0, 1.0], device=device))
    print(f"   Error from global optimum: {error:.6f}")


def hyperparameter_tuning_example():
    """Example of hyperparameter tuning for DE calibration"""

    print("\n⚙️  Hyperparameter Tuning Example")
    print("=" * 45)

    # Define hyperparameter grid
    mutation_values = [0.5, 0.8, 1.2, (0.5, 1.0), (0.6, 1.2)]  # Include ranges
    crossover_values = [0.3, 0.7, 0.9]
    popsize_values = [20, 50, 100]

    print("Grid search over:")
    print(f"  Mutation: {mutation_values}")
    print(f"  Crossover: {crossover_values}")
    print(f"  Population size: {popsize_values}")

    # Simple test function for demonstration
    def sphere_function(x):
        """Simple sphere function for testing"""
        return torch.sum(x**2)

    bounds = [(-5.0, 5.0)] * 5  # 5D sphere function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_config = None
    best_result = float('inf')

    total_combinations = len(mutation_values) * len(crossover_values) * len(popsize_values)
    current_combination = 0

    for mutation in mutation_values:
        for crossover in crossover_values:
            for popsize in popsize_values:
                current_combination += 1

                print(f"\n🔄 Testing {current_combination}/{total_combinations}: "
                      f"mutation={mutation}, crossover={crossover}, popsize={popsize}")

                try:
                    de = DifferentialEvolution(
                        bounds=bounds,
                        popsize=popsize,
                        mutation=mutation,
                        crossover=crossover,
                        seed=42,
                        device=device
                    )

                    best_params, best_fitness, _ = de.optimize(
                        objective_func=sphere_function,
                        max_iter=50,  # Short runs for grid search
                        verbose=False
                    )

                    if best_fitness < best_result:
                        best_result = best_fitness
                        best_config = {
                            'mutation': mutation,
                            'crossover': crossover,
                            'popsize': popsize,
                            'result': best_fitness
                        }

                    print(f"   Result: {best_fitness:.6f}")

                except Exception as e:
                    print(f"   Failed: {e}")

    if best_config:
        print(f"\n🏆 Best hyperparameters:")
        print(f"   Mutation: {best_config['mutation']}")
        print(f"   Crossover: {best_config['crossover']}")
        print(f"   Population size: {best_config['popsize']}")
        print(f"   Best result: {best_config['result']:.6f}")
    else:
        print("\n❌ No successful configurations found")


if __name__ == "__main__":
    print("🧬 Differential Evolution Examples for GARCH Calibration")
    print("=" * 70)

    # Run examples
    simple_de_example()
    custom_de_example()
    hyperparameter_tuning_example()

    print("\n🎉 All examples completed!")
