import json

import numpy as np
from numpy import mean, var
from scipy.stats import kurtosis, skew

r = 0.05
M = 10
N = 512
TN = [10, 50, 100, 252, 365, 512]
dt = 1 / N
Z = np.random.normal(0, 1, (N+1, M))

def mc(omega, alpha, beta, gamma, lambda_):
    # Ensure all parameters are scalar floats
    omega = float(omega)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    lambda_ = float(lambda_)

    num_point = N+1
    Rt = np.zeros((num_point, M))
    ht = np.zeros((num_point, M))

    # Calculate initial variance - ensure scalar operations
    initial_var = (omega + alpha)/(1.0 - beta - alpha * gamma**2)
    ht[0] = initial_var
    Rt[0] = 0

    for i in range(1, num_point):
        ht[i] = omega + beta*ht[i-1] + alpha*(Z[i-1] - gamma*np.sqrt(ht[i-1]))**2
        Rt[i] = r + lambda_*ht[i] + np.sqrt(ht[i])*Z[i]
    return Rt

def compute_moments(Rt, TN):
    """Compute four moments for given time points"""
    moments = {}
    for t in TN:
        Rt_t = Rt[:t+1].flatten()  # Ensure 1D array for moment calculations
        moments[str(t)] = {  # Convert t to string for JSON compatibility
            'mean': float(np.mean(Rt_t)),
            'variance': float(np.var(Rt_t)),
            'skewness': float(skew(Rt_t)),
            'kurtosis': float(kurtosis(Rt_t))
        }
    return moments

def process_parameter_set(param_dict, param_set_name):
    """Process a single parameter set and return moments"""
    print(f"  📊 Computing moments for {param_set_name}...")

    # Extract parameters and ensure they are scalar floats
    omega = float(param_dict['omega'])
    alpha = float(param_dict['alpha'])
    beta = float(param_dict['beta'])
    gamma = float(param_dict['gamma'])
    lambda_ = float(param_dict['lambda'])

    print(f"    Parameters: omega={omega:.8f}, alpha={alpha:.8f}, beta={beta:.3f}, gamma={gamma:.2f}, lambda={lambda_:.2f}")

    # Run Monte Carlo simulation
    Rt = mc(omega, alpha, beta, gamma, lambda_)

    # Compute moments for all time points
    moments = compute_moments(Rt, TN)

    return {
        'parameters': param_dict,
        'moments_by_time': moments
    }

# Load calibration results from all_calibration_results.json
print("Loading calibration results from calibration_results/all_calibration_results.json...")
with open('calibration_results/all_calibration_results.json', 'r') as f:
    calibration_data = json.load(f)

print(f"✅ Found {len(calibration_data)} datasets")

all_mc_results = {}

# Process each dataset
for dataset_key, dataset_results in calibration_data.items():
    print(f"\n{'='*60}")
    print(f"🔬 Processing Monte Carlo for {dataset_key}")

    # Check if calibration was successful
    if 'error' in dataset_results:
        print(f"⚠️  Skipping {dataset_key} - calibration failed: {dataset_results.get('error', 'Unknown error')}")
        all_mc_results[dataset_key] = {
            'dataset_info': dataset_results.get('dataset_info', {}),
            'status': 'skipped_due_to_calibration_failure',
            'error': dataset_results.get('error', 'Unknown error')
        }
        continue

    dataset_info = dataset_results.get('dataset_info', {})
    print(f"Dataset: {dataset_info.get('dataset_name', 'Unknown')}")

    # Initialize results for this dataset
    dataset_mc_results = {
        'dataset_info': dataset_info,
        'parameter_sets': {}
    }

    # Process each parameter set (true, initial_guess, final_calibrated)
    parameter_sets = ['true_parameters', 'initial_guess_parameters', 'final_calibrated_parameters']

    for param_set_name in parameter_sets:
        if param_set_name in dataset_results:
            try:
                result = process_parameter_set(dataset_results[param_set_name], param_set_name)
                dataset_mc_results['parameter_sets'][param_set_name] = result
                print(f"    ✅ {param_set_name} completed")
            except Exception as e:
                print(f"    ❌ Error processing {param_set_name}: {e}")
                dataset_mc_results['parameter_sets'][param_set_name] = {
                    'error': str(e),
                    'status': 'failed'
                }

    # Store results for this dataset
    all_mc_results[dataset_key] = dataset_mc_results
    print(f"✅ Monte Carlo completed for {dataset_key}")

# Save all results to JSON
output_file = 'mc_moments_all_datasets.json'
with open(output_file, 'w') as f:
    json.dump(all_mc_results, f, indent=2)

print(f"\n🎉 Monte Carlo analysis completed for all datasets!")
print(f"💾 Results saved to {output_file}")

# Print summary statistics
successful_datasets = 0
total_parameter_sets = 0
successful_parameter_sets = 0

for dataset_key, dataset_result in all_mc_results.items():
    if 'parameter_sets' in dataset_result:
        successful_datasets += 1
        for param_set_name, param_result in dataset_result['parameter_sets'].items():
            total_parameter_sets += 1
            if 'error' not in param_result:
                successful_parameter_sets += 1

print(f"\n📈 Summary:")
print(f"  Total datasets: {len(all_mc_results)}")
print(f"  Successful datasets: {successful_datasets}")
print(f"  Total parameter sets processed: {total_parameter_sets}")
print(f"  Successful parameter sets: {successful_parameter_sets}")

# Print sample results for the first successful dataset
for dataset_key, dataset_result in all_mc_results.items():
    if 'parameter_sets' in dataset_result and 'true_parameters' in dataset_result['parameter_sets']:
        true_params = dataset_result['parameter_sets']['true_parameters']
        if 'moments_by_time' in true_params:
            print(f"\n📊 Sample results for {dataset_key} true parameters at t=252:")
            moments_252 = true_params['moments_by_time']['252']
            print(f"  Mean: {moments_252['mean']:.6f}")
            print(f"  Variance: {moments_252['variance']:.6f}")
            print(f"  Skewness: {moments_252['skewness']:.6f}")
            print(f"  Kurtosis: {moments_252['kurtosis']:.6f}")
            break
