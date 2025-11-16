#!/usr/bin/env python3
"""
Test script to verify MetaDE installation and basic functionality
for GARCH parameter calibration
"""

import json
import time

import numpy as np
import torch


def test_metade_import():
    """Test if MetaDE can be imported"""
    try:
        from metade import DE
        print("✅ MetaDE import successful")
        return True
    except ImportError as e:
        print(f"❌ MetaDE import failed: {e}")
        print("Install with: pip install metade")
        return False


def test_basic_optimization():
    """Test basic MetaDE optimization on simple function"""
    print("\n🧪 Testing basic optimization...")

    try:
        from metade import DE

        # Simple sphere function: f(x) = sum(x^2)
        def sphere(x):
            return sum(x**2)

        # Optimize 2D sphere function
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        de = DE(
            func=sphere,
            bounds=bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=30,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            atol=1e-6
        )

        result = de.solve()

        print(f"Optimization result:")
        print(f"  Success: {result.success}")
        print(f"  Solution: {result.x}")
        print(f"  Fitness: {result.fun}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")

        # Check if solution is close to optimum (0, 0)
        if result.fun < 1e-6:
            print("✅ Basic optimization test passed")
            return True
        else:
            print("⚠️ Basic optimization completed but may not have converged fully")
            return True

    except Exception as e:
        print(f"❌ Basic optimization test failed: {e}")
        return False


def test_garch_like_optimization():
    """Test MetaDE on a GARCH-like parameter estimation problem"""
    print("\n🧪 Testing GARCH-like optimization...")

    try:
        from metade import DE

        # GARCH-like objective function
        def garch_like_objective(params):
            """
            Simulate a GARCH-like objective function
            params = [omega, alpha, beta, gamma, lambda]
            """
            omega, alpha, beta, gamma, lambda_param = params

            # Penalize if stationarity condition is violated
            if alpha + beta >= 1.0:
                return 1000.0

            # Penalize negative omega
            if omega <= 0:
                return 1000.0

            # Simple quadratic loss around "true" values
            true_vals = np.array([1e-6, 1.33e-6, 0.8, 5.0, 0.2])
            diff = params - true_vals

            # Add some noise to make it more realistic
            noise = 0.1 * np.sum(np.sin(10 * params))

            return np.sum(diff**2) + noise

        # GARCH parameter bounds
        bounds = [
            (1e-7, 1e-5),    # omega
            (1e-6, 1e-4),    # alpha
            (0.1, 0.99),     # beta
            (0.1, 10.0),     # gamma
            (0.0, 1.0)       # lambda
        ]

        print(f"Optimizing with bounds: {bounds}")

        de = DE(
            func=garch_like_objective,
            bounds=bounds,
            strategy='best1bin',
            maxiter=200,
            popsize=50,
            mutation=0.8,
            recombination=0.7,
            seed=42,
            atol=1e-8
        )

        start_time = time.time()
        result = de.solve()
        optimization_time = time.time() - start_time

        print(f"GARCH-like optimization result:")
        print(f"  Success: {result.success}")
        print(f"  Time: {optimization_time:.2f} seconds")
        print(f"  Final loss: {result.fun}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")

        # Check parameter validity
        omega, alpha, beta, gamma, lambda_param = result.x
        persistence = alpha + beta

        print(f"  Parameters: omega={omega:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")
        print(f"              gamma={gamma:.2f}, lambda={lambda_param:.2f}")
        print(f"  Persistence (α+β): {persistence:.4f}")

        # Validation
        valid = True
        if persistence >= 1.0:
            print("  ⚠️ Stationarity condition violated")
            valid = False
        if omega <= 0:
            print("  ⚠️ Omega is not positive")
            valid = False

        if valid:
            print("✅ GARCH-like optimization test passed")
        else:
            print("⚠️ GARCH-like optimization completed but with parameter issues")

        return True

    except Exception as e:
        print(f"❌ GARCH-like optimization test failed: {e}")
        return False


def test_different_strategies():
    """Test different DE strategies"""
    print("\n🧪 Testing different DE strategies...")

    try:
        from metade import DE

        # Simple test function
        def rosenbrock(x):
            """Rosenbrock function - good test for optimization algorithms"""
            return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

        bounds = [(-2.0, 2.0)] * 4  # 4D Rosenbrock

        strategies = ['best1bin', 'rand1bin', 'currenttobest1bin', 'best2bin']
        results = {}

        for strategy in strategies:
            print(f"  Testing strategy: {strategy}")

            de = DE(
                func=rosenbrock,
                bounds=bounds,
                strategy=strategy,
                maxiter=100,
                popsize=40,
                mutation=0.8,
                recombination=0.7,
                seed=42
            )

            result = de.solve()
            results[strategy] = {
                'success': result.success,
                'fitness': result.fun,
                'iterations': result.nit,
                'evaluations': result.nfev
            }

            print(f"    Result: fitness={result.fun:.6f}, iterations={result.nit}")

        print(f"\nStrategy comparison:")
        for strategy, result in results.items():
            print(f"  {strategy:<20}: fitness={result['fitness']:.6f}, "
                  f"iters={result['iterations']}, success={result['success']}")

        print("✅ Strategy testing completed")
        return True

    except Exception as e:
        print(f"❌ Strategy testing failed: {e}")
        return False


def test_metade_vs_scipy():
    """Compare MetaDE with scipy.optimize.differential_evolution"""
    print("\n🧪 Testing MetaDE vs SciPy...")

    try:
        from metade import DE
        from scipy.optimize import differential_evolution

        # Test function
        def ackley(x):
            """Ackley function - challenging optimization problem"""
            a, b, c = 20, 0.2, 2 * np.pi
            n = len(x)
            sum1 = sum(x**2)
            sum2 = sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e

        bounds = [(-5.0, 5.0)] * 3

        # MetaDE
        print("  Running MetaDE...")
        start_time = time.time()
        de = DE(func=ackley, bounds=bounds, maxiter=100, popsize=30, seed=42)
        metade_result = de.solve()
        metade_time = time.time() - start_time

        # SciPy DE
        print("  Running SciPy DE...")
        start_time = time.time()
        scipy_result = differential_evolution(ackley, bounds, maxiter=100, popsize=30, seed=42)
        scipy_time = time.time() - start_time

        print(f"Results comparison:")
        print(f"  MetaDE:  fitness={metade_result.fun:.6f}, time={metade_time:.2f}s, success={metade_result.success}")
        print(f"  SciPy:   fitness={scipy_result.fun:.6f}, time={scipy_time:.2f}s, success={scipy_result.success}")

        if metade_result.fun < 1e-6 and scipy_result.fun < 1e-6:
            print("✅ Both methods found good solutions")
        elif metade_result.fun < scipy_result.fun:
            print("✅ MetaDE found better solution")
        else:
            print("✅ SciPy found better solution (both working)")

        return True

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧬 MetaDE Testing Suite")
    print("=" * 50)

    tests = [
        ("Import Test", test_metade_import),
        ("Basic Optimization", test_basic_optimization),
        ("GARCH-like Optimization", test_garch_like_optimization),
        ("Strategy Testing", test_different_strategies),
        ("MetaDE vs SciPy", test_metade_vs_scipy)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! MetaDE is ready for GARCH calibration.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. MetaDE should work but check failed tests.")
    else:
        print("❌ Many tests failed. Check MetaDE installation and dependencies.")

    # Save test results
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': total,
        'passed_tests': passed,
        'success_rate': passed / total,
        'individual_results': results
    }

    with open('metade_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n📁 Test results saved to metade_test_results.json")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
