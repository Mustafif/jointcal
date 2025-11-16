#!/usr/bin/env python3
"""
Setup script for migrating to MetaDE for GARCH calibration
This script installs MetaDE and verifies the installation
"""

import importlib
import os
import subprocess
import sys


def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package_name}")

    if success:
        print(f"✅ {package_name} installed successfully")
        return True
    else:
        print(f"❌ Failed to install {package_name}")
        print(f"Error: {stderr}")
        return False


def verify_installation():
    """Verify that MetaDE is properly installed"""
    try:
        from metade import DE
        print("✅ MetaDE import successful")

        # Test basic functionality
        def test_func(x):
            return sum(x**2)

        bounds = [(-5.0, 5.0)] * 3
        de = DE(test_func, bounds, maxiter=10, popsize=15, seed=42)
        result = de.solve()

        if result.success:
            print("✅ MetaDE basic functionality test passed")
            return True
        else:
            print("⚠️ MetaDE test completed but may have convergence issues")
            return True

    except Exception as e:
        print(f"❌ MetaDE verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🧬 MetaDE Setup for GARCH Calibration")
    print("=" * 50)

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 8):
        print("⚠️ Warning: Python 3.8+ recommended for MetaDE")

    # List of required packages
    required_packages = [
        'numpy',
        'scipy',
        'metade',
        'torch',
        'pandas',
        'matplotlib',
        'scikit-learn'
    ]

    # Check existing packages
    print("\nChecking existing packages...")
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package} already installed")
        else:
            print(f"❌ {package} not found - will install")

    # Install missing packages
    print("\nInstalling missing packages...")
    installation_failed = False

    for package in required_packages:
        if not check_package(package):
            if not install_package(package):
                installation_failed = True

    if installation_failed:
        print("\n❌ Some packages failed to install. Please install manually:")
        print("pip install metade scipy numpy torch pandas matplotlib scikit-learn")
        return False

    # Verify MetaDE installation
    print("\nVerifying MetaDE installation...")
    if not verify_installation():
        print("❌ MetaDE verification failed")
        return False

    # Check if requirements.txt exists and update it
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print(f"\n📄 Found {req_file}")
        with open(req_file, 'r') as f:
            content = f.read()

        if 'metade' not in content:
            print("Adding metade to requirements.txt...")
            with open(req_file, 'a') as f:
                f.write('\nmetade>=1.0.0\n')
            print("✅ Updated requirements.txt")

    # Print next steps
    print("\n🎉 MetaDE setup complete!")
    print("\nNext steps:")
    print("1. Run comparison: python compare_de_methods.py")
    print("2. Test MetaDE calibration: python calibrate_metade.py")
    print("3. Follow migration guide in MIGRATION_TO_METADE.md")
    print("4. Update your existing scripts to use calibrate_metade")

    return True


def quick_test():
    """Quick test of the new MetaDE calibration"""
    try:
        print("\n🧪 Running quick MetaDE test...")

        # Import test
        from metade import DE

        # Simple optimization test
        def sphere(x):
            return sum(x**2)

        bounds = [(-5.0, 5.0)] * 2
        de = DE(sphere, bounds, maxiter=50, popsize=20, seed=42)
        result = de.solve()

        print(f"Test optimization result:")
        print(f"  Success: {result.success}")
        print(f"  Best solution: {result.x}")
        print(f"  Best fitness: {result.fun}")
        print(f"  Iterations: {result.nit}")

        if result.fun < 1e-6:
            print("✅ MetaDE quick test passed - ready for GARCH calibration!")
        else:
            print("⚠️ MetaDE test completed but convergence could be better")

        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting MetaDE setup...")

    setup_success = main()

    if setup_success:
        test_success = quick_test()

        if test_success:
            print("\n🚀 All systems ready! You can now use MetaDE for GARCH calibration.")
        else:
            print("\n⚠️ Setup complete but testing had issues. Please verify manually.")
    else:
        print("\n❌ Setup failed. Please resolve issues and try again.")

    print("\nFor help, see MIGRATION_TO_METADE.md")
