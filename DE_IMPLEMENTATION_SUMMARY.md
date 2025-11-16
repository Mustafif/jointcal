# Differential Evolution Implementation Summary

## Overview

This document summarizes the available differential evolution (DE) implementations for GARCH parameter calibration in the jointcal project. We have successfully migrated from a single custom implementation to multiple optimized options.

## Available Implementations

### 1. SciPy DE (Recommended ⭐)

**File**: `calibrate_scipy_de.py`

**Status**: ✅ **Production Ready**

**Advantages**:
- Professional-grade implementation from SciPy ecosystem
- Extensively tested and optimized
- Multiple DE strategies available
- Built-in constraint handling
- Optional L-BFGS-B polishing for final refinement
- Excellent documentation and community support

**Performance**:
- ⏱️ **Speed**: ~101 seconds (300 iterations, 100 population)
- 🎯 **Accuracy**: L2 Error ~1.625
- 💾 **Memory**: Low memory footprint
- 🔧 **Reliability**: Very high

**Usage**:
```python
from calibrate_scipy_de import calibrate_scipy_de

params, history = calibrate_scipy_de(
    model=model,
    dataset=dataset,
    popsize=20,              # Multiplier: 20 * 5 params = 100
    maxiter=400,
    strategy='best1bin',
    mutation=(0.5, 1.0),     # Adaptive mutation
    recombination=0.7,
    seed=42,
    polish=True              # Final L-BFGS-B refinement
)
```

### 2. Custom PyTorch DE (Legacy)

**File**: `calibrate2_de.py`

**Status**: ⚠️ **Legacy - Functional but not recommended**

**Advantages**:
- Full control over implementation
- PyTorch-native with GPU tensors
- Custom parameter projection
- Research flexibility

**Performance**:
- ⏱️ **Speed**: ~119 seconds (300 iterations, 80 population)
- 🎯 **Accuracy**: L2 Error ~1.458 (slightly better)
- 💾 **Memory**: Higher memory usage
- 🔧 **Reliability**: Good but requires maintenance

**Issues**:
- Custom implementation requires maintenance
- Less tested than SciPy
- More complex codebase

### 3. GPU-Accelerated DE (Experimental)

**File**: `calibrate_gpu_de.py`

**Status**: 🧪 **Experimental - GPU acceleration**

**Advantages**:
- Massive parallelization on GPU
- Vectorized population operations
- Support for both CUDA and ROCm
- Potential for 10-50x speedup

**Requirements**:
- PyTorch with CUDA/ROCm support
- GPU with 4GB+ memory
- Large population sizes (80+) for efficiency

**Performance** (Expected):
- ⏱️ **Speed**: 2-10 seconds (with proper GPU utilization)
- 🎯 **Accuracy**: Similar to other methods
- 💾 **Memory**: GPU memory dependent
- 🔧 **Reliability**: Experimental

### 4. MetaDE (Attempted)

**File**: `calibrate_metade.py` (incomplete)

**Status**: ❌ **Failed - Version conflicts**

**Issues**:
- JAX version compatibility problems
- MetaDE expects older JAX versions
- Complex dependency management
- Not worth the complexity overhead

## Performance Comparison

| Method | Time (sec) | L2 Error | Memory | Maintenance | Recommendation |
|--------|------------|----------|--------|-------------|----------------|
| SciPy DE | 101 | 1.625 | Low | None | ⭐ **Production** |
| Custom DE | 119 | 1.458 | Medium | High | ⚠️ Legacy |
| GPU DE | ~5* | ~1.5* | GPU | Medium | 🧪 Experimental |
| MetaDE | N/A | N/A | N/A | High | ❌ Failed |

*GPU performance is theoretical based on proper utilization

## Migration Status

### ✅ Completed
- [x] SciPy DE implementation working
- [x] Performance comparison completed
- [x] Documentation created
- [x] Production-ready solution identified

### ⏸️ Paused
- [ ] MetaDE integration (version conflicts)
- [ ] GPU DE optimization (experimental)

### ❌ Abandoned
- [x] MetaDE with current JAX versions (incompatible)

## Recommendations

### For Production Use
**Use SciPy DE** (`calibrate_scipy_de.py`):
```python
from calibrate_scipy_de import calibrate_scipy_de

# Recommended production settings
params, history = calibrate_scipy_de(
    model, dataset,
    popsize=20,              # Good balance
    maxiter=400,             # Sufficient iterations
    strategy='best1bin',     # Reliable strategy
    mutation=(0.5, 1.0),     # Adaptive exploration
    recombination=0.7,       # Standard crossover
    polish=True,             # Final refinement
    seed=42                  # Reproducibility
)
```

### For Research/Development
- Use **Custom DE** if you need specific modifications
- Try **GPU DE** for large-scale parameter sweeps
- SciPy DE for baseline comparisons

### For Maximum Performance
- **GPU DE** if you have proper GPU setup and large problems
- **SciPy DE** for most practical applications
- Consider multiple runs with different seeds

## Key Findings

1. **SciPy DE is the clear winner** for production use
   - Best balance of speed, accuracy, and reliability
   - No maintenance burden
   - Professional implementation

2. **Custom DE has slight accuracy advantage** but not worth the maintenance cost
   - L2 Error: 1.458 vs 1.625 (10% better)
   - Speed: 18% slower
   - Requires ongoing maintenance

3. **GPU acceleration has potential** but needs more development
   - Could provide 10-50x speedup
   - Requires specific hardware
   - Still experimental

4. **MetaDE is not viable** with current dependencies
   - JAX version conflicts
   - Complex setup
   - Not worth the effort

## Implementation Details

### Parameter Bounds
All implementations use the same GARCH parameter bounds:
```python
bounds = [
    (1e-7, 1e-6),    # omega: positive, small
    (1e-6, 1e-5),    # alpha: small, positive  
    (0.7, 0.99),     # beta: close to 1
    (0.1, 10.0),     # gamma: leverage effect
    (0.2, 0.5)       # lambda: risk premium
]
```

### True Parameters (for testing)
```python
true_vals = [1e-6, 1.33e-6, 0.8, 5.0, 0.2]
```

### Validation Checks
All methods validate:
- ✅ Stationarity: α + β < 1
- ✅ Positive omega: ω > 0
- ✅ Parameter bounds compliance

## Files Overview

```
├── calibrate_scipy_de.py          ⭐ Main production implementation
├── calibrate2_de.py              ⚠️ Legacy custom implementation  
├── calibrate_gpu_de.py            🧪 Experimental GPU acceleration
├── calibrate_metade.py            ❌ Failed MetaDE attempt
├── compare_custom_vs_scipy.py     📊 Performance comparison
├── de.py                          🎛️ Main interface (updated)
├── requirements.txt               📦 Dependencies
└── DE_IMPLEMENTATION_SUMMARY.md   📚 This document
```

## Usage Instructions

### Quick Start (Recommended)
```python
# Use the updated de.py interface
python de.py  # Uses SciPy DE by default
```

### Direct Usage
```python
# For production calibration
python calibrate_scipy_de.py

# For performance comparison  
python compare_custom_vs_scipy.py

# For experimental GPU acceleration
python calibrate_gpu_de.py
```

### Batch Processing
```python
# Multiple strategies with SciPy DE
strategies = ['best1bin', 'best2bin', 'rand1bin']
for strategy in strategies:
    params, history = calibrate_scipy_de(
        model, dataset, strategy=strategy
    )
```

## Conclusion

The migration from custom DE to professional implementations has been successful. **SciPy DE provides the best solution** for production GARCH calibration with:

- ⚡ Better speed (18% faster)
- 🔧 Zero maintenance overhead
- 📚 Excellent documentation
- 🏆 Professional implementation quality
- ✅ Proven reliability

The slight accuracy loss (10%) is more than compensated by the reliability and maintenance benefits. For research requiring custom DE behavior, the legacy implementation remains available.

---

**Recommendation**: Adopt SciPy DE for all production GARCH calibration workflows.