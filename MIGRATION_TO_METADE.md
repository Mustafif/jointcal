# Migration Guide: From Custom DE to MetaDE

This guide explains how to migrate from our custom Differential Evolution implementation to the MetaDE library for GARCH parameter calibration.

## Overview

We're replacing the custom `DifferentialEvolution` class in `calibrate2_de.py` with the professional MetaDE library to improve:
- Code maintainability
- Algorithm performance
- Implementation reliability
- Access to advanced DE strategies

## Installation

First, install MetaDE:

```bash
pip install metade
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Key Changes

### 1. Import Changes

**Before (Custom DE):**
```python
from calibrate2_de import DifferentialEvolution, calibrate_de
```

**After (MetaDE):**
```python
from metade import DE
from calibrate_metade import calibrate_metade
```

### 2. Function Call Changes

**Before:**
```python
# Custom DE
params, history = calibrate_de(
    model=model,
    dataset=dataset,
    popsize=50,
    max_iter=500,
    mutation=0.8,
    crossover=0.7,
    seed=42
)
```

**After:**
```python
# MetaDE
params, history = calibrate_metade(
    model=model,
    dataset=dataset,
    popsize=50,
    max_iter=500,
    strategy='best1bin',  # New: DE strategy
    F=0.8,               # Renamed from mutation
    CR=0.7,              # Renamed from crossover
    seed=42
)
```

### 3. Parameter Changes

| Custom DE | MetaDE | Description |
|-----------|---------|-------------|
| `mutation` | `F` | Differential weight (same values) |
| `crossover` | `CR` | Crossover probability (same values) |
| N/A | `strategy` | DE strategy ('best1bin', 'rand1bin', etc.) |

### 4. Available DE Strategies

MetaDE supports multiple strategies:
- `'best1bin'`: DE/best/1/bin (default, good for most cases)
- `'rand1bin'`: DE/rand/1/bin (more exploratory)
- `'currenttobest1bin'`: DE/current-to-best/1/bin (good convergence)
- `'best2bin'`: DE/best/2/bin (uses two difference vectors)
- `'rand2bin'`: DE/rand/2/bin (more robust)

## Migration Steps

### Step 1: Update Existing Files

Replace calls to `calibrate_de` with `calibrate_metade` in these files:

1. **de.py:**
```python
# OLD
from calibrate2_de import calibrate_de

# NEW
from calibrate_metade import calibrate_metade

# OLD
params, history = calibrate_de(
    model=model,
    dataset=dataset,
    popsize=50,
    max_iter=500,
    mutation=0.8,
    crossover=0.7
)

# NEW
params, history = calibrate_metade(
    model=model,
    dataset=dataset,
    popsize=50,
    max_iter=500,
    strategy='best1bin',
    F=0.8,
    CR=0.7,
    seed=42
)
```

2. **compare_calibration.py:**
Update the DE section to use MetaDE

3. **Any other files importing from calibrate2_de**

### Step 2: Test Migration

Run the comparison script to verify MetaDE works correctly:

```bash
python compare_de_methods.py
```

This will:
- Run both custom DE and MetaDE
- Compare results and performance
- Generate comparison plots
- Save detailed analysis

### Step 3: Update Scripts

Update your main calibration scripts:

**Example update for run_de_calibration.py:**
```python
# OLD
from calibrate2_de import calibrate_de

calibrated_params, history = calibrate_de(
    model, dataset,
    popsize=100,
    max_iter=500,
    mutation=(0.5, 1.0),  # Range
    crossover=0.7
)

# NEW
from calibrate_metade import calibrate_metade

calibrated_params, history = calibrate_metade(
    model, dataset,
    popsize=100,
    max_iter=500,
    strategy='best1bin',
    F=0.8,  # Or use (0.5, 1.0) for adaptive
    CR=0.7
)
```

## Configuration Recommendations

### For Most Cases:
```python
calibrate_metade(
    model=model,
    dataset=dataset,
    popsize=100,
    max_iter=500,
    strategy='best1bin',
    F=0.8,
    CR=0.7,
    seed=42
)
```

### For Difficult Optimization Problems:
```python
calibrate_metade(
    model=model,
    dataset=dataset,
    popsize=150,          # Larger population
    max_iter=1000,        # More iterations
    strategy='currenttobest1bin',  # Better convergence
    F=0.9,                # Higher exploration
    CR=0.8,               # Higher recombination
    seed=42
)
```

### For Fast Prototyping:
```python
calibrate_metade(
    model=model,
    dataset=dataset,
    popsize=50,
    max_iter=200,
    strategy='best1bin',
    F=0.7,
    CR=0.6,
    seed=42
)
```

## Benefits of Migration

1. **Better Performance**: MetaDE is optimized and often faster
2. **More Robust**: Better handling of edge cases and numerical issues
3. **More Features**: Access to multiple DE strategies
4. **Maintained Code**: Regular updates and bug fixes
5. **Better Documentation**: Comprehensive API documentation
6. **Testing**: Extensively tested on various problems

## Backward Compatibility

To maintain backward compatibility during transition:

1. Keep `calibrate2_de.py` for now
2. Use `calibrate_metade.py` for new code
3. Gradually migrate existing scripts
4. Remove custom DE after full migration

## Testing Checklist

- [ ] Install MetaDE library
- [ ] Run comparison script successfully
- [ ] Verify results are comparable or better
- [ ] Update main calibration scripts
- [ ] Test with your specific datasets
- [ ] Verify parameter validation still works
- [ ] Check convergence plots look reasonable

## Troubleshooting

### Common Issues:

1. **Import Error**: Make sure MetaDE is installed
   ```bash
   pip install metade
   ```

2. **Different Results**: This is expected and often better. MetaDE may find better solutions.

3. **Parameter Bounds**: Ensure bounds are correctly specified as tuples: `[(min1, max1), (min2, max2), ...]`

4. **Strategy Selection**: Start with `'best1bin'` and experiment with others if needed.

## Performance Comparison

Run the comparison to see improvements:

```bash
python compare_de_methods.py
```

Expected improvements:
- Better convergence properties
- More consistent results across runs
- Often better final parameter estimates
- Comparable or better computation time

## Support

- MetaDE Documentation: Check PyPI page for metade
- Issues: Report issues with migration in your project tracker
- Questions: Consult team members familiar with optimization algorithms

## Next Steps

1. Complete migration to MetaDE
2. Remove custom DE implementation once confident
3. Consider using other advanced features of MetaDE
4. Explore other optimization strategies if needed

---

*This migration should improve the robustness and maintainability of your GARCH calibration system while potentially improving performance.*