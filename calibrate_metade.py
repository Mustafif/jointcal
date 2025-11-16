import json
import numpy as np
import torch
from scipy.optimize import differential_evolution
import time

from cal_loss import Calibration_Loss
from dataset2 import cal_dataset
from hn import HestonNandiGARCH

MODEL_PATH = "saved_models/varying_garch_dataset_50x30_5params_20250827/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaDEWrapper:
    """
    MetaDE-inspired Differential Evolution implementation using SciPy backend
    This provides the same interface as MetaDE but uses scipy.optimize.differential_evolution
    which is more stable and doesn't require JAX dependencies.
    """
    
    def __init__(self, func, bounds, strategy='best1bin', maxiter=1000, popsize=50, 
                 atol=1e-6, seed=None, mutation=(0.5, 1.0), recombination=0.7,
                 polish=True, init='latinhypercube'):
        """
        Initialize MetaDE-style optimizer
        
        Args:
            func: Objective function to minimize
            bounds: List of (min, max) tuples for each parameter
            strategy: DE strategy
            maxiter: Maximum iterations
            popsize: Population size multiplier
            atol: Absolute tolerance
            seed: Random seed
            mutation: Mutation factor or range
            recombination: Crossover probability
            polish: Whether to polish final result
            init: Population initialization method
        """
        self.func = func
        self.bounds = bounds
        self.strategy = strategy
        self.maxiter = maxiter
        self.popsize = popsize
        self.atol = atol
        self.seed = seed
        self.mutation = mutation
        self.recombination = recombination
        self.polish = polish
        self.init = init
        
        # Track convergence
        self.convergence_history = []
        self.iteration_count = 0
        
    def solve(self, callback=None):
        """
        Solve the optimization problem
        
        Returns:
            result: Optimization result with MetaDE-style interface
        """
        
        def wrapped_callback(xk, convergence):
            """Wrapper for callback function"""
            self.iteration_count += 1
            current_loss = self.func(xk)
            self.convergence_history.append(current_loss)
            
            if callback:
                callback(xk, convergence)
                
        print(f"🧬 Starting MetaDE-style optimization...")
        print