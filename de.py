from calibrate2_de import calibrate_de

# Run DE calibration
params, history = calibrate_de(
    model=model,
    dataset=dataset,
    popsize=50,      # Population size
    max_iter=500,    # Maximum iterations
    mutation=0.8,    # Mutation factor (or (0.5, 1.0) for range)
    crossover=0.7    # Crossover probability
)
