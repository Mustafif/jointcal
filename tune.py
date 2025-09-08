import json
import optuna
from optuna.trial import TrialState

# Import your existing training code
from iv import train_and_evaluate
from main import datasets

dataset_train, dataset_test = datasets[0].datasets()

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    Returns the validation loss to be minimized.
    """

    # Suggest hyperparameters
    params = {
        "lr": trial.suggest_loguniform("lr", 1e-6, 1e-2),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-8, 1e-4),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 20, 100),
        "dropout_rate": trial.suggest_uniform("dropout_rate", 0.0, 0.5),
    }

    # Train and evaluate
    _, _, val_loss, _, _, _, _ = train_and_evaluate(dataset_train, dataset_test, params)

    # Report intermediate results for pruning
    trial.report(val_loss, step=0)

    # If the trial should be pruned, stop early
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss


def main():
    # Create a study for minimization
    study = optuna.create_study(
        direction="minimize",
        study_name="AO-ANN Hyperparameter Optimization",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    # Run the optimization
    study.optimize(objective, n_trials=50, timeout=None)

    # Print summary
    print("\n=== Optuna Study Summary ===")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")

    # Best trial
    print("\n=== Best Trial ===")
    trial = study.best_trial
    print(f"Validation Loss: {trial.value}")
    print("Best Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save best params to JSON
    with open("best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    print("\nBest parameters saved to best_params.json")


if __name__ == "__main__":
    main()
