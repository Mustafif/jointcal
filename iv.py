import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import pandas as pd
from loss import calculate_loss
import numpy as np
import json
from sklearn.model_selection import train_test_split
from utils import save_model_checkpoint
from model import IV

def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
):
    """
    Train the model with improved learning rate scheduling and regularization

    Args:
        model: PyTorch neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run training on (CPU/GPU)
        epochs: Number of training epochs

    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    # Initialize learning rate scheduler using OneCycleLR policy
    # This implements a cyclical learning rate that starts at a low value,
    # increases to max_lr, then decreases again following a cosine curve
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Percentage of training spent increasing learning rate
        anneal_strategy="cos",  # Use cosine annealing
    )

    train_losses = []  # Track training losses over epochs
    val_losses = []  # Track validation losses over epochs

    # Training loop for specified number of epochs
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0

        # Train on batches of data
        for batch_X, batch_y in train_loader:
            # Move data to specified device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()  # Reset gradients

            output = model(batch_X.float())  # Forward pass
            target = batch_y.float().view_as(output)  # Reshape target to match output
            loss = criterion(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate
            train_loss += loss.item()  # Accumulate batch loss

        # Calculate average training loss for epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)
                loss = criterion(output, target)
                val_loss += loss.item()

        # Calculate average validation loss for epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(
            "Epoch {}: Train Loss {:.4f} Val Loss {:.4f}".format(
                epoch + 1, avg_train_loss, avg_val_loss
            )
        )

    return model, train_losses, val_losses


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates model performance on a dataset using the provided data loader.

    Args:
        model: PyTorch neural network model to evaluate
        data_loader: DataLoader containing evaluation data
        criterion: Loss function used for evaluation
        device: Device to run evaluation on (CPU/GPU)

    Returns:
        avg_loss: Average loss across all batches
        predictions: Array of model predictions
        targets: Array of target values
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss counter
    predictions = []  # List to store model predictions
    targets = []  # List to store targets

    with torch.no_grad():  # Disable gradient computation for evaluation
        for X, Y in data_loader:  # Loop through batches
            X, Y = X.to(device), Y.to(device)  # Move data to device
            output = model(X)  # Get model predictions
            # Reshape target to match output dimensions
            target = Y.float().view_as(output)
            loss = criterion(output, target)  # Calculate loss for batch
            total_loss += loss.item()  # Add batch loss to total

            # Store predictions and targets as flattened arrays
            predictions.extend(
                output.cpu().numpy().flatten()
            )  # Convert predictions to numpy array
            targets.extend(Y.cpu().numpy().flatten())  # Convert targets to numpy array

    avg_loss = total_loss / len(
        data_loader
    )  # Calculate average loss across all batches
    return avg_loss, np.array(predictions), np.array(targets)


def train_val_split(dataset, val_size=0.2, random_state=42):
    """
    Splits a dataset into training and validation sets by creating samplers.

    Args:
        dataset: The full dataset to split
        val_size: Fraction of data to use for validation (default 0.2 or 20%)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        train_sampler: Sampler object for training data
        val_sampler: Sampler object for validation data
    """
    # Get indices of the full dataset
    indices = list(range(len(dataset)))

    # Split indices into train and validation
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=random_state
    )

    # Create samplers for train and validation
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


def iv_main(dataset_train, dataset_test, name, params_path):
    """
    Main function for training and evaluating the AO-ANN model.

    Args:
        dataset_train: Training dataset
        dataset_test: Test dataset
        name: Name identifier for the model
        params_path: Path to JSON file containing model hyperparameters

    This function:
    1. Sets up training environment (device, precision)
    2. Loads hyperparameters from JSON
    3. Creates data samplers and loaders for train/val/test
    4. Initializes model, loss function and optimizer
    5. Trains the model
    6. Evaluates performance on train and test sets
    7. Saves predictions and metrics to files
    """
    torch.set_float32_matmul_precision("high")
    num_workers = 6
    device = torch.device("cuda")
    print(f"Using device: {device}")
    params = json.load(open(params_path))
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    epochs = params["epochs"]
    target_scaler = dataset_train.target_scaler

    # Create samplers
    train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.2)
    test_sampler = RandomSampler(dataset_test)

    # Create data loaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_train,  # Using test dataset for validation
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    dropout_rate = params["dropout_rate"]
    # Model setup
    model = IV(dropout_rate=dropout_rate).to(device)
    criterion = nn.HuberLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Training
    trained_model, tl, vl = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs
    )

    # Evaluation
    train_loss, train_pred, train_target = evaluate_model(
        trained_model, train_loader, criterion, device
    )
    test_loss, test_pred, test_target = evaluate_model(
        trained_model, test_loader, criterion, device
    )

    # Convert predictions and targets to plain floats
    train_pred = [float(x) for x in train_pred]
    train_target = [float(x) for x in train_target]

    test_pred = [float(x) for x in test_pred]
    test_target = [float(x) for x in test_target]

    # Save results
    train_df = pd.DataFrame({"predictions": train_pred, "targets": train_target})
    train_df.to_csv("train_results.csv", index=False)

    test_df = pd.DataFrame({"predictions": test_pred, "targets": test_target})
    test_df.to_csv("test_results.csv", index=False)

    # Calculate and print loss details
    print("In-sample (Training) Performance:")
    train_loss_details = calculate_loss("train_results.csv", target_scaler)
    for key, value in train_loss_details.items():
        print(f"{key}: {value}")

    print("\nOut-of-sample (Test) Performance:")
    test_loss_details = calculate_loss("test_results.csv", target_scaler)
    for key, value in test_loss_details.items():
        print(f"{key}: {value}")

    # Save metrics
    metrics = {"in_sample": train_loss_details, "out_of_sample": test_loss_details}
    save_model_checkpoint(trained_model, name, metrics, tl, vl)
