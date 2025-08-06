import iv
import torch

def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs
):
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
        for batch_X, batch_y, N, M in train_loader:
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

        # # Validation phase
        # model.eval()  # Set model to evaluation mode
        # val_loss = 0
        # with torch.no_grad():  # Disable gradient computation
        #     for batch_X, batch_y in val_loader:
        #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        #         output = model(batch_X.float())
        #         target = batch_y.float().view_as(output)
        #         loss = criterion(output, target)
        #         val_loss += loss.item()

        # # Calculate average validation loss for epoch
        # avg_val_loss = val_loss / len(val_loader)
        # val_losses.append(avg_val_loss)
        print(
            "Epoch {}: Train Loss {:.4f}".format(
                epoch + 1, avg_train_loss
            )
        )

    return model, train_losses, val_losses
