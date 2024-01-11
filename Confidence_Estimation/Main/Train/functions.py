import torch as tr

def train_model(model, train_dataloader, loss_fn, optimizer, device, parallel_transformations=None, epochs=100, val_dataloader=None, patience=10):
    """
    Train a model with the given parameters, supporting multi-GPU or CPU environments.

    Args:
        model: The model to train.
        train_dataloader: DataLoader for the training dataset.
        loss_fn: Loss function.
        optimizer: Optimizer for the model.
        device: The device to train the model on ('cuda' or 'cpu').
        parallel_transformations: Optional list of parallel transformations to apply.
        epochs: Total number of training epochs.
        val_dataloader: Optional DataLoader for the validation dataset.
        patience: Number of epochs for early stopping patience.

    Returns:
        The trained model.
    """

    model = model.to(device)  # Move the model to the specified device
    model.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    total_batches = len(train_dataloader)
    print_interval = total_batches // 10  # Interval for printing progress

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device

            # Apply parallel transformations if any
            if parallel_transformations:
                for t in parallel_transformations:
                    inputs = t(inputs)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()  # Clear existing gradients
            loss.backward()
            optimizer.step()

            # Print training progress
            if i % print_interval == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}/{total_batches}, Loss: {loss.item():.4f}')

        # Validation phase
        if val_dataloader is not None:
            val_loss = 0.0
            model.eval()  # Set model to evaluation mode
            with tr.no_grad():
                for val_inputs, val_targets in val_dataloader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss += loss_fn(val_outputs, val_targets).item()

            val_loss /= len(val_dataloader)

            # Early stopping condition
            if patience is not None and val_loss >= best_val_loss:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print('Early stopping triggered at epoch:', epoch+1)
                    break
            else:
                best_val_loss = val_loss
                epochs_no_improve = 0

            # Print validation progress
            print(f'End of Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
        else:
            # Print progress for epoch without validation
            print(f'End of Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("Finished Training")
    return model
