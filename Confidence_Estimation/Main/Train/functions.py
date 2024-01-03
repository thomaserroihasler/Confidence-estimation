def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, parallel_transformations=None, epochs=100, patience=10):
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        for inputs, targets in train_dataloader:
            # Compute predictions and loss
            if parallel_transformations:
                for t in parallel_transformations:
                    inputs = t(inputs)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # Compute gradients and update weights
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_outputs = model(val_inputs)
                val_loss += loss_fn(val_outputs, val_targets).item()

        val_loss /= len(val_dataloader)

        # Check early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping triggered at epoch:', epoch+1)
                break

        # Print progress
        if (epoch % 50 == 49):
            print('Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, epochs, loss.item(), val_loss))

    return model
