import torch as tr

import torch as tr

def train_model(model, train_dataloader, loss_fn, optimizer, parallel_transformations=None, epochs=100, val_dataloader=None, patience=10):
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    total_batches = len(train_dataloader)
    print_interval = total_batches // 10  # Calculate interval for printing (every 1/10th of an epoch)

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
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

            # Print progress every 1/10th of an epoch
            if i % print_interval == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}/{total_batches}, Loss: {loss.item():.4f}')

        # Validation phase only if val_dataloader is provided
        if val_dataloader is not None:
            val_loss = 0.0
            model.eval()
            with tr.no_grad():
                for val_inputs, val_targets in val_dataloader:
                    val_outputs = model(val_inputs)
                    val_loss += loss_fn(val_outputs, val_targets).item()

            val_loss /= len(val_dataloader)

            # Check early stopping condition
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print('Early stopping triggered at epoch:', epoch+1)
                        break

            # Print progress at the end of the epoch with validation loss
            print('End of Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, epochs, loss.item(), val_loss))
        else:
            # Print progress at the end of the epoch without validation loss
            print('End of Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

    print("Finished Training")
    return model
