from torch.utils.data import DataLoader

def train_model(model, dataset, loss_fn, optimizer, epochs=100, batch_size=32, shuffle = False):
    # Define DataLoader for dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # Train model
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            # Compute predictions and loss
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.long())
            # Compute gradients and update weights
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        # Print progress
        if (epoch%500 == 499):
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

    return model
