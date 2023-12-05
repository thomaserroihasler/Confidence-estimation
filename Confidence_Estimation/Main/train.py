from torch.utils.data import DataLoader
import torch as tr

def train_model(model, dataset, loss_fn, optimizer, epochs=100, batch_size=32, shuffle = False):
    # Define DataLoader for dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # Train model
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            # Compute predictions and loss
            #print(model)
            outputs = model(inputs)
            #print(outputs.dtype, targets.dtype)
            #print(outputs.shape,targets.shape)
            #print(outputs,targets)
            #print('in the train function',outputs.dtype, targets.dtype)
            loss = loss_fn(outputs, targets.long())
            # Compute gradients and update weights
            loss.backward(retain_graph=True)
            optimizer.step()
            # if (epoch % 100 == 3):
            #     print('for epoch number ',(epoch))
            #     for name, param in model.named_parameters():
            #         print(name, param.data)
            #         print('the gradients of the parameter are',param.grad)
            optimizer.zero_grad()
        # Print progress
        if (epoch%500 == 499):
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

    return model
