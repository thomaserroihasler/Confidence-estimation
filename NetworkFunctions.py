import torch as tr

def get_accuracy(loader, model, device):
    """
    Calculate the accuracy of a model on a given dataset.

    Args:
        loader: DataLoader for the dataset.
        model: Neural network model to evaluate.
        device: The device (CPU or GPU) to perform calculations on.

    Returns:
        Accuracy as a percentage.
    """
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode to disable dropout, etc.
    with tr.no_grad():  # Disable gradient calculations
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the specified device
            outputs = model(images)
            _, predicted = tr.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set the model back to training mode
    return 100 * correct / total
