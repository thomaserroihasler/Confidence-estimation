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


def device_saved_DICT(dictionary):

    if next(iter(dictionary.keys())).startswith('module.'):
        # If the first state dictionary has a 'module' key, it was saved on a GPU
        return 'cuda'
    else:
        # If the first state dictionary does not have a 'module' key, it was saved on the CPU
        return 'cpu'

def device_saved_FILE(file_location):
    # Load the state dictionaries from the file
    state_dicts = tr.load(file_location)
    # Check if the first state dictionary has a 'module' key
    return device_saved_DICT(state_dicts[0])

def to_device_state_dict(state_dict, device):
    device_str = str(device)

    if device_str != 'cpu' and not device_str.startswith('cuda'):
        raise ValueError(f"Invalid device: {device}. Only 'cpu' and devices starting with 'cuda' are supported.")
    if device_str == 'cuda' and not tr.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    new_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, tr.Tensor):
            if device_str.startswith('cuda') and not key.startswith('module.'):
                #print(key)
                #key = 'module.' + key
                key =key
                #print(key)
            elif device_str == 'cpu' and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value.to(device)
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_networks(network, file_location):
    # Load the state dictionaries from the file
    state_dicts = tr.load(file_location)

    # Check if it's a single state dictionary
    if isinstance(state_dicts, dict):
        state_dicts = [state_dicts]

    network_device = network.parameters().__next__().device
    # Create a list to store the loaded networks
    loaded_networks = []
    #print('in the load networks function',network_device)
    # Loop over the state dictionaries
    for state_dict in state_dicts:
        # Create a new network of the same type as the input network
        loaded_network = cp.deepcopy(network)
        loaded_network.load_state_dict(to_device_state_dict(state_dict, network_device))

        # Add the loaded network to the list of loaded networks
        loaded_networks.append(loaded_network)

    return loaded_networks

