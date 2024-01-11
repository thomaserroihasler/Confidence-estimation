import torch as tr
import copy as cp


def get_accuracy(loader, model, device):
    """
    Calculate the accuracy of a model on a given dataset.

    This function iterates over the provided data loader, computes the model's
    predictions, and compares them with the actual labels to calculate accuracy.

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
    with tr.no_grad():  # Disable gradient calculations for efficiency
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the specified device
            outputs = model(images)
            _, predicted = tr.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Reset the model to training mode after evaluation
    return 100 * correct / total  # Return accuracy as a percentage


def device_saved_DICT(dictionary):
    """
    Determine the device a model state dictionary was saved on.

    This function checks if the keys of the state dictionary start with 'module.',
    which indicates that it was saved on a GPU, otherwise it assumes a CPU.

    Args:
        dictionary: State dictionary of a model.

    Returns:
        String indicating the device type ('cuda' or 'cpu').
    """
    if next(iter(dictionary.keys())).startswith('module.'):
        return 'cuda'  # Indicates model was saved with DataParallel on GPU
    else:
        return 'cpu'  # Indicates model was saved on CPU


def device_saved_FILE(file_location):
    """
    Determine the device type from a saved model file.

    This function loads a model state dictionary from a file and then
    uses `device_saved_DICT` to determine the device type.

    Args:
        file_location: Path to the model file.

    Returns:
        String indicating the device type ('cuda' or 'cpu').
    """
    state_dicts = tr.load(file_location)
    return device_saved_DICT(state_dicts[0])  # Use the first state dictionary for checking


def to_device_state_dict(state_dict, device):
    """
    Convert a model's state dictionary to a specified device.

    This function moves each tensor in the state dictionary to the specified device.
    It also adjusts the keys if necessary, depending on whether the model is being
    moved to or from a GPU.

    Args:
        state_dict: Model's state dictionary.
        device: The target device.

    Returns:
        A new state dictionary with all tensors moved to the specified device.
    """
    device_str = str(device)
    # Validate device
    if device_str != 'cpu' and not device_str.startswith('cuda'):
        raise ValueError(f"Invalid device: {device}. Only 'cpu' and devices starting with 'cuda' are supported.")
    if device_str == 'cuda' and not tr.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")

    new_state_dict = {}
    print(device_str)
    for key, value in state_dict.items():
        if isinstance(value, tr.Tensor):
            # Adjust the keys if necessary for GPU compatibility
            if device_str.startswith('cuda') and not key.startswith('module.'):
                # For GPU, prepend 'module.' if not present
                key = 'module.'+key  # No change made, adjust as needed
            elif device_str == 'cpu' and key.startswith('module.'):
                # For CPU, remove 'module.' prefix if present
                key = key[7:]
            new_state_dict[key] = value.to(device)
        else:
            new_state_dict[key] = value
    return new_state_dict

def remove_module(state_dict, device):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
            new_state_dict[key] = value.to(device)
        else:
            key = key
            new_state_dict[key] = value.to(device)
    return new_state_dict


def load_networks(network, file_location):
    """
    Load multiple networks from a file.

    This function loads state dictionaries from a file and applies them to
    copies of the provided network model. It supports loading multiple models
    if the file contains a list of state dictionaries.

    Args:
        network: The base neural network model to copy.
        file_location: Path to the file containing state dictionaries.

    Returns:
        A list of neural network models with loaded states.
    """
    state_dicts = tr.load(file_location)

    # Check if the file contains a single state dictionary or a list
    if isinstance(state_dicts, dict):
        state_dicts = [state_dicts]

    network_device = network.parameters().__next__().device
    loaded_networks = []

    for state_dict in state_dicts:
        #print((state_dict))
        # Deep copy the provided network and load each state dictionary
        loaded_network = cp.deepcopy(network)
        #loaded_network.load_state_dict(to_device_state_dict(state_dict, network_device))
        loaded_network.load_state_dict(remove_module(state_dict,network_device))
        loaded_networks.append(loaded_network)

    return loaded_networks
