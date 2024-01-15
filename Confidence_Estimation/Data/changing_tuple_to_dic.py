import ast  # For safely evaluating strings as Python expressions
import sys
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path
import torch as tr

def load_weights_from_pt_file(file_path):
    # Load the data from the .pt file
    network_data = tr.load(file_path)

    # Initialize an empty dictionary to store weights
    weights_dicts = []
    # Iterate over each network's data in the tuple
    for network in network_data:
        # Extract all but the last element (accuracy) which are the weight tuples
        for a in network[:-1]:
            weights_dict = {}
            for weight_name, weight_tensor in a.items():
                weights_dict[weight_name] = weight_tensor
            weights_dicts += [weights_dict]
    print(weights_dicts)
    return weights_dicts
# Example usage
file_path = '../../Networks/'+'CIFAR-10'+'/VGG/'+'save.pt'
file_path_2 = '../../Networks/'+'CIFAR-10'+'/VGG/'+'save.pth'
weights = load_weights_from_pt_file(file_path)
tr.save(weights,file_path_2)
