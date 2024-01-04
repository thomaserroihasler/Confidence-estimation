import torch as tr
import os
import sys
from torch.utils.data import DataLoader

# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

# Import necessary modules from Confidence Estimation
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name
from Confidence_Estimation.Data.Data_sets.functions import load_and_preprocess_data
from Confidence_Estimation.Configurations.definitions import *
from Confidence_Estimation.Configurations.functions import *

# Get the device name for PyTorch operations
device = print_device_name()

# assigning all necessary variables
additional_transformations = GENERATION_OUTPUT_ADDITIONAL_TRANSFORMATIONS  # Additional transformations for data generation output
basic_transformations = BASIC_TRANSFORMATIONS                             # Basic transformations for the dataset
batch_size = GENERATION_OUTPUT_BATCH_SIZE                                 # Batch size for data generation output
classes_to_include = Classes_to_consider(NUMBER_OF_CLASSES)               # List of class indices to include in the dataset
dataset_config = CONFIG[DATASET_NAME]                                     # Configuration settings for the dataset
dataset_name = DATASET_NAME                                               # Name of the dataset
model_config = MODELS[NETWORK_NAME]                                       # Configuration settings for the model
network_location = NETWORK_LOCATION                                       # Location for saving/loading the network model
network_name = NETWORK_NAME                                               # Name of the network (model)
number_of_classes = NUMBER_OF_CLASSES                                     # Number of classes in the dataset
number_of_transformations = GENERATION_OUTPUT_NUMBER_OF_TRANSFORMATIONS   # Number of transformations for data generation output
output_location = OUTPUT_LOCATION                                         # Location for saving the model output
split_sizes = SPLIT_SIZES                                                 # Split proportions for train, validation, test sets

# Load datasets with transformations
_, validation_set, test_set = load_and_preprocess_data(dataset_name,basic_transformations, split_sizes, classes_to_include)
_, validation_set_nt, test_set_nt = load_and_preprocess_data(dataset_name,basic_transformations, split_sizes, classes_to_include)

# Define the input shape for the model, including batch size
input_shape = (1, *dataset_config['input_dim'])  # Prepends batch size of 1

# Initialize the model with the specified configuration and move it to the appropriate device (GPU/CPU)
model = model_config['model'](*model_config['args'](dataset_config, number_of_classes)).to(device)
model.load_state_dict(tr.load(network_location))

# Creating DataLoaders for datasets
validation_loader = DataLoader(CustomTransformDataset(validation_set, transform=additional_transformations, N=number_of_transformations), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(CustomTransformDataset(test_set, transform=additional_transformations, N=number_of_transformations), batch_size=batch_size, shuffle=False)

# DataLoaders for the datasets with only ToTensor transformation
validation_loader_nt = DataLoader(validation_set_nt, batch_size=batch_size, shuffle=False)
test_loader_nt = DataLoader(test_set_nt, batch_size=batch_size, shuffle=False)

# Ensure the 'Data' directory exists before saving data
if not os.path.exists(output_location):
    os.makedirs(output_location)

# Dictionary to store model outputs and labels
all_model_data = {}

model.to(device)
model.eval()  # Set the model to evaluation mode
model_data = {}  # Dictionary to store outputs and labels for this model

# Process validation and test sets
for dataset_name, loader, loader_nt in [('validation', validation_loader, validation_loader_nt),
                                        ('test', test_loader, test_loader_nt)]:
    all_outputs_transformed = []
    all_outputs_original = []
    all_labels = []

    # Process dataset with N transformations
    for inputs, labels in loader:
        # Reshape inputs if they have an extra dimension for transformations
        if inputs.dim() == 5:  # Assuming shape [batch_size, N, channels, height, width]
            batch_size, N, C, H, W = inputs.size()
            inputs = inputs.view(-1, C, H, W)  # Merge batch and N dimensions

        inputs = inputs.to(device)
        labels = labels.to(device)  # No need to repeat labels

        with tr.no_grad():
            outputs = model(inputs)
        # Reshape outputs to [N, set_size, dimension_of_output]
        outputs = outputs.view(-1, N, *outputs.shape[1:])
        all_outputs_transformed.append(outputs)
        all_labels.extend(labels)

    # Concatenate all outputs and labels on GPU and then move to CPU
    final_output_transformed = tr.cat(all_outputs_transformed, dim=0).cpu()  # Concatenate along set_size dimension
    final_labels = tr.tensor(all_labels).cpu()

    # Process dataset with only ToTensor transformation
    for inputs, labels in loader_nt:
        inputs = inputs.to(device)

        with tr.no_grad():
            outputs = model(inputs)
        all_outputs_original.append(outputs.cpu())

    # Concatenate all outputs for original dataset
    final_output_original = tr.cat(all_outputs_original, dim=0)
    # Store outputs and labels in the dictionary
    model_data[dataset_name] = {
        'transformed_outputs': final_output_transformed,
        'original_outputs': final_output_original,
        'labels': final_labels
    }

# Store the model's data in the main dictionary
all_model_data[network_name] = model_data

# Save the entire dictionary of model data
tr.save(all_model_data, output_location + '/all_model_data.pt')