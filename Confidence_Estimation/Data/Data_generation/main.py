import torch as tr
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

# Import necessary modules from Confidence Estimation
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name
from Confidence_Estimation.Data.Data_sets.definitions import CustomTransformDataset
from Confidence_Estimation.Data.Data_sets.functions import load_and_preprocess_data
from Confidence_Estimation.Configurations.definitions import *

# Get the device name for PyTorch operations
device = print_device_name()

# Load datasets with transformations
_, validation_set, test_set = load_and_preprocess_data(DATASET_NAME,BASIC_TRANSFORMATIONS, SPLIT_SIZES, NUMBER_OF_CLASSES)
_, validation_set_nt, test_set_nt = load_and_preprocess_data(DATASET_NAME, BASIC_TRANSFORMATIONS,SPLIT_SIZES, NUMBER_OF_CLASSES)

# Initialize network
if NUMBER_OF_CLASSES is None:
    network = SimpleCNN(CONFIG[DATASET_NAME]['input_dim'], len(CONFIG[DATASET_NAME]['classes']))
else:
    network = SimpleCNN(CONFIG[DATASET_NAME]['input_dim'], min(len(CONFIG[DATASET_NAME]['classes']), NUMBER_OF_CLASSES))

Networks = {
    'SimpleCNN': network
}
# Creating DataLoaders for datasets

validation_loader = DataLoader(CustomTransformDataset(validation_set, transform=ADDITIONAL_TRANSFORMATIONS, N=NUMBER_OF_TRANSFORMATIONS), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(CustomTransformDataset(test_set, transform=ADDITIONAL_TRANSFORMATIONS, N=NUMBER_OF_TRANSFORMATIONS), batch_size=BATCH_SIZE, shuffle=False)

# DataLoaders for the datasets with only ToTensor transformation
validation_loader_nt = DataLoader(validation_set_nt, batch_size=BATCH_SIZE, shuffle=False)
test_loader_nt = DataLoader(test_set_nt, batch_size=BATCH_SIZE, shuffle=False)

# Ensure the 'Data' directory exists before saving data
if not os.path.exists(OUTPUT_LOCATION):
    os.makedirs(OUTPUT_LOCATION)

# Dictionary to store model outputs and labels
all_model_data = {}


for model_name, model in Networks.items():
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
    all_model_data[model_name] = model_data

# Save the entire dictionary of model data
tr.save(all_model_data, OUTPUT_LOCATION + '/all_model_data.pt')