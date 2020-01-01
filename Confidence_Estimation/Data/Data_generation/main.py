import torch as tr
import os
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

# Import necessary modules from Confidence Estimation
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name
from Confidence_Estimation.Data.Data_sets.definitions import CustomTransformDataset
from Confidence_Estimation.Data.Data_sets.functions import load_and_preprocess_data
from Confidence_Estimation.Data.Data_sets.configurations import CONFIG
from Confidence_Estimation.Data.Data_visualization.functions import plot_sample_images
from Confidence_Estimation.Networks_and_predictors.Networks.definitions import SimpleCNN

# Get the device name for PyTorch operations
device = print_device_name()

# Dataset and Output Configuration
DATASET_NAME = 'MNIST'
OUTPUT_LOCATION = '../../../Data/Outputs/' + DATASET_NAME
NUMBER_OF_CLASSES = None

# Split sizes for non-standard datasets
SPLIT_SIZES = {
    "train": 0.6,  # 60% data for training
    "validation": 0.2,  # 20% data for validation
    "test": 0.2  # 20% data for testing
}

BASIC_TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor()
])

# Define transformations
TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Define minimal ToTensor transformation

# Load datasets with transformations
_, validation_set, test_set = load_and_preprocess_data(DATASET_NAME, TRANSFORMATIONS, SPLIT_SIZES, NUMBER_OF_CLASSES)

# Load datasets with only ToTensor transformation
_, validation_set_nt, test_set_nt = load_and_preprocess_data(DATASET_NAME, BASIC_TRANSFORMATIONS, SPLIT_SIZES, NUMBER_OF_CLASSES)

# Define batch size
BATCH_SIZE = 32

# Initialize network
if NUMBER_OF_CLASSES is None:
    network = SimpleCNN(CONFIG[DATASET_NAME]['input_dim'], len(CONFIG[DATASET_NAME]['classes']))
else:
    network = SimpleCNN(CONFIG[DATASET_NAME]['input_dim'], min(len(CONFIG[DATASET_NAME]['classes']), NUMBER_OF_CLASSES))

Networks = {
    'SimpleCNN': network
}

# Creating DataLoaders for datasets
N = 10  # Number of transformations per image
validation_loader = DataLoader(CustomTransformDataset(validation_set, transform=TRANSFORMATIONS, N=N), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(CustomTransformDataset(test_set, transform=TRANSFORMATIONS, N=N), batch_size=BATCH_SIZE, shuffle=False)

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
        for batch in loader:
            inputs, labels = batch
            # CONVERT INPUTS TO TENSORS & PARRALELIZE
            #plot_sample_images(inputs, labels, 'test')
            inputs = [item.to(device) for sublist in inputs for item in sublist]  # Flatten and send to device
            labels = labels.to(device).repeat(N)  # Repeat labels N times

            with tr.no_grad():
                outputs = [model(input_tensor.unsqueeze(0)) for input_tensor in inputs]  # Process each transformed input
            all_outputs_transformed.extend(outputs)
            all_labels.extend([label for label in labels])

        # Process dataset with only ToTensor transformation
        for inputs, labels in loader_nt:
            inputs = inputs.to(device)

            with tr.no_grad():
                outputs = model(inputs)
            all_outputs_original.append(outputs.cpu())

        # Concatenate all outputs and labels
        final_output_transformed = tr.cat([output.cpu() for output in all_outputs_transformed], dim=0)
        final_output_original = tr.cat(all_outputs_original, dim=0)
        final_labels = tr.tensor(all_labels)

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