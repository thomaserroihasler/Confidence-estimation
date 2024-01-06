import torch as tr
import torch.nn.functional as F
import os
import sys
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Other.Function_approximators.definitions import KNNGaussianKernel
from Confidence_Estimation.Other.Confidences.definitions import TemperatureScaledConfidence, AveragetemperatureScaledConfidence
from Confidence_Estimation.Main.Train.functions import train_model
from Confidence_Estimation.Other.Measures.definitions import CrossEntropy
from Confidence_Estimation.Configurations.definitions import*
from Confidence_Estimation.Configurations.functions import*
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name, verify_and_create_folder

# Print and return the name of the device (GPU/CPU) being used
device = print_device_name()

# Define parameters

# Assigning all necessary variables
batch_size = VALIDATION_BATCH_SIZE                                # Batch size for training data
calibration_location = CALIBRATION_LOCATION                       # folder location for calibration methods
classes_to_include = Classes_to_consider(NUMBER_OF_CLASSES)       # List of class indices to include in the dataset
criterion = LOSS_FUNCTIONS[TRAINING_LOSS_FUNCTION]                # Loss function to be used during training
dataset_config = CONFIG[DATASET_NAME]                             # Configuration settings for the dataset
dataset_name = DATASET_NAME                                       # Name of the dataset
early_stopping = EARLY_STOPPING                                   # Early stopping criteria for training
learning_rate = VALIDATION_LEARNING_RATE                          # Learning rate for training
model_config = MODELS[NETWORK_NAME]                               # Configuration settings for the model
momentum = VALIDATION_MOMENTUM                                    # Momentum for the optimizer
number_of_classes = NUMBER_OF_CLASSES                             # Number of classes in the dataset
number_of_epochs = VALIDATION_NUMBER_OF_EPOCHS                    # Total number of epochs for training
optimizer = VALIDATION_OPTIMIZER                                  # Optimizer for training
output_location = OUTPUT_LOCATION                                 # Location for saving the model output
output_file_path = OUTPUT_FILE_PATH                               # Save location for the output
split_sizes = SPLIT_SIZES                                         # Split proportions for train, validation, test sets
weight_decay = VALIDATION_WEIGHT_DECAY                              # Weight decay factor for the optimizer
Number_of_nearest_neighbors_normal= VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL
Number_of_nearest_neighbors_transformed = VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED
# Load the model data
all_model_data = tr.load(output_file_path)

# Extract validation data
validation_data = all_model_data[NETWORK_NAME]['validation']

# Transformed outputs
transformed_outputs = validation_data['transformed_outputs']

# Normal outputs
normal_outputs = validation_data['original_outputs']

# Labels
labels = validation_data['labels']

# Initialize temperature scaling models
temp_scaled_model = TemperatureScaledConfidence()
temp_scaled_model.change_behaviour()
avg_temp_scaled_model = AveragetemperatureScaledConfidence()
avg_temp_scaled_model.change_behaviour()

# Apply softmax to the normal and transformed outputs
softmax_normal_outputs = F.softmax(normal_outputs, dim=-1)
softmax_transformed_outputs = F.softmax(transformed_outputs, dim=-1)
softmax_mean_transformed_outputs = tr.mean(softmax_transformed_outputs,dim = 1)

# Calculate the maximum values and indices for the normal and transformed outputs
naive_confidence, max_indices_normal = tr.max(softmax_normal_outputs, dim=-1)

# Compute the validity of each prediction
prediction_validity = (max_indices_normal == labels).int()

# Compute the mean confidence for the transformed outputs along the direction of max_indices_normal
transformed_mean_confidence = softmax_mean_transformed_outputs[tr.arange(softmax_mean_transformed_outputs.size(0)), max_indices_normal]

# Create datasets for training temperature scaling models
labels_and_preds = tr.stack((labels, max_indices_normal), dim=1)
validation_dataset_normal = TensorDataset(normal_outputs, labels_and_preds)
validation_dataset_transformed = TensorDataset(transformed_outputs, labels_and_preds)

# make dataloaders of the different datasets

validation_loader_normal = DataLoader(validation_dataset_normal, batch_size=batch_size, shuffle=False)
validation_loader_transformed = DataLoader(validation_dataset_transformed, batch_size=batch_size, shuffle=False)

# Define loss function and optimizers
loss_fn = CrossEntropy()
optimizer_temp = optim.Adam(temp_scaled_model.parameters(), lr=0.001)
optimizer_avg_temp = optim.Adam(avg_temp_scaled_model.parameters(), lr=0.001)

# Train temperature scaling models

temp_scaled_model = train_model(temp_scaled_model, validation_loader_normal, loss_fn, optimizer_temp,None, number_of_epochs, None, None)
avg_temp_scaled_model = train_model(avg_temp_scaled_model, validation_loader_transformed, loss_fn, optimizer_avg_temp,None, number_of_epochs, None, None)


# Initialize KNNGaussianKernel for normal and transformed outputs with prediction validity
gaussian_kernel_normal = KNNGaussianKernel(naive_confidence, prediction_validity, Number_of_nearest_neighbors_normal)
gaussian_kernel_transformed = KNNGaussianKernel(transformed_mean_confidence, prediction_validity, Number_of_nearest_neighbors_transformed)

# Save the trained temperature scaling models and Gaussian Kernels
verify_and_create_folder(calibration_location)

tr.save(temp_scaled_model.state_dict(), os.path.join(CALIBRATION_LOCATION, 'temperature_scaled_model.pt'))
tr.save(avg_temp_scaled_model.state_dict(), os.path.join(CALIBRATION_LOCATION, 'avg_temperature_scaled_model.pt'))
tr.save(gaussian_kernel_normal, os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_normal.pt'))
tr.save(gaussian_kernel_transformed, os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_transformed.pt'))
