import numpy as np
import matplotlib.pyplot as plt
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
from Confidence_Estimation.Other.Measures.definitions import*
from Confidence_Estimation.Configurations.definitions import*
from Confidence_Estimation.Configurations.functions import*
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name, verify_and_create_folder

# Print and return the name of the device (GPU/CPU) being used
device = print_device_name()

# Define parameters

# Assigning all necessary variables
batch_size = VALIDATION_BATCH_SIZE                                # Batch size for training data
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
weight_decay = VALIDATION_WEIGHT_DECAY                            # Weight decay factor for the optimizer
Number_of_nearest_neighbors_normal= VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL
Number_of_nearest_neighbors_transformed = VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED
# Load the model data
all_model_data = tr.load(output_file_path)

# Extract validation data
test_data = all_model_data[NETWORK_NAME]['test']

# Transformed outputs
transformed_outputs = test_data['transformed_outputs']

# Normal outputs
normal_outputs = test_data['original_outputs']

# Labels
labels = test_data['labels']
# Load the model data

# Apply softmax to the transformed outputs along the last dimension
softmax_transformed_outputs = F.softmax(transformed_outputs, dim=-1)

# Calculate the mean of the softmax-transformed outputs along the dimension of transformations
mean_softmax_transformed_outputs = softmax_transformed_outputs.mean(dim=1)

# Apply softmax to the normal outputs along the last dimension
softmax_normal_outputs = F.softmax(normal_outputs, dim=-1)

# Calculate the maximum values and indices for the normal outputs
_, max_indices_normal = tr.max(softmax_normal_outputs, dim=-1)

# Compute the validity of each prediction (1 for correct, 0 for incorrect)
prediction_validity = (max_indices_normal == labels).int()

selected_mean_softmax_transformed_outputs = mean_softmax_transformed_outputs[tr.arange(mean_softmax_transformed_outputs.size(0)), max_indices_normal]
# Similarly, select the corresponding maximum values from the normal outputs
selected_softmax_normal_outputs = softmax_normal_outputs[tr.arange(softmax_normal_outputs.size(0)), max_indices_normal]

# plot on the x axis the selected outputs and the validity of prediction as a y axis
plt.figure(figsize=(12, 6))

# Plot for normal outputs
plt.subplot(1, 2, 1)
plt.scatter(selected_softmax_normal_outputs.numpy(), prediction_validity.numpy(), alpha=0.5)
plt.title('Prediction Validity vs Selected Normal Outputs')
plt.xlabel('Selected Output Confidence Scores')
plt.ylabel('Prediction Validity (1 = Correct, 0 = Incorrect)')
plt.grid(True)

# Plot for transformed outputs
plt.subplot(1, 2, 2)
plt.scatter(selected_mean_softmax_transformed_outputs.numpy(), prediction_validity.numpy(), alpha=0.5)
plt.title('Prediction Validity vs Selected Transformed Outputs')
plt.xlabel('Selected Output Confidence Scores')
plt.ylabel('Prediction Validity (1 = Correct, 0 = Incorrect)')
plt.grid(True)

plt.tight_layout()
plt.show()
# Load the Gaussian Kernels
kernel_normal = tr.load(os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_normal.pt'))
kernel_transformed = tr.load(os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_transformed.pt'))

# plot the kernel functions
input_values = np.linspace(0, 1, 100)

# Apply the Gaussian kernels to these input values
output_values_normal = kernel_normal(tr.tensor(input_values, dtype=tr.float32)).detach().numpy()
output_values_transformed = kernel_transformed(tr.tensor(input_values, dtype=tr.float32)).detach().numpy()

# Plot the kernel functions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(input_values, output_values_normal, label='Normal Kernel')
plt.xlabel('Input Values')
plt.ylabel('Output Values')
plt.title('Normal Gaussian Kernel')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(input_values, output_values_transformed, label='Transformed Kernel')
plt.xlabel('Input Values')
plt.ylabel('Output Values')
plt.title('Transformed Gaussian Kernel')
plt.legend()

plt.suptitle('Gaussian Kernel Functions')
plt.show()

# Apply the kernels to the test data
test_scores_normal = kernel_normal(selected_softmax_normal_outputs).squeeze()
test_scores_transformed = kernel_transformed(selected_mean_softmax_transformed_outputs).squeeze()

# Calculate performance metrics for normal kernel
ece_normal = ECE()(test_scores_normal, prediction_validity)
mie_normal = MIE()(test_scores_normal, prediction_validity)
_,_,aurc_normal = AURC(test_scores_normal, prediction_validity)

# Calculate performance metrics for transformed kernel
ece_transformed = ECE()(test_scores_transformed, prediction_validity)
mie_transformed = MIE()(test_scores_transformed, prediction_validity)
_, _ , aurc_transformed = AURC(test_scores_transformed, prediction_validity)

# Print or store the resultsd
print("Normal Kernel Metrics:")
print("ECE:", ece_normal.item())
print("MIE:", mie_normal.item())
print("AURC:", aurc_normal.item())

print("\nTransformed Kernel Metrics:")
print("ECE:", ece_transformed.item())
print("MIE:", mie_transformed.item())
print("AURC:", aurc_transformed.item())
