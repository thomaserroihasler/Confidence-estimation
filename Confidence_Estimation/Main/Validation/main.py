import torch as tr
import torch.nn.functional as F
import os
import sys
from torch.utils.data import TensorDataset

# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Other.Function_approximators.definitions import KNNGaussianKernel
from Confidence_Estimation.Other.Confidences.definitions import TemperatureScaledConfidence, AveragetemperatureScaledConfidence
from Confidence_Estimation.Main.Train.functions import train_model
from Confidence_Estimation.Other.Measures.definitions import CrossEntropy
from Confidence_Estimation.Configurations.definitions import*
from Confidence_Estimation.Configurations.functions import*

# Define parameters
Number_of_nearest_neighbors_normal = VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL
Number_of_nearest_neighbors_transformed = VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED
Number_of_epochs = VALIDATION_NUMBER_OF_EPOCHS
Batch_size = VALIDATION_BATCH_SIZE

# Load the model data
all_model_data = tr.load(OUTPUT_LOCATION + '/all_model_data.pt')

# Extract validation data
validation_data = all_model_data['SimpleCNN']['validation']

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
train_dataset_normal = TensorDataset(normal_outputs, labels_and_preds)
train_dataset_transformed = TensorDataset(transformed_outputs, labels_and_preds)

# Define loss function and optimizers
loss_fn = CrossEntropy()
optimizer_temp = optim.Adam(temp_scaled_model.parameters(), lr=0.001)
optimizer_avg_temp = optim.Adam(avg_temp_scaled_model.parameters(), lr=0.001)

# Train temperature scaling models
optimizer_temp = create_optimizer(TRAINING_OPTIMIZER,TRAINING_LEARNING_RATE,TRAINING_MOMENTUM,TRAINING_WEIGHT_DECAY)
optimizer_avg_temp = create_optimizer(TRAINING_OPTIMIZER,TRAINING_LEARNING_RATE,TRAINING_MOMENTUM,TRAINING_WEIGHT_DECAY)
temp_scaled_model = train_model(temp_scaled_model, train_dataset_normal, loss_fn, optimizer_temp, epochs=Number_of_epochs, batch_size=Batch_size)
avg_temp_scaled_model = train_model(avg_temp_scaled_model, train_dataset_transformed, loss_fn, optimizer_avg_temp,  epochs=Number_of_epochs, batch_size=Batch_size)

# Initialize KNNGaussianKernel for normal and transformed outputs with prediction validity
gaussian_kernel_normal = KNNGaussianKernel(softmax_normal_outputs, prediction_validity, Number_of_nearest_neighbors_normal)
gaussian_kernel_transformed = KNNGaussianKernel(softmax_mean_transformed_outputs, prediction_validity, Number_of_nearest_neighbors_transformed)

# Save the trained temperature scaling models and Gaussian Kernels
if not os.path.exists(CALIBRATION_LOCATION):
    os.makedirs(CALIBRATION_LOCATION)
tr.save(temp_scaled_model.state_dict(), os.path.join(CALIBRATION_LOCATION, 'temperature_scaled_model.pt'))
tr.save(avg_temp_scaled_model.state_dict(), os.path.join(CALIBRATION_LOCATION, 'avg_temperature_scaled_model.pt'))
tr.save(gaussian_kernel_normal, os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_normal.pt'))
tr.save(gaussian_kernel_transformed, os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_transformed.pt'))
