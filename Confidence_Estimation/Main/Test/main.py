import torch as tr
import os
import sys
import torch.nn.functional as F

# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Other.Measures.definitions import AURC, ECE, MIE
from Confidence_Estimation.Configurations.definitions import *

# Load the model data
all_model_data = tr.load(OUTPUT_LOCATION + '/all_model_data.pt')

# Extract test data
test_data = all_model_data['SimpleCNN']['test']

# Transformed outputs
transformed_outputs = test_data['transformed_outputs']

# Apply softmax to the transformed outputs along the last dimension
softmax_transformed_outputs = F.softmax(transformed_outputs, dim=-1)

# Calculate the mean of the softmax-transformed outputs along the dimension of transformations
mean_softmax_transformed_outputs = softmax_transformed_outputs.mean(dim=1)

# Normal outputs (assuming these are the 'original_outputs' in your data structure)
normal_outputs = test_data['original_outputs']

# Apply softmax to the normal outputs along the last dimension
softmax_normal_outputs = F.softmax(normal_outputs, dim=-1)

# Calculate the maximum values and indices for the normal outputs
_, max_indices_normal = tr.max(softmax_normal_outputs, dim=-1)

# Labels (assuming they are the same for both normal and transformed outputs)
labels = test_data['labels']

# Compute the validity of each prediction (1 for correct, 0 for incorrect)
prediction_validity = (max_indices_normal == labels).int()

selected_mean_softmax_transformed_outputs = mean_softmax_transformed_outputs[tr.arange(mean_softmax_transformed_outputs.size(0)), max_indices_normal]
# Similarly, select the corresponding maximum values from the normal outputs
selected_softmax_normal_outputs = softmax_normal_outputs[tr.arange(softmax_normal_outputs.size(0)), max_indices_normal]

# Load the Gaussian Kernels
kernel_normal = tr.load(os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_normal.pt'))
kernel_transformed = tr.load(os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_transformed.pt'))

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
