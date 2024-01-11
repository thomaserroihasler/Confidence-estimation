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
from Confidence_Estimation.Other.Measures.definitions import AURC as aurc
from Confidence_Estimation.Other.Measures.definitions import *
from Confidence_Estimation.Configurations.definitions import *
from Confidence_Estimation.Configurations.functions import *
from Confidence_Estimation.Other.Useful_functions.definitions import *
from Confidence_Estimation.Data.Data_visualization.functions import *

# Print and return the name of the device (GPU/CPU) being used
device = print_device_name()

# Define parameters
batch_size = VALIDATION_BATCH_SIZE
binning_strategy = TEST_BINNING
classes_to_include = Classes_to_consider(NUMBER_OF_CLASSES)
criterion = LOSS_FUNCTIONS[TRAINING_LOSS_FUNCTION]
dataset_config = CONFIG[DATASET_NAME]
dataset_name = DATASET_NAME
early_stopping = EARLY_STOPPING
ece = ECE(**binning_strategy)
final_data_location = FINAL_DATA_LOCATION
learning_rate = VALIDATION_LEARNING_RATE
model_config = MODELS[NETWORK_NAME]
momentum = VALIDATION_MOMENTUM
number_of_classes = NUMBER_OF_CLASSES
optimizer = VALIDATION_OPTIMIZER
output_location = OUTPUT_LOCATION
output_file_path = OUTPUT_FILE_PATH
split_sizes = SPLIT_SIZES
weight_decay = VALIDATION_WEIGHT_DECAY
Number_of_nearest_neighbors_normal = VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL
Number_of_nearest_neighbors_transformed = VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED
mie = MIE(**binning_strategy)

metrics = {
    "normal": {},
    "transformed": {},
    "normal kernel": {},
    "transformed kernel": {},
    "temp_scaled_normal": {},
    "temp_scaled_transformed": {}
}

# Load the model data and ensure tensors are on the correct device
all_model_data = tr.load(output_file_path, map_location=device)

# Extract validation data and move tensors to the device
test_data = all_model_data[NETWORK_NAME]['test']
for k, v in test_data.items():
    if isinstance(v, tr.Tensor):
        test_data[k] = v.to(device)

# Transformed and normal outputs, labels
transformed_outputs = test_data['transformed_outputs']
normal_outputs = test_data['original_outputs']
labels = test_data['labels']

# Softmax and processing of outputs
softmax_transformed_outputs = F.softmax(transformed_outputs, dim=-1)
print(softmax_transformed_outputs[0],normal_outputs[0])
mean_softmax_transformed_outputs = softmax_transformed_outputs.mean(dim=1)
softmax_normal_outputs = F.softmax(normal_outputs, dim=-1)
print(softmax_transformed_outputs[0],softmax_normal_outputs[0])
_, max_indices_normal = tr.max(softmax_normal_outputs, dim=-1)
print(max_indices_normal[0])
prediction_validity = (max_indices_normal == labels).int()
print((max_indices_normal == labels).float().mean())

# Select relevant softmax outputs
selected_mean_softmax_transformed_outputs = mean_softmax_transformed_outputs[tr.arange(mean_softmax_transformed_outputs.size(0)), max_indices_normal]
selected_softmax_normal_outputs = softmax_normal_outputs[tr.arange(softmax_normal_outputs.size(0)), max_indices_normal]
print(selected_mean_softmax_transformed_outputs[0],selected_softmax_normal_outputs[0])
# Load and move Gaussian Kernels to the device

ece_normal = ece(selected_softmax_normal_outputs, prediction_validity)
mie_normal = mie(selected_softmax_normal_outputs, prediction_validity)
x2, y2, aurc_normal = aurc(selected_softmax_normal_outputs, prediction_validity)

ece_transformed = ece(selected_mean_softmax_transformed_outputs, prediction_validity)
mie_transformed = mie(selected_mean_softmax_transformed_outputs, prediction_validity)
x2, y2, aurc_transformed = aurc(selected_mean_softmax_transformed_outputs, prediction_validity)


kernel_normal = tr.load(os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_normal.pt'), map_location=device)
kernel_transformed = tr.load(os.path.join(CALIBRATION_LOCATION, 'gaussian_kernel_transformed.pt'), map_location=device)

# Apply kernels to the test data
test_scores_normal_kernel = kernel_normal(selected_softmax_normal_outputs)
test_scores_transformed_kernel = kernel_transformed(selected_mean_softmax_transformed_outputs)

# Binning and reliability diagrams

normal_bins = Bin_edges(selected_softmax_normal_outputs, **binning_strategy)
transformed_bins = Bin_edges(selected_mean_softmax_transformed_outputs, **binning_strategy)

normal_kernel_bins = Bin_edges(test_scores_normal_kernel, **binning_strategy)
transformed_kernel_bins = Bin_edges(test_scores_transformed_kernel, **binning_strategy)

reliability_diagram(normal_kernel_bins, test_scores_normal_kernel, prediction_validity,'normal','normal.png')
reliability_diagram(transformed_kernel_bins, test_scores_transformed_kernel, prediction_validity,'transformed','transformed.png')

# Calculate performance metrics
ece_normal_kernel = ece(test_scores_normal_kernel, prediction_validity)
mie_normal_kernel = mie(test_scores_normal_kernel, prediction_validity)
x1, y1, aurc_normal_kernel = aurc(test_scores_normal_kernel, prediction_validity)

# Performance metrics for transformed kernel
ece_transformed_kernel = ece(test_scores_transformed_kernel, prediction_validity)
mie_transformed_kernel = mie(test_scores_transformed_kernel, prediction_validity)
x2, y2, aurc_transformed_kernel = aurc(test_scores_transformed_kernel, prediction_validity)

plt.figure(figsize=(10, 6))

# Plot the first curve
plt.plot(x1, y1, label='Curve 1', color='blue', marker='o')

# Plot the second curve
plt.plot(x2, y2, label='Curve 2', color='red', marker='x')

# Add labels, title, and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Two Curves Comparison')
plt.legend()

# Save the figure
plt.savefig('two_curves_plotafter.png')

# Show the figure
plt.show()
plt.close()
# # Temperature scaling models
# temp_scaled_model = TemperatureScaledConfidence()
# temp_scaled_model.load_state_dict(tr.load(os.path.join(CALIBRATION_LOCATION, 'temperature_scaled_model.pt'), map_location=device))
# temp_scaled_model.to(device)
# avg_temp_scaled_model = AveragetemperatureScaledConfidence()
# avg_temp_scaled_model.load_state_dict(tr.load(os.path.join(CALIBRATION_LOCATION, 'avg_temperature_scaled_model.pt'), map_location=device))
# avg_temp_scaled_model.to(device)
#
# # Apply temperature scaling and calculate metrics
# with tr.no_grad():
#     temp_scaled_normal_confidences = temp_scaled_model(normal_outputs)
#     temp_scaled_transformed_confidences = avg_temp_scaled_model(transformed_outputs)
#
# normal_bins = Bin_edges(temp_scaled_normal_confidences, **binning_strategy)
# transformed_bins = Bin_edges(temp_scaled_transformed_confidences, **binning_strategy)
#
# # reliability_diagram(normal_bins, temp_scaled_normal_confidences, prediction_validity)
# # reliability_diagram(transformed_bins, temp_scaled_transformed_confidences, prediction_validity)
#
# ece_temp_scaled_normal = ece(temp_scaled_normal_confidences, prediction_validity)
# mie_temp_scaled_normal = mie(temp_scaled_normal_confidences, prediction_validity)
# _, _, aurc_temp_scaled_normal = aurc(temp_scaled_normal_confidences, prediction_validity)
#
# ece_temp_scaled_transformed = ece(temp_scaled_transformed_confidences, prediction_validity)
# mie_temp_scaled_transformed = mie(temp_scaled_transformed_confidences, prediction_validity)
# _, _, aurc_temp_scaled_transformed = aurc(temp_scaled_transformed_confidences, prediction_validity)
#
# # Store metrics

metrics["normal"]['ECE'] = ece_normal.item()
metrics["normal"]['MIE'] = mie_normal.item()
metrics["normal"]['AURC'] = aurc_normal.item()

metrics["transformed"]['ECE'] = ece_transformed.item()
metrics["transformed"]['MIE'] = mie_transformed.item()
metrics["transformed"]['AURC'] = aurc_transformed.item()

metrics["normal kernel"]['ECE'] = ece_normal_kernel.item()
metrics["normal kernel"]['MIE'] = mie_normal_kernel.item()
metrics["normal kernel"]['AURC'] = aurc_normal_kernel.item()

metrics["transformed kernel"]['ECE'] = ece_transformed_kernel.item()
metrics["transformed kernel"]['MIE'] = mie_transformed_kernel.item()
metrics["transformed kernel"]['AURC'] = aurc_transformed_kernel.item()

print("the metrics are: ")
print(metrics)

# Save the metrics dictionary
verify_and_create_folder(final_data_location)
metrics_save_path = os.path.join(final_data_location, "performance_metrics.pth")
tr.save(metrics, metrics_save_path)
print("Metrics saved to:", metrics_save_path)
