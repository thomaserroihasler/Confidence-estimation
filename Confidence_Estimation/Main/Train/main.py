# Import necessary libraries
import torch as tr
import sys
from torch.utils.data import DataLoader
from functions import train_model

# Update the system path to include the directory for Confidence Estimation
# This ensures that Python can find and import modules from this directory
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

# Import specific functions and configurations from the Confidence Estimation module
from Confidence_Estimation.Data.Data_processing.functions import load_and_preprocess_data
from Confidence_Estimation.Configurations.definitions import *
from Confidence_Estimation.Configurations.functions import *
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name

# Print and return the name of the device (GPU/CPU) being used
device = print_device_name()

# Initialize a list of classes to be included in the model
# If a specific number of classes is defined, use that range; otherwise, use None
if NUMBER_OF_CLASSES:
    classes_to_include  = list(range(0,NUMBER_OF_CLASSES))
else:
    classes_to_include = None

# Load and preprocess the datasets (training, validation, testing) based on the specified dataset name and classes
train_dataset, val_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME,classes_to_include)

# Create data loaders for each dataset with the specified batch size and shuffle settings
batch_size = TRAINING_BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Retrieve dataset and model configurations based on specified names
dataset_config = CONFIG[DATASET_NAME]
model_config = MODELS[NETWORK_NAME]

# Define the input shape for the model, including batch size
input_shape = (1, *dataset_config['input_dim'])  # Prepends batch size of 1

# Initialize the model with the specified configuration and move it to the appropriate device (GPU/CPU)
model = model_config['model'](*model_config['args'](dataset_config, NUMBER_OF_CLASSES)).to(device)

# Training Loop and configurations

# Set up the loss function, optimizer, and other training parameters
criterion = TRAINING_LOSS_FUNCTION
optimizer = create_optimizer(TRAINING_OPTIMIZER,TRAINING_LEARNING_RATE,TRAINING_MOMENTUM,TRAINING_WEIGHT_DECAY)
parallel_transformations = TRAINING_PARALLEL_TRANSFORMATIONS
number_of_epochs = TRAINING_NUMBER_OF_EPOCHS
early_stopping = EARLY_STOPPING

# Train the model using the specified parameters and datasets
model = train_model(model,train_loader,val_loader,criterion,optimizer,parallel_transformations,number_of_epochs,early_stopping)

# Indicate that training is complete
print("Finished Training")

# Save the trained model to a specified path
save_path = f'./Networks/{NETWORK_NAME.lower()}_{DATASET_NAME.lower()}.pth'
print('Network saved location is', save_path)
tr.save(model.state_dict(), save_path)
