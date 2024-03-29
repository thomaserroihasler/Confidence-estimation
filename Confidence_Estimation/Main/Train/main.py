# Import necessary libraries
import torch as tr
import sys
from torch.utils.data import DataLoader
from functions import train_model

# Update the system path to include the directory for Confidence Estimation
# This ensures that Python can find and import modules from this directory
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

# Import specific functions and configurations from the Confidence_Estimation module
from Confidence_Estimation.Data.Data_sets.functions import load_and_preprocess_data
from Confidence_Estimation.Configurations.definitions import *
from Confidence_Estimation.Configurations.functions import *
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name, verify_and_create_folder
from Confidence_Estimation.Main.Test.functions import test_model
from Confidence_Estimation.Other.Measures.definitions import TrueFalseMeasure
from Confidence_Estimation.Networks_and_predictors.Predictors.definitions import MaximalLogitPredictor
# Print and return the name of the device (GPU/CPU) being used
device = print_device_name()

# Assigning all necessary variables
additional_transformations = TRAINING_ADDITIONAL_TRANSFORMATIONS  # Additional transformations for training data
basic_transformations = BASIC_TRANSFORMATIONS                     # Basic transformations for the dataset
batch_size = TRAINING_BATCH_SIZE                                  # Batch size for training data
classes_to_include = Classes_to_consider(NUMBER_OF_CLASSES)       # List of class indices to include in the dataset
criterion = LOSS_FUNCTIONS[TRAINING_LOSS_FUNCTION]                # Loss function to be used during training
dataset_config = CONFIG[DATASET_NAME]                             # Configuration settings for the dataset
dataset_name = DATASET_NAME                                       # Name of the dataset
early_stopping = EARLY_STOPPING                                   # Early stopping criteria for training
learning_rate = TRAINING_LEARNING_RATE                            # Learning rate for training
max_logit_predictor = MaximalLogitPredictor()                 # prediction of the network
model_config = MODELS[NETWORK_NAME]                               # Configuration settings for the model
momentum = TRAINING_MOMENTUM                                      # Momentum for the optimizer
network_location = NETWORK_LOCATION                               # Location for saving/loading the network model
network_name = NETWORK_NAME                                       # Name of the network (model)
number_of_classes = NUMBER_OF_CLASSES                             # Number of classes in the dataset
number_of_epochs = TRAINING_NUMBER_OF_EPOCHS                      # Total number of epochs for training
optimizer = TRAINING_OPTIMIZER                                    # Optimizer for training
output_location = OUTPUT_LOCATION                                 # Location for saving the model output
parallel_transformations = TRAINING_PARALLEL_TRANSFORMATIONS      # Parallel transformations to be applied during training
prediction_criterion = TrueFalseMeasure()                         # measure for the prediction of the network (accuracy)
network_file_path = NETWORK_FILE_PATH                             # Save path for the network dictionary
split_sizes = SPLIT_SIZES                                         # Split proportions for train, validation, test sets
weight_decay = TRAINING_WEIGHT_DECAY                              # Weight decay factor for the optimizer

# Load and preprocess the datasets (training, validation, testing) based on the specified dataset name and classes
train_dataset, val_dataset, test_dataset = load_and_preprocess_data(dataset_name,basic_transformations,split_sizes,classes_to_include)

# Create data loaders for each dataset with the specified batch size and shuffle settings
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the input shape for the model, including batch size
input_shape = (1, *dataset_config['input_dim'])  # Prepends batch size of 1

# Initialize the model with the specified configuration and move it to the appropriate device (GPU/CPU)
if number_of_classes == None:
    number_of_classes = len(dataset_config['classes'])

model = model_config['model'](*model_config['args'](dataset_config, number_of_classes)).to(device)

# Train the model using the specified parameters and datasets
optimizer = create_optimizer(optimizer,model.parameters(), learning_rate, momentum, weight_decay)  # Optimizer for training

model = train_model(model,train_loader,criterion,optimizer,device,parallel_transformations,number_of_epochs,val_loader,None)

test_cross_entropy, accuracy = test_model(model, test_loader, criterion, max_logit_predictor,device, prediction_criterion, batch_size)
# generate the appropriate folder
verify_and_create_folder(network_location)

#Save the trained model to a specified path
print('Network saved location is', network_file_path)
tr.save(model.state_dict(), network_file_path)