import torch as tr
import sys
from torch.utils.data import DataLoader
from Confidence_Estimation.Data.Data_processing.functions import load_and_preprocess_data
from Confidence_Estimation.Configurations.Configurations import *
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name
from functions import train_model
# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path
device = print_device_name()

# creates a list of all classes to consider
if NUMBER_OF_CLASSES:
    classes_to_include  = list(range(0,NUMBER_OF_CLASSES))
else:
    classes_to_include = None

# extracts the relevant datasets
train_dataset, val_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME,classes_to_include)

# Data Loaders
batch_size = TRAINING_BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataset_config = CONFIG[DATASET_NAME]
model_config = MODEL_CONFIG[NETWORK_NAME]

input_shape = (1, *dataset_config['input_dim'])  # Prepends batch size of 1
model = model_config['model'](*model_config['args'](dataset_config, NUMBER_OF_CLASSES)).to(device)

# Training Loop and configurations

criterion = TRAINING_LOSS_FUNCTION
optimizer = OPTIMIZERS[TRAINING_OPTIMIZER]
parallel_transformations = TRAINING_PARALLEL_TRANSFORMATIONS
number_of_epochs = TRAINING_NUMBER_OF_EPOCHS
early_stopping = EARLY_STOPPING

model = train_model(model,train_loader,val_loader,criterion,parallel_transformations,number_of_epochs,early_stopping)

print("Finished Training")

save_path = f'./Networks/{NETWORK_NAME.lower()}_{DATASET_NAME.lower()}.pth'
print('Network saved location is', save_path)
tr.save(model.state_dict(), save_path)