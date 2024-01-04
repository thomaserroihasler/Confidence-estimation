import torch as tr
import sys
from torch.utils.data import DataLoader

from Confidence_Estimation.Networks_and_predictors.Networks.functions import  get_accuracy
from Confidence_Estimation.Data.Data_transformations.definitions import  DynamicDiffeomorphism
from Confidence_Estimation.Data.Data_processing.functions import load_and_preprocess_data
from Confidence_Estimation.Configurations.definitions import *
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name

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
train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=False)

dataset_config = CONFIG[DATASET_NAME]
model_config = MODEL_CONFIG[NETWORK_NAME]
criterion = TRAINING_LOSS_FUNCTION

input_shape = (1, *dataset_config['input_dim'])  # Prepends batch size of 1
model = model_config['model'](*model_config['args'](dataset_config, NUMBER_OF_CLASSES)).to(device)
optimizer = OPTIMIZERS[TRAINING_OPTIMIZER]

# Training Loop

for epoch in range(TRAINING_NUMBER_OF_EPOCHS):
    model.train()  # Ensure model is in training mode
    for i, (images, labels) in enumerate(train_loader):# Training for current epoch
        images, labels = images.to(device), labels.to(device)
        if TRAINING_USE_DIFFEOMORPHISM:  # Apply diffeomorphism transformation if the flag is set
            diffeo_transform = DynamicDiffeomorphism(
                TRAINING_DIFFEOMORPHISM_PARAMS["temperature scale"],
                TRAINING_DIFFEOMORPHISM_PARAMS["c"])
            images = diffeo_transform(images)
        if TRAINING_USE_MIXUP: # Apply mixup if the flag is set
            images, labels_a, labels_b, lam = mixup_data(images, labels, TRAINING_MIXUP_ALPHA, device=device)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodically check the validation accuracy during training
        if (i + 1) % int(len(train_loader) / 3) == 0:
            print(f'Epoch [{epoch + 1}/{TRAINING_NUMBER_OF_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            val_accuracy = get_accuracy(val_loader, model, device)
            print(f'Current validation accuracy: {val_accuracy:.2f}%')

            # Check if validation accuracy is above the threshold
            if val_accuracy >= VALIDATION_ACCURACY_THRESHOLD:
                print(f"Reached desired validation accuracy of {VALIDATION_ACCURACY_THRESHOLD:.2f}%. Stopping training.")
                early_stopping = True  # Set the early stopping flag
                break  # Break out of the batch loop

    if early_stopping:
        break  # Break out of the epoch loop if early stopping was triggered

print("Finished Training")

# Test the model
model.eval()
with tr.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = tr.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

# Save Model
save_path = f'./Networks/{NETWORK_NAME.lower()}_{DATASET_NAME.lower()}.pth'
print('Network saved location is', save_path)
tr.save(model.state_dict(), save_path)