import torch as tr
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset

from Confidence_Estimation.Data.Data_sets.definitions import CustomImageDataset
from Confidence_Estimation.Networks_and_predictors.Networks.definitions import VGG, ResNet18, SimpleCNN # Assuming this import from Networks.py
from Confidence_Estimation.Networks_and_predictors.Networks.functions import  get_accuracy
from Confidence_Estimation.Data.Data_transformations.definitions import  DynamicDiffeomorphism
from Confidence_Estimation.Other.Useful_functions.definitions import print_device_name, plot_sample_images
from Confidence_Estimation.Other.Measures.definitions import LabelSmoothedCrossEntropyLoss

device = print_device_name()

LOSS_FUNCTIONS = {
    'label_smoothed_crossentropy': LabelSmoothedCrossEntropyLoss(epsilon=0.1),
    'cross_entropy': CrossEntropyLoss(),
    'mse': MSELoss()  # Not typically used for classification
}

H_FLIP = False
V_FLIP = False
RANDOM_CROP = False
COLOR_JITTER = False
ROTATION = False  # New Flag for rotation
ROTATION_DEGREES = (-180, 180)  # Define the range for rotation
USE_MIXUP = False  # Add a flag to turn on or off mixup
USE_DIFFEOMORPHISM = False  # Flag to turn on or off the diffeomorphism transformations
TEMPERATURE_SCALE = 1.0
DIFFEOMORPHISM_DEGREES_OF_FREEDOM = 5

DIFFEOMORPHISM_PARAMS = {
    "temperature scale": 1,  # Initialize with desired value for temperature
    "c": 5               # Initialize with desired value for c
}

def get_transforms(dataset_name):
    base_transforms = []

    if dataset_name == 'MNIST':
        if H_FLIP:
            base_transforms.append(transforms.RandomHorizontalFlip())
        if RANDOM_CROP:
            base_transforms.append(transforms.RandomCrop(28, padding=4))
        if ROTATION:
            base_transforms.append(transforms.RandomRotation(ROTATION_DEGREES))
        base_transforms.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif dataset_name in ['CIFAR-10', 'CIFAR-100']:
        if H_FLIP:
            base_transforms.append(transforms.RandomHorizontalFlip())
        if V_FLIP:
            base_transforms.append(transforms.RandomVerticalFlip())
        if RANDOM_CROP:
            base_transforms.append(transforms.RandomCrop(32, padding=4))
        if COLOR_JITTER:
            base_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        if ROTATION:
            base_transforms.append(transforms.RandomRotation(ROTATION_DEGREES))
        base_transforms.extend(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    elif dataset_name == 'HAM-10000':
        base_transforms.extend([transforms.Resize((128, 128))])  # Original rotation for HAM-10000
        if H_FLIP:
            base_transforms.append(transforms.RandomHorizontalFlip())
        if V_FLIP:
            base_transforms.append(transforms.RandomVerticalFlip())
        if ROTATION:
            base_transforms.append(transforms.RandomRotation(ROTATION_DEGREES))
        base_transforms.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    return base_transforms
# Configuration
CONFIG = {
    'MNIST': {
        'loader': datasets.MNIST,
        'transforms': get_transforms('MNIST'),
        'input_dim': (1, 28, 28),
        'path': './data/MNIST',
        'classes': list(range(10))
    },
    'HAM-10000': {
        'loader': CustomImageDataset,
        'transforms': get_transforms('HAM-10000'),
        'input_dim': (3, 128, 128),
        'path': './Dataset/HAM10000_combined/images',
        'label_path': './Dataset/HAM10000_combined/Labels.txt',
        'classes': list(range(7))
    },
    'CIFAR-10': {
        'loader': datasets.CIFAR10,
        'transforms': get_transforms('CIFAR-10'),
        'input_dim': (3, 32, 32),
        'path': './data/CIFAR10',
        'classes': list(range(10))
    },
    'CIFAR-100': {
        'loader': datasets.CIFAR100,
        'transforms': get_transforms('CIFAR-100'),
        'input_dim': (3, 32, 32),
        'path': './data/CIFAR100',
        'classes': list(range(100))
    },
}

# Model Initialization
MODEL_CONFIG = {
    'VGG': {
        'model': VGG,
        'args': lambda config, n: ('VGG11', config['input_dim'][0], n, config['input_dim'])
    },
    'ResNet18': {
        'model': ResNet18,
        'args': lambda config, n: (config['input_dim'][0], n)
    },
    'SimpleCNN': {
        'model': SimpleCNN,
        'args': lambda config, n: (config['input_dim'], n)
    }
}

def filter_classes(dataset, classes_to_include, dataset_name):
    if dataset_name != 'HAM-10000':
        indices = [i for i in range(len(dataset)) if dataset.targets[i] in classes_to_include]
    else:
        indices = [i for i in range(len(dataset)) if dataset.dataset.labels[i] in classes_to_include]
    return Subset(dataset, indices)

def load_and_preprocess_data(dataset_name, classes_to_include=None):
    transform = transforms.Compose(CONFIG[dataset_name]['transforms'])
    test_and_val_size = 0
    if dataset_name != 'HAM-10000':
        train_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=True, download=True,transform=transform)
        test_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=False, download=True, transform=transform)
        test_dataset_size = len(test_dataset)

    else:
        # For HAM-10000 custom dataset
        full_dataset = CONFIG[dataset_name]['loader'](CONFIG[dataset_name]['path'], CONFIG[dataset_name]['label_path'], transform=transform)
        total_size = len(full_dataset)
        train_size = total_size - int(0.3 * total_size)  # Remaining 70% for training
        test_dataset_size = total_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_dataset_size])

    # If there are specific classes to include, filter datasets
    if classes_to_include:
        train_dataset = filter_classes(train_dataset, classes_to_include, dataset_name)
        test_dataset = filter_classes(test_dataset, classes_to_include, dataset_name)
        test_dataset_size = len(test_dataset)
        val_size = test_dataset_size // 2
        test_size = test_dataset_size - val_size

    val_size = test_dataset_size // 2
    test_size = test_dataset_size - val_size
    print("val_size:", val_size)
    print("test_size:", test_size)
    print("test_dataset_size (after split):", len(test_dataset))
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

DATASET_NAME = 'MNIST'

MODEL_NAME = 'SimpleCNN'

LOSS_NAME = 'cross_entropy'  # label_smoothed_crossentropy or 'cross_entropy', 'mse'

N = 10

classes_to_include  = list(range(0,N)) # None

train_dataset, val_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME,classes_to_include)

# Data Loaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# plot_sample_images(train_loader, "Train Dataset Sample Images")
# plot_sample_images(val_loader, "Validation Dataset Sample Images")
# plot_sample_images(test_loader, "Test Dataset Sample Images")

dataset_config = CONFIG[DATASET_NAME]
model_config = MODEL_CONFIG[MODEL_NAME]
criterion = LOSS_FUNCTIONS[LOSS_NAME]
input_shape = (1, *dataset_config['input_dim'])  # Prepends batch size of 1

model = model_config['model'](*model_config['args'](dataset_config, N)).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

MIXUP_ALPHA = 1.0  # Mixup interpolation coefficient

# Training Loop
NUMBER_OF_EPOCHS = 100
VALIDATION_ACCURACY_THRESHOLD = 95
early_stopping = False  # Flag to indicate whether early stopping was triggered

for epoch in range(NUMBER_OF_EPOCHS):
    model.train()  # Ensure model is in training mode

    # Training for current epoch
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Apply diffeomorphism transformation if the flag is set
        if USE_DIFFEOMORPHISM:
            diffeo_transform = DynamicDiffeomorphism(
                DIFFEOMORPHISM_PARAMS["temperature scale"],
                DIFFEOMORPHISM_PARAMS["c"]
            )
            images = diffeo_transform(images)

        # Apply mixup if the flag is set
        if USE_MIXUP:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA, device=device)
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
            print(f'Epoch [{epoch + 1}/{NUMBER_OF_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
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
save_path = f'./Networks/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}.pth'
print('Network saved location is', save_path)
tr.save(model.state_dict(), save_path)