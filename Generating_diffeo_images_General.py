import torch as tr
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset
from Data_sets import CustomImageDataset
from Networks import VGG, ResNet18, SimpleCNN, load_networks # Assuming this import from Networks.py
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from DataProcessing import DynamicDiffeomorphism
import matplotlib.pyplot as plt
import sys

# Check if sufficient arguments are provided
if len(sys.argv) < 3:
    raise ValueError("MODEL_NAME and DATASET_NAME must be provided as arguments")

MODEL_NAME = 'SimpleCNN'
DATASET_NAME = 'HAM-10000'

device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
print(device)

# Loss and Optimizer
class LabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        n_classes = outputs.size(-1)
        log_preds = tr.nn.functional.log_softmax(outputs, dim=-1)
        loss = -log_preds.gather(dim=-1, index=targets.unsqueeze(-1))
        loss = loss.squeeze(-1) * (1-self.epsilon) - (self.epsilon / n_classes) * log_preds.sum(dim=-1)
        return loss.mean()


def mixup_data(x, y, alpha=1.0, device=device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = tr.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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

NOISE = False
DIFFEOMORPHISM_PARAMS = {
    "temperature scale": 1.0,  # Initialize with desired value for temperature
    "c": 5
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
        'path': './data',
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
        train_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=True, download=True, transform=transform)
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

DATASET_NAME = 'HAM-10000'

MODEL_NAME = 'SimpleCNN'

LOSS_NAME = 'cross_entropy'  # label_smoothed_crossentropy or 'cross_entropy', 'mse'

NUMBER_OF_CLASSES = 10

classes_to_include  = list(range(0,NUMBER_OF_CLASSES)) # None

train_dataset, val_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME,classes_to_include)

# Data Loaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

dataset_config = CONFIG[DATASET_NAME]
model_config = MODEL_CONFIG[MODEL_NAME]
input_shape = (1, *dataset_config['input_dim'])  # Prepends batch size of 1

model = model_config['model'](*model_config['args'](dataset_config, NUMBER_OF_CLASSES)).to(device)

# Define the path to the saved model
file_location = f'./Networks/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}.pth'

# Use the provided function to load the model(s)
loaded_networks = load_networks(model, file_location)

# Assuming you only saved one model in the file and want to use the first loaded network
model = loaded_networks[0]

# Training Loop

TOTAL_DIFFEO_NUMBER = 75

def plot_sample_transformed_images(inputs, title, n=5):
    """Plots n sample transformed images"""
    fig, axes = plt.subplots(1, n, figsize=(10, 3))

    for i, ax in enumerate(axes):
        if inputs[i].shape[0] == 1:  # if grayscale
            ax.imshow(inputs[i].squeeze().cpu().numpy(), cmap='gray')
        else:  # if RGB
            ax.imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def apply_diffeomorphism(inputs, TOTAL_DIFFEO_NUMBER):
    """Apply diffeomorphism transformations on inputs."""
    #inputs is of dimension 4 can you make expanded_inputs
    #print(inputs.shape)
    #print(TOTAL_DIFFEO_NUMBER.type)
    expanded_inputs = inputs.repeat(TOTAL_DIFFEO_NUMBER, 1, 1, 1)
    #plot_sample_transformed_images(expanded_inputs, "expanded Images")
    #print(expanded_inputs.shape,inputs.shape)
    # Apply the diffeomorphism (assuming it can handle batch processing)
    diffeo_transform = DynamicDiffeomorphism(
        DIFFEOMORPHISM_PARAMS["temperature scale"],
        DIFFEOMORPHISM_PARAMS["c"], NOISE
    )
    transformed_inputs = diffeo_transform(expanded_inputs)
    #print(transformed_inputs.shape)
    #plot_sample_transformed_images(transformed_inputs, "expanded transformed Images")
    return transformed_inputs

def evaluate_with_transformations(loader, model, TOTAL_DIFFEO_NUMBER):
    original_outputs = []
    transformed_outputs = []
    labels = []
    accuracies = []

    total_batches = len(loader)
    batches_10_percent = total_batches // 10  # Calculate 10% of total batches

    with tr.no_grad():
        # Add a counter for the number of processed batches
        batch_counter = 0
        for batch_inputs, batch_labels in loader:
            batch_size = batch_inputs.size(0)
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            # Original outputs
            outputs = model(batch_inputs)
            original_outputs.append(outputs.cpu())

            # Apply transformations
            #print(batch_inputs.shape,batch_size)
            transformed_inputs = apply_diffeomorphism(batch_inputs, TOTAL_DIFFEO_NUMBER)
            #print(transformed_inputs.shape)
            temp_transformed_inputs = transformed_inputs.view(TOTAL_DIFFEO_NUMBER,batch_size, *batch_inputs.shape[1:])

            # print('first image shape',first_image_transformations.shape)
            # plot_sample_transformed_images(first_image_transformations, "Transformed Images",2)
            outputs_transformed = model(transformed_inputs)

            # Reshape outputs_transformed here
            outputs_transformed = outputs_transformed.reshape(TOTAL_DIFFEO_NUMBER,batch_size, -1).permute(1, 0, 2)
            transformed_outputs.append(outputs_transformed.cpu())

            # Labels and accuracies
            labels.append(batch_labels.cpu())
            _, predicted = outputs.max(1)
            batch_accuracy = (predicted == batch_labels).int().cpu()  # Convert bools to ints here
            accuracies.append(batch_accuracy)

            # Increment the batch counter and print every 10% batches
            batch_counter += 1
            if batch_counter % batches_10_percent == 0:
                print(f"Processed {batch_counter * 100 // total_batches}% of batches.")

    # Convert lists to tensors
    original_outputs = tr.cat(original_outputs, 0)
    transformed_outputs = tr.cat(transformed_outputs, 0)
    labels = tr.cat(labels, 0)
    accuracies = tr.cat(accuracies, 0)  # Use tr.stack instead of tr.tensor
    #print(transformed_outputs.shape)
    return original_outputs, transformed_outputs, labels, accuracies

# Evaluation on validation and test

# fuse and then resplit

val_original_outputs, val_transformed_outputs, val_labels, val_accuracies = evaluate_with_transformations(val_loader, model, TOTAL_DIFFEO_NUMBER)
test_original_outputs, test_transformed_outputs, test_labels, test_accuracies = evaluate_with_transformations(test_loader, model, TOTAL_DIFFEO_NUMBER)

tr.save({
    'val_original_outputs': val_original_outputs,
    'val_transformed_outputs': val_transformed_outputs,
    'val_labels': val_labels,
    'val_accuracies': val_accuracies,
    'test_original_outputs': test_original_outputs,
    'test_transformed_outputs': test_transformed_outputs,
    'test_labels': test_labels,
    'test_accuracies': test_accuracies
}, f'./Data/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}.pth')

print('Diffeo data saved location is', f'./Data/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}.pth')