import torch as tr
print(tr.cuda.is_available())
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset
from Data_sets import CustomImageDataset
from Networks import VGG, ResNet18, SimpleCNN # Assuming this import from Networks.py
from torch.nn import CrossEntropyLoss, MSELoss
from Functions import get_accuracy
import numpy as np
from DataProcessing import DynamicDiffeomorphism, typical_displacement
import matplotlib.pyplot as plt

import torch
print(torch.version.cuda)

device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
print(device)


def plot_sample_images(loader, title, n=5):
    """Plots n sample images from the given data loader"""
    dataiter = iter(loader)
    images, labels = dataiter.next()
    fig, axes = plt.subplots(1, n, figsize=(10, 3))

    for i, ax in enumerate(axes):
        if images[i].shape[0] == 1:  # if grayscale
            ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        else:  # if RGB
            ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')

    plt.suptitle(title)
    plt.show()

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
ROTATION = True  # New Flag for rotation
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

DATASET_NAME = 'CIFAR-100' # MNIST CIFAR-10, CIFAR-100 , HAM-10000

MODEL_NAME = 'VGG' #  VGG, ResNet18, SimpleCNN

LOSS_NAME = 'cross_entropy'  # label_smoothed_crossentropy or 'cross_entropy', 'mse'

N = 10

classes_to_include  = list(range(0,N)) # None

train_dataset, val_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME,classes_to_include)

# Data Loaders
BATCH_SIZE = 32
LEARNING_RATE = 0.01

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

# save initially the model
save_path = f'./Networks/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}_batch{0}.pth'
tr.save(model.state_dict(), save_path)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

MIXUP_ALPHA = 1.0  # Mixup interpolation coefficient

# Training Loop
NUMBER_OF_EPOCHS = 100
num_epochs = NUMBER_OF_EPOCHS
VALIDATION_ACCURACY_THRESHOLD = 60  # Set your desired threshold here
early_stopping = False  # Flag to indicate whether early stopping was triggered
# Set the frequency at which to save the model
BATCH_FREQUENCY = 200
# Initialize the total batch count
total_batch_count = 0

for epoch in range(num_epochs):
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

        # Increment the total batch count
        total_batch_count += 1

        # Save the network every BATCH_FREQUENCY batches
        if total_batch_count % int(len(train_loader)/ BATCH_FREQUENCY ) == 0:
            save_path = f'./Networks/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}_batch{total_batch_count}.pth'
            torch.save(model.state_dict(), save_path)
            print(total_batch_count, save_path)
        # Periodically check the validation accuracy during training
        if (i + 1) % int(len(train_loader) / 3) == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
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
tr.save(model.state_dict(), save_path)

