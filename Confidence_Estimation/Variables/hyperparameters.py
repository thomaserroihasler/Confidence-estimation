import torch as tr
import os
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATASET_NAME = 'MNIST'
NETWORK_NAME = 'VGG'

INPUT_LOCATION = '../../../Data/Inputs/' + DATASET_NAME
NUMBER_OF_CLASSES = None

# Split sizes for non-standard datasets
SPLIT_SIZES = {
    "train": 0.6,  # 60% data for training
    "validation": 0.2,  # 20% data for validation
    "test": 0.2  # 20% data for testing
}

# ACTIVATING DIFFERENT TRANSFORMATIONS

# Transformations that are always used on the data
BASIC_TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor()
])

# Define transformations
TRAINING_TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

LEARNING_RATE = 0.01
LOSS_FUNCTION = 'Cross-entropy'

VALIDATION_TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

OUTPUT_LOCATION = '../../../Data/Outputs/' + DATASET_NAME
