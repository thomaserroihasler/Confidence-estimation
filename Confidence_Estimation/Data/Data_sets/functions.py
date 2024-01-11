import torch as tr
import numpy as np
import random as rd
from torch.utils.data import Subset,  Dataset
import sys
import torchvision.transforms as transforms

new_path =  sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

# from Confidence_Estimation.Data.Data_sets.configurations import CONFIG
from Confidence_Estimation.Data.Data_sets.definitions import*
from Confidence_Estimation.Other.Useful_functions.definitions import get_device
from Confidence_Estimation.Data.Data_sets.configurations import CONFIG
device = get_device()

def filter_classes(dataset, classes_to_include, dataset_name):
    if dataset_name != 'HAM-10000':
        indices = [i for i in range(len(dataset)) if dataset.targets[i] in classes_to_include]
    else:
        indices = [i for i in range(len(dataset)) if dataset.dataset.labels[i] in classes_to_include]
    return Subset(dataset, indices)

def fuse_datasets(dataset1, dataset2):
    """ Combine two datasets into a single dataset. """
    # Combine the two datasets into a single list of samples
    samples = list(dataset1) + list(dataset2)

    class FusedDataset(tr.utils.data.Dataset):
        """ A dataset class that encapsulates the fused datasets. """
        def __len__(self):
            """ Return the total number of samples"""
            return len(samples)

        def __getitem__(self, idx):
            """ Return the sample at the specified index """
            return samples[idx]

    return FusedDataset()

def Pre_training_processing(dataset, input_transforms, output_transforms):
    """ Apply transformations to both inputs and outputs of a dataset. """
    transformed_dataset = TransformedDataset(dataset, input_transforms, output_transforms)
    return transformed_dataset

def shuffle_dataset(dataset):
    """ Shuffle the order of samples in a dataset. """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    rd.shuffle(indices)  # Randomly shuffle indices
    shuffled_dataset = tr.utils.data.Subset(dataset, indices)  # Create a shuffled subset
    return shuffled_dataset

def get_dataset_splits(dataset, train_N, val_N, test_N):
    """ Split a dataset into train, validation, and test subsets. """
    # Calculate the sizes of each dataset
    total = train_N + val_N + test_N
    train_ratio = train_N / total
    val_ratio = val_N / total
    test_ratio = test_N / total

    total_len = len(dataset)
    train_len = round(total_len * train_ratio)
    val_len = round(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    # Use PyTorch's Subset function to split the dataset
    train_set = Subset(dataset, range(train_len))
    val_set = Subset(dataset, range(train_len, train_len + val_len))
    test_set = Subset(dataset, range(train_len + val_len, total_len))

    return train_set, val_set, test_set

def random_subset(d, f):
    """
    Return a random subset of a dataset.
    """
    assert 0 <= f <= 1, "f must be a value between 0 and 1."
    subset_size = int(f * len(d))  # Calculate the size of the subset
    subset_indices = tr.randperm(len(d))[:subset_size]  # Randomly select indices
    subset = Subset(d, subset_indices)  # Create a subset
    return subset

def filter_classes(dataset, classes_to_include, dataset_name):
    if dataset_name != 'HAM-10000':
        indices = [i for i in range(len(dataset)) if dataset.targets[i] in classes_to_include]
    else:
        indices = [i for i in range(len(dataset)) if dataset.dataset.labels[i] in classes_to_include]
    return Subset(dataset, indices)

from torch.utils.data import random_split

def load_and_preprocess_data(dataset_name, transformations, split_sizes = None, classes_to_include=None):
    if dataset_name != 'HAM-10000':
        train_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=True, download=True, transform=transformations)
        test_val_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=False, download=True, transform=transformations)
        # fuse the datasets no shuffling
        full_dataset =  fuse_datasets(train_dataset,test_val_dataset)

        if split_sizes:

            # For standard datasets
            total_size = len(full_dataset)
            train_size = int(split_sizes["train"] * total_size)
            val_size = int(split_sizes["validation"] * total_size)
            test_size = total_size - train_size - val_size
            train_indices = list(range(0, train_size))
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, total_size))

            # Subset the dataset
            train_dataset = tr.utils.data.Subset(full_dataset, train_indices)
            val_dataset = tr.utils.data.Subset(full_dataset, val_indices)
            test_dataset = tr.utils.data.Subset(full_dataset, test_indices)

#            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        else:
            val_dataset, test_dataset = random_split((test_val_dataset, [0.5, 0.5]))
    else:
        # For HAM-10000 custom dataset
        full_dataset = CONFIG[dataset_name]['loader'](CONFIG[dataset_name]['path'], CONFIG[dataset_name]['label_path'], transform=transformations)
        total_size = len(full_dataset)
        train_size = int(split_sizes["train"] * total_size)
        val_size = int(split_sizes["validation"] * total_size)
        test_size = total_size - train_size - val_size
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))

        # Subset the dataset
        train_dataset = tr.utils.data.Subset(full_dataset, train_indices)
        val_dataset = tr.utils.data.Subset(full_dataset, val_indices)
        test_dataset = tr.utils.data.Subset(full_dataset, test_indices)

#        train_dataset, remaining_dataset = random_split(full_dataset, [train_size, total_size - train_size])
#        val_dataset, test_dataset = random_split(remaining_dataset, [val_size, test_size])

    # If there are specific classes to include, filter datasets
    if classes_to_include:
        train_dataset = filter_classes(train_dataset, classes_to_include, dataset_name)
        val_dataset = filter_classes(val_dataset, classes_to_include, dataset_name)
        test_dataset = filter_classes(test_dataset, classes_to_include, dataset_name)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

def load_and_preprocess_data(dataset_name, transformations, split_sizes=None, classes_to_include=None):
    if dataset_name != 'HAM-10000':
        # Load standard datasets
        train_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=True, download=True, transform=transformations)
        test_val_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=False, download=True,  transform=transformations)

        # Fuse the datasets without shuffling
        full_dataset = fuse_datasets(train_dataset, test_val_dataset)
    else:
        # Load HAM-10000 custom dataset
        full_dataset = CONFIG[dataset_name]['loader'](CONFIG[dataset_name]['path'], CONFIG[dataset_name]['label_path'],   transform=transformations)

    # Calculate split sizes
    total_size = len(full_dataset)
    if split_sizes:
        train_size = int(split_sizes["train"] * total_size)
        val_size = int(split_sizes["validation"] * total_size)
    else:
        # Default split sizes if not provided
        train_size = int(0.6 * total_size)  # Example default split
        val_size = int(0.2 * total_size)  # Example default split
    test_size = total_size - train_size - val_size

    # Generate indices for each subset
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    # Subset the dataset
    train_dataset = tr.utils.data.Subset(full_dataset, train_indices)
    val_dataset = tr.utils.data.Subset(full_dataset, val_indices)
    test_dataset = tr.utils.data.Subset(full_dataset, test_indices)

    # If there are specific classes to include, filter datasets
    if classes_to_include:
        train_dataset = filter_classes(train_dataset, classes_to_include, dataset_name)
        val_dataset = filter_classes(val_dataset, classes_to_include, dataset_name)
        test_dataset = filter_classes(test_dataset, classes_to_include, dataset_name)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def Normalization(dataset_name):
    base_transforms = []
    if dataset_name == 'MNIST':
        base_transforms.extend(
            [transforms.Normalize((0.5,), (0.5,))])
    elif dataset_name in ['CIFAR-10', 'CIFAR-100']:
        base_transforms.extend(
            [transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    elif dataset_name == 'HAM-10000':
        base_transforms.extend(
            [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    return base_transforms



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, device = device):
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
