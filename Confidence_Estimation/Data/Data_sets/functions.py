import torch as tr
import random as rd
from torch.utils.data import Subset,  Dataset,random_split
import torchvision.transforms as transforms
import sys
import os
new_path =  sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Data.Data_sets.configurations import CONFIG

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

class TransformedDataset(Dataset):
    """ A dataset that applies transformations to inputs and outputs of another dataset. """
    def __init__(self, dataset, input_transforms, output_transforms):
        """ Initialize with the base dataset and transformation functions. """
        super(TransformedDataset, self).__init__()
        self.dataset = dataset  # Original dataset
        self.input_transforms = input_transforms  # Input transformations
        self.output_transforms = output_transforms  # Output transformations

    def __len__(self):
        """ Return the length of the base dataset """
        return len(self.dataset)

    def __getitem__(self, idx):
        """ Return transformed inputs and outputs at the specified index """
        if isinstance(idx, slice):
            # Handling slicing of the dataset
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        inputs, outputs = self.dataset[idx]
        # Apply input transformations
        for transform in self.input_transforms:
            inputs = transform(inputs)
        # Apply output transformations
        for transform in self.output_transforms:
            outputs = transform(outputs)

        return inputs, outputs

class Deterministic_Dataset:
    """ A dataset that creates a deterministic subset from another dataset. """
    def __init__(self, dataset, N):
        """ Initialize with the base dataset and the size of the subset. """
        self.N = N  # Number of samples in the subset
        self.indices = list(range(len(dataset)))[:N]  # Indices of the subset
        self.subset = [dataset[idx] for idx in self.indices]  # Subset of the dataset

    def __getitem__(self, index):
        """ Return the sample at the specified index """
        return self.subset[index]

    def __len__(self):
        """ Return the length of the subset """
        return self.N

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


def load_and_preprocess_data(dataset_name, transformations, split_sizes , classes_to_include=None):
    test_and_val_size = 0

    if dataset_name != 'HAM-10000':
        train_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=True, download=True, transform=transformations)
        test_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=False, download=True, transform=transformations)
        test_dataset_size = len(test_dataset)

    else:
        # For HAM-10000 custom dataset
        full_dataset = CONFIG[dataset_name]['loader'](CONFIG[dataset_name]['path'], CONFIG[dataset_name]['label_path'], transform=transformations)
        total_size = len(full_dataset)
        train_size = int(split_sizes["train"]*total_size)  # Remaining 70% for training
        test_dataset_size = total_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_dataset_size])

    # If there are specific classes to include, filter datasets
    if classes_to_include:
        train_dataset = filter_classes(train_dataset, classes_to_include, dataset_name)
        test_dataset = filter_classes(test_dataset, classes_to_include, dataset_name)
        test_dataset_size = len(test_dataset)

    val_size = test_dataset_size // 2
    test_size = test_dataset_size - val_size

    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset