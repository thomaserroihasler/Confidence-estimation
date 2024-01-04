import torch as tr
from torch.utils.data import Dataset

class ConvertLabelsToInt:
    """Converts labels to integer tensors."""

    def __call__(self, label):
        """Converts a label to a PyTorch tensor of type int."""
        return tr.tensor(label, dtype=tr.int)

class Deterministic_Dataset:
    """Creates a deterministic subset from another dataset."""

    def __init__(self, dataset, N):
        """Initializes with the base dataset and the size of the subset."""
        self.N = N  # Number of samples in the subset
        self.indices = list(range(len(dataset)))[:N]  # Indices of the subset
        self.subset = [dataset[idx] for idx in self.indices]  # Subset of the dataset

    def __getitem__(self, index):
        """Returns the sample at the specified index."""
        return self.subset[index]

    def __len__(self):
        """Returns the length of the subset."""
        return self.N

class TransformedDataset(Dataset):
    """Applies transformations to inputs and outputs of another dataset."""

    def __init__(self, dataset, input_transforms, output_transforms):
        """Initializes with the base dataset and transformation functions."""
        super(TransformedDataset, self).__init__()
        self.dataset = dataset  # Original dataset
        self.input_transforms = input_transforms  # Input transformations
        self.output_transforms = output_transforms  # Output transformations

    def __len__(self):
        """Returns the length of the base dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns transformed inputs and outputs at the specified index."""
        if isinstance(idx, slice):
            # Handles slicing of the dataset
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

class TensorDataset(Dataset):
    """Pairs inputs and outputs into a dataset."""

    def __init__(self, inputs, outputs):
        """Initializes the dataset with input and output data."""
        super(TensorDataset, self).__init__()
        assert len(inputs) == len(outputs), "Inputs and outputs must have the same length"
        self.inputs = inputs  # Input data
        self.outputs = outputs  # Output data

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """Returns the input-output pair at the specified index."""
        return self.inputs[idx], self.outputs[idx]

class CustomTransformDataset(Dataset):
    """Applies a transformation N times to each item in a dataset."""

    def __init__(self, dataset, transform=None, N=10):
        """Initializes with the base dataset, transformation function, and N repetitions."""
        self.dataset = dataset
        self.transform = transform
        self.N = N

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns transformed images and label at the specified index."""
        img, label = self.dataset[idx]
        if self.transform:
            # Applies the transformation N times and creates a tensor
            transformed_imgs = tr.stack([self.transform(img) for _ in range(self.N)])
        else:
            # Repeats the image tensor N times if no transformation
            transformed_imgs = img.unsqueeze(0).repeat(self.N, 1, 1, 1)

        return transformed_imgs, label
