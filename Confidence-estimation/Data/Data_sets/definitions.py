import torch as tr
from torch.utils.data import Dataset
import numpy as np
import math as mt

class TensorDataset(Dataset):
    """ Custom dataset for pairing inputs and outputs. """
    def __init__(self, inputs, outputs):
        # Initialize the dataset with input and output data.
        super(TensorDataset, self).__init__()
        assert len(inputs) == len(outputs), "Inputs and outputs must have the same length"
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        # Return the total number of samples.
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return the input-output pair at the specified index
        return self.inputs[idx], self.outputs[idx]

class CustomTransformDataset(Dataset):
    """Dataset that applies transformations N times to each item."""

    def __init__(self, dataset, transform=None, N=10):
        self.dataset = dataset
        self.transform = transform
        self.N = N

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Check if the image is a PIL Image or ndarray before applying the transformation
        if self.transform:
            if not isinstance(img, tr.Tensor):
                transformed_imgs = [self.transform(img) for _ in range(self.N)]
            else:
                # The image is already a tensor, so we skip the ToTensor transformation
                transformed_imgs = [img for _ in range(self.N)]
        else:
            transformed_imgs = [img for _ in range(self.N)]

        return transformed_imgs, label
