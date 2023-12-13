import torch as tr
from torch.utils.data import Dataset


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
        print(type(self.inputs[idx]))
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
        if self.transform:
            # Apply the transformation N times and create a tensor
            transformed_imgs = tr.stack([self.transform(img) for _ in range(self.N)])
        else:
            # If no transformation, repeat the image tensor N times
            transformed_imgs = img.unsqueeze(0).repeat(self.N, 1, 1, 1)

        return transformed_imgs, label

