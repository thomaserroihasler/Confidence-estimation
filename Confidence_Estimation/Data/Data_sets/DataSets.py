import torch as tr
from torch.utils.data import Dataset
import numpy as np
import math as mt
import os  # Used for handling file paths
from PIL import Image  # Used for image processing

class TensorDataset(Dataset):
    """ Custom dataset for pairing inputs and outputs. """
    def __init__(self, inputs, outputs):
        """ Initialize the dataset with input and output data. """
        super(TensorDataset, self).__init__()
        assert len(inputs) == len(outputs), "Inputs and outputs must have the same length"
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        """ Return the total number of samples """
        return len(self.inputs)

    def __getitem__(self, idx):
        """ Return the input-output pair at the specified index """
        return self.inputs[idx], self.outputs[idx]

def create_dataset(inputs, outputs):
    """ Creates a TensorDataset with the given inputs and outputs. """
    dataset = TensorDataset(inputs, outputs)
    return dataset

class CustomImageDataset(Dataset):
    """ A custom dataset for loading images with labels. """
    def __init__(self, img_dir, labels_path, transform=None):
        """ Initialize the dataset with image directory, label path, and optional transform. """
        self.img_dir = img_dir  # Directory containing images
        self.images = []  # List to store image file paths
        self.labels = []  # List to store labels
        self.transform = transform  # Transformations to be applied to images

        # Load labels and image paths
        with open(labels_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_file, label = line.strip().split(' ')
                self.images.append(os.path.join(img_dir, img_file))
                self.labels.append(int(label))

    def __len__(self):
        """ Return the total number of samples in the dataset """
        return len(self.labels)

    def __getitem__(self, idx):
        """ Fetch the image and label at the specified index. """
        img_path = self.images[idx]  # Path of the image
        image = Image.open(img_path).convert("RGB")  # Load image and convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply transformation
        label = self.labels[idx]  # Corresponding label
        return image, label

class SyntheticConfidenceDataset(Dataset):
    """ A dataset for generating synthetic data based on a confidence distribution. """
    def __init__(self, N_classes, confidence_distribution, accuracy_function, noise_function, N_data):
        """ Initialize the dataset with class count, confidence distribution, accuracy function, noise function, and data count. """
        self.N_classes = N_classes  # Number of classes
        self.confidence_distribution = confidence_distribution  # Function for confidence values
        self.accuracy_function = accuracy_function  # Function to calculate accuracy
        self.noise_function = noise_function  # Function to add noise
        self.N_data = N_data  # Number of data points

        self.data = self._generate_data()  # Generate the synthetic data

    def __getitem__(self, index):
        """ Return the data at the specified index. """
        return self.data[index]

    def __len__(self):
        """ Return the total number of samples in the dataset. """
        return len(self.data)

    def _generate_data(self):
        """ Generate synthetic data based on the defined functions. """
        data = []
        for _ in range(self.N_data):
            c = self.confidence_distribution()  # Sample a confidence value
            prob_field = self._generate_prob_field(c)  # Generate a probability field
            y = self._generate_y(c, prob_field)  # Generate a label

            x = tr.log(tr.tensor(prob_field, dtype=tr.float32))  # Log-transform the probability field
            data.append((x, tr.tensor(y, dtype=tr.int32)))  # Append data tuple

        return data

    def _generate_prob_field(self, confidence):
        """ Generate a probability field for the classes. """
        i = np.random.randint(self.N_classes)  # Random class index
        rest = 1 - confidence  # Remaining confidence

        prob_field = np.random.rand(self.N_classes - 1)  # Probabilities for other classes
        prob_field /= prob_field.sum()  # Normalize
        prob_field *= rest  # Adjust based on remaining confidence
        prob_field = np.insert(prob_field, i, confidence)  # Insert high confidence at class index

        return prob_field

    def _generate_y(self, confidence, prob_field):
        """ Generate the label based on confidence and probability field. """
        true_prediction = np.argmax(prob_field)  # Determine true prediction
        # Calculate label based on accuracy function and noise
        if np.random.rand() < self.accuracy_function(confidence) + self.noise_function(confidence) * (2 * np.random.rand() - 1):
            return true_prediction
        else:
            classes = list(range(self.N_classes))
            classes.remove(true_prediction)  # Remove true prediction from choices
            return np.random.choice(classes)  # Randomly choose other class

class BlackWhiteImageDataset(Dataset):
    """ A dataset for generating black and white images with equal number of pixels. """
    def __init__(self, N_pixels, N_data):
        """ Initialize the dataset with the number of pixels and the number of data points. """
        self.N_pixels = N_pixels  # Total number of pixels in an image
        self.N_data = N_data  # Number of images to generate
        self.image_size = int(mt.sqrt(N_pixels))  # Calculate image size

    def __len__(self):
        """ Return the total number of samples in the dataset. """
        return self.N_data

    def __getitem__(self, index):
        """ Return the image and label at the specified index. """
        if isinstance(index, slice):
            # Handling slicing of the dataset
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        while True:
            image = np.random.rand(self.image_size, self.image_size)  # Generate random image
            rounded_image = np.round(image).astype(int)  # Round to create black and white image
            black_pixels = np.count_nonzero(rounded_image == 0)
            white_pixels = np.count_nonzero(rounded_image == 1)

            if black_pixels != white_pixels:
                label = 0 if black_pixels > white_pixels else 1  # Determine label based on majority color
                break

        image_tensor = tr.tensor(image, dtype=tr.float32)  # Convert to tensor
        label_tensor = tr.tensor(label, dtype=tr.long)  # Label tensor

        return image_tensor, label_tensor

class ProbabilisticBlackWhiteImageDataset(Dataset):
    """ A dataset for generating black and white images with probabilistic labeling. """
    def __init__(self, N_pixels, N_data):
        """ Initialize the dataset with the number of pixels and the number of data points. """
        self.N_pixels = N_pixels  # Total number of pixels
        self.N_data = N_data  # Number of images
        self.image_size = int(mt.sqrt(N_pixels))  # Calculate image size

    def __len__(self):
        """ Return the total number of samples in the dataset. """
        return self.N_data

    def __getitem__(self, index):
        """ Return the image and label at the specified index. """
        if isinstance(index, slice):
            # Handling slicing of the dataset
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        while True:
            image = np.random.rand(self.image_size, self.image_size)  # Random image generation
            rounded_image = np.round(image).astype(int)  # Round to create black and white image
            white_pixels = np.count_nonzero(rounded_image == 1)
            black_pixels = np.count_nonzero(rounded_image == 0)

            if black_pixels != white_pixels:
                break

        p_label_1 = white_pixels / self.N_pixels  # Probability of labeling as 1
        label = np.random.choice([0, 1], p=[1 - p_label_1, p_label_1])  # Probabilistic label assignment

        image_tensor = tr.tensor(image, dtype=tr.float32)  # Convert image to tensor
        label_tensor = tr.tensor(label, dtype=tr.long)  # Convert label to tensor

        return image_tensor, label_tensor
