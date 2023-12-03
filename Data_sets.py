import torch as tr
from torch.utils.data import Dataset
import numpy as np
import math as mt
import os  # Used for handling file paths
from PIL import Image  # Used for image processing

class CustomImageDataset(Dataset):
    """A custom dataset for loading images with labels."""

    def __init__(self, img_dir, labels_path, transform=None):
        """
        Initializes the dataset.

        Args:
            img_dir (str): Directory containing images.
            labels_path (str): Path to the file containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir  # Directory of images
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
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetches the image and label at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the corresponding label.
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")  # Load image
        if self.transform:
            image = self.transform(image)  # Apply transformation
        label = self.labels[idx]
        return image, label


class SyntheticConfidenceDataset(Dataset):
    """A dataset for generating synthetic data based on a confidence distribution."""

    def __init__(self, N_classes, confidence_distribution, accuracy_function, noise_function, N_data):
        """
        Initializes the dataset.

        Args:
            N_classes (int): Number of classes.
            confidence_distribution (callable): Function to generate confidence values.
            accuracy_function (callable): Function to calculate accuracy.
            noise_function (callable): Function to add noise.
            N_data (int): Number of data points.
        """
        self.N_classes = N_classes
        self.confidence_distribution = confidence_distribution
        self.accuracy_function = accuracy_function
        self.N_data = N_data
        self.noise_function = noise_function

        self.data = self._generate_data()

    def __getitem__(self, index):
        """Returns the data at the specified index."""
        return self.data[index]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def _generate_data(self):
        """Generates synthetic data based on the defined functions."""
        data = []

        for _ in range(self.N_data):
            c = self.confidence_distribution()  # Sample confidence
            prob_field = self._generate_prob_field(c)  # Generate probability field
            y = self._generate_y(c, prob_field)  # Generate label

            x = tr.log(tr.tensor(prob_field, dtype=tr.float32))  # Log-transform probability field
            data.append((x, tr.tensor(y, dtype=tr.int32)))

        return data

    def _generate_prob_field(self, confidence):
        """Generates a probability field for the classes."""
        i = np.random.randint(self.N_classes)
        rest = 1 - confidence

        prob_field = np.random.rand(self.N_classes - 1)
        prob_field /= prob_field.sum()
        prob_field *= rest
        prob_field = np.insert(prob_field, i, confidence)

        return prob_field

    def _generate_y(self, confidence, prob_field):
        """Generates the label based on confidence and probability field."""
        true_prediction = np.argmax(prob_field)
        if np.random.rand() < self.accuracy_function(confidence) + self.noise_function(confidence) * (
                2 * np.random.rand() - 1):
            return true_prediction
        else:
            classes = list(range(self.N_classes))
            classes.remove(true_prediction)
            return np.random.choice(classes)


class BlackWhiteImageDataset(Dataset):
    """A dataset for generating black and white images with equal number of pixels."""

    def __init__(self, N_pixels, N_data):
        """
        Initializes the dataset.

        Args:
            N_pixels (int): Total number of pixels in an image.
            N_data (int): Number of images to generate.
        """
        self.N_pixels = N_pixels
        self.N_data = N_data
        self.image_size = int(mt.sqrt(N_pixels))  # Calculate image size

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.N_data

    def __getitem__(self, index):
        """Returns the image and label at the specified index."""
        if isinstance(index, slice):
            # Handle slicing
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        while True:
            image = np.random.rand(self.image_size, self.image_size)
            rounded_image = np.round(image).astype(int)
            black_pixels = np.count_nonzero(rounded_image == 0)
            white_pixels = np.count_nonzero(rounded_image == 1)

            if black_pixels != white_pixels:
                label = 0 if black_pixels > white_pixels else 1
                break

        image_tensor = tr.tensor(image, dtype=tr.float32)
        label_tensor = tr.tensor(label, dtype=tr.long)

        return image_tensor, label_tensor


class ProbabilisticBlackWhiteImageDataset(Dataset):
    """A dataset for generating black and white images with probabilistic labeling."""

    def __init__(self, N_pixels, N_data):
        """
        Initializes the dataset.

        Args:
            N_pixels (int): Total number of pixels in an image.
            N_data (int): Number of images to generate.
        """
        self.N_pixels = N_pixels
        self.N_data = N_data
        self.image_size = int(mt.sqrt(N_pixels))  # Calculate image size

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.N_data

    def __getitem__(self, index):
        """Returns the image and label at the specified index."""
        if isinstance(index, slice):
            # Handle slicing
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        while True:
            image = np.random.rand(self.image_size, self.image_size)
            rounded_image = np.round(image).astype(int)
            white_pixels = np.count_nonzero(rounded_image == 1)
            black_pixels = np.count_nonzero(rounded_image == 0)

            if black_pixels != white_pixels:
                break

        p_label_1 = white_pixels / self.N_pixels  # Probability of labeling as 1
        label = np.random.choice([0, 1], p=[1 - p_label_1, p_label_1])  # Sample label

        image_tensor = tr.tensor(image, dtype=tr.float32)
        label_tensor = tr.tensor(label, dtype=tr.long)

        return image_tensor, label_tensor
