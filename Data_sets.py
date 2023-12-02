import torch as tr
from torch.utils.data import Dataset
import numpy as np
import math as mt

import os  # Needed for joining paths
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_path, transform=None):
        self.img_dir = img_dir
        self.images = []
        self.labels = []
        self.transform = transform

        # Load the labels and images
        with open(labels_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_file, label = line.strip().split(' ')
                self.images.append(os.path.join(img_dir, img_file))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# takes in a distribution, number of classes, and a number of data points
class Synthetic_confidence_dataset(Dataset):
    def __init__(self, N_classes, confidence_distribution, accuracy_function, noise_function, N_data):
        self.N_classes = N_classes
        self.confidence_distribution = confidence_distribution
        self.accuracy_function = accuracy_function
        self.N_data = N_data
        self.noise_function = noise_function

        self.data = self._generate_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _generate_data(self):
        data = []

        for _ in range(self.N_data):
            # Sample a confidence value
            c = self.confidence_distribution()

            # Generate a probability field with the sampled confidence as max
            prob_field = self._generate_prob_field(c)

            # Generate y based on accuracy function and noise
            y = self._generate_y(c, prob_field)

            # Generate tensor x
            x = tr.log(tr.tensor(prob_field, dtype=tr.float32))

            data.append((x, tr.tensor(y, dtype=tr.int32)))

        return data

    def _generate_prob_field(self, confidence):
        i = np.random.randint(self.N_classes)
        rest = 1 - confidence

        prob_field = np.random.rand(self.N_classes - 1)
        prob_field /= prob_field.sum()
        prob_field = prob_field * rest

        # Insert the confidence value at the random index 'i'
        prob_field = np.insert(prob_field, i, confidence)

        # Return the probability field and the argmax of the field
        return prob_field

    def _generate_y(self, confidence, prob_field):
        true_prediction = np.argmax(prob_field)
        if np.random.rand() < self.accuracy_function(confidence) + self.noise_function(confidence)*(2*np.random.rand()-1):
            return true_prediction
        else:
            # Create a list of all classes
            classes = list(range(self.N_classes))
            # Remove the true prediction
            classes.remove(true_prediction)
            # Randomly select from the remaining classes
            return np.random.choice(classes)

class BlackWhiteImageDataset(Dataset):
    def __init__(self, N_pixels, N_data):
        self.N_pixels = N_pixels
        self.N_data = N_data
        self.image_size = int(mt.sqrt(N_pixels))

    def __len__(self):
        return self.N_data

    def __getitem__(self, index):

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        image = np.random.rand(self.image_size, self.image_size)
        rounded_image = np.round(image).astype(int)

        while True:
            black_pixels = np.count_nonzero(rounded_image == 0)
            white_pixels = np.count_nonzero(rounded_image == 1)

            if black_pixels == white_pixels:
                # Resample the image
                image = np.random.rand(self.image_size, self.image_size)
                rounded_image = np.round(image).astype(int)
            else:
                label = 0 if black_pixels > white_pixels else 1
                break

        # Convert image and label to tensors
        image_tensor = tr.tensor(image, dtype=tr.float32)
        label_tensor = tr.tensor(label, dtype=tr.long)

        return image_tensor, label_tensor

class ProbabilisticBlackWhiteImageDataset(Dataset):
    def __init__(self, N_pixels, N_data):
        self.N_pixels = N_pixels
        self.N_data = N_data
        self.image_size = int(mt.sqrt(N_pixels))

    def __len__(self):
        return self.N_data

    def __getitem__(self, index):

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        while True:
            image = np.random.rand(self.image_size, self.image_size)
            rounded_image = np.round(image).astype(int)

            white_pixels = np.count_nonzero(rounded_image == 1)
            black_pixels = np.count_nonzero(rounded_image == 0)

            if black_pixels != white_pixels:
                break

        p_label_1 = white_pixels / self.N_pixels

        # Sample the label from the conditional distribution
        label = np.random.choice([0, 1], p=[1 - p_label_1, p_label_1])

        # Convert image and label to tensors
        image_tensor = tr.tensor(image, dtype=tr.float32)
        label_tensor = tr.tensor(label, dtype=tr.long)

        return image_tensor, label_tensor

