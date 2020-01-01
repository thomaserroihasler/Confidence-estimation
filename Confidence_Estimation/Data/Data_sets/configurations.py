from torchvision import datasets
import sys

new_path =  sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Data.Data_sets.definitions import CustomTransformDataset

CONFIG = {
    'MNIST': {
        'loader': datasets.MNIST,
        'input_dim': (1, 28, 28),
        'path': '../../../Data/Inputs/MNIST',
        'classes': list(range(10))
    },
    'HAM-10000': {
        'loader': CustomTransformDataset,
        'input_dim': (3, 128, 128),
        'path': '../../../Data/Inputs/HAM-10000/images',
        'label_path': '../../../Data/Inputs/HAM-10000/Labels.txt',
        'classes': list(range(7))
    },
    'CIFAR-10': {
        'loader': datasets.CIFAR10,
        'input_dim': (3, 32, 32),
        'path': '../../../Data/Inputs/CIFAR-10',
        'classes': list(range(10))
    },
    'CIFAR-100': {
        'loader': datasets.CIFAR100,
        'input_dim': (3, 32, 32),
        'path': '../../../Data/Inputs/CIFAR-100',
        'classes': list(range(100))
    },
}