from Confidence_Estimation.Other.Measures.definitions import LabelSmoothedCrossEntropyLoss
from Confidence_Estimation.Networks_and_predictors.Networks.definitions import VGG, ResNet18, SimpleCNN # Assuming this import from Networks.py
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss, MSELoss
from torchvision import datasets
import torch.optim as optim

### META VARIABLES ###

LOSS_FUNCTIONS = {
    'label_smoothed_crossentropy': LabelSmoothedCrossEntropyLoss(epsilon=0.1),
    'cross_entropy': CrossEntropyLoss(),
    'mse': MSELoss()  # Not typically used for classification
}

OPTIMIZERS = {
    'SGD': {
        'optimizer': optim.SGD,
        'args': lambda params, lr, momentum, weight_decay: {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    },
    'Adam': {
        'optimizer': optim.Adam,
        'args': lambda params, lr, weight_decay: {'lr': lr, 'weight_decay': weight_decay}
    },
    'RMSprop': {
        'optimizer': optim.RMSprop,
        'args': lambda params, lr, momentum: {'lr': lr, 'momentum': momentum}
    }
}

BASIC_TRANSFORMATIONS = transforms.Compose([ transforms.ToTensor()])

### DATA SET ###

DATASET_NAME = 'MNIST' # Name of the dataset
NUMBER_OF_CLASSES = None #Number of classes to consider
SPLIT_SIZES = { "train": 0.6, "validation": 0.2, "test": 0.2} # Split sizes (usually for non-standard datasets)

CONFIG = { # Configurations for each dataset
    'MNIST': {
        'loader': datasets.MNIST,
        'transforms': get_transforms('MNIST'),
        'input_dim': (1, 28, 28),
        'path': './data/MNIST',
        'classes': list(range(10))
    },
    'HAM-10000': {
        'loader': CustomImageDataset,
        'transforms': get_transforms('HAM-10000'),
        'input_dim': (3, 128, 128),
        'path': './Dataset/HAM10000_combined/images',
        'label_path': './Dataset/HAM10000_combined/Labels.txt',
        'classes': list(range(7))
    },
    'CIFAR-10': {
        'loader': datasets.CIFAR10,
        'transforms': get_transforms('CIFAR-10'),
        'input_dim': (3, 32, 32),
        'path': './data/CIFAR10',
        'classes': list(range(10))
    },
    'CIFAR-100': {
        'loader': datasets.CIFAR100,
        'transforms': get_transforms('CIFAR-100'),
        'input_dim': (3, 32, 32),
        'path': './data/CIFAR100',
        'classes': list(range(100))
    },
}
### NETWORK ###

NETWORK_NAME = 'VGG'
MODELS = { # Model Configurations
    'VGG': {
        'model': VGG,
        'args': lambda config, n: ('VGG11', config['input_dim'][0], n, config['input_dim'])
    },
    'ResNet18': {
        'model': ResNet18,
        'args': lambda config, n: (config['input_dim'][0], n)
    },
    'SimpleCNN': {
        'model': SimpleCNN,
        'args': lambda config, n: (config['input_dim'], n)
    }
}

### NETWORK TRAINING ###

TRAINING_LOSS_FUNCTION = 'Cross-entropy'
TRAINING_BATCH_SIZE = 32

TRAINING_OPTIMIZER = 'SGD'
TRAINING_LEARNING_RATE = 0.01
TRAINING_MOMENTUM = 0.9
TRAINING_WEIGHT_DECAY = 0.0005

TRAINING_H_FLIP = False
TRAINING_V_FLIP = False
TRAINING_RANDOM_CROP = False
TRAINING_COLOR_JITTER = False
TRAINING_ROTATION = False  # New Flag for rotation
TRAINING_ROTATION_DEGREES = (-180, 180)  # Define the range for rotation
TRAINING_USE_MIXUP = False  # Add a flag to turn on or off mixup
TRAINING_MIXUP_ALPHA = 1.0  # Mixup interpolation coefficient
TRAINING_USE_DIFFEOMORPHISM = False  # Flag to turn on or off the diffeomorphism transformations
TRAINING_DIFFEOMORPHISM_DEGREES_OF_FREEDOM = 5
TRAINING_DIFFEOMORPHISM_TEMPERATURE_SCALE = 1
TRAINING_DIFFEOMORPHISM_PARAMS = {"temperature scale": TRAINING_DIFFEOMORPHISM_TEMPERATURE_SCALE,  "c": TRAINING_DIFFEOMORPHISM_DEGREES_OF_FREEDOM}

TRAINING_ADDITIONAL_TRANSFORMATIONS = transforms.Compose([transforms.ToTensor()])
TRAINING_PARALLEL_TRANSFORMATIONS = None

TRAINING_NUMBER_OF_EPOCHS = 1
VALIDATION_ACCURACY_THRESHOLD = 95
EARLY_STOPPING = False  # Flag to indicate whether early stopping was triggered

### CONFIDENCE ESTIMATOR VALIDATION ###

VALIDATION_USE_DIFFEOMORPHISM = False  # Flag to turn on or off the diffeomorphism transformations
VALIDATION_TEMPERATURE_SCALE = 1.0
VALIDATION_DIFFEOMORPHISM_DEGREES_OF_FREEDOM = 5
VALIDATION_DIFFEOMORPHISM_TEMPERATURE_SCALE = 1
VALIDATION_DIFFEOMORPHISM_PARAMS = {"temperature scale": VALIDATION_DIFFEOMORPHISM_TEMPERATURE_SCALE,  "c": VALIDATION_DIFFEOMORPHISM_DEGREES_OF_FREEDOM}

VALIDATION_ADDITIONAL_TRANSFORMATIONS = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip()     # Randomly flip the image vertically
])

VALIDATION_NUMBER_OF_TRANSFORMATIONS = 2
VALIDATION_BATCH_SIZE = 32

VALIDATION_LEARNING_RATE = 0.01
VALIDATION_LOSS_FUNCTION = 'Cross-entropy'
VALIDATION_NUMBER_OF_EPOCHS = 100

VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL = 20
VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED = 20

VALIDATION_DIFFEOMORPHISM_PARAMS = {"temperature scale": 1,"c": 5}

### CONFIDENCE ESTIMATORS TEST ###

TEST_NUMBER_OF_TRANSFORMATIONS = 2
TEST_BATCH_SIZE = 32

TEST_LEARNING_RATE = 0.01
TEST_LOSS_FUNCTION = 'Cross-entropy'
TEST_NUMBER_OF_EPOCHS = 100

TEST_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL = 20
TEST_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED = 20

### FILE LOCATIONS ###

INPUT_LOCATION = '../../../Data/Inputs/' + DATASET_NAME
OUTPUT_LOCATION = '../../../Data/Outputs/' + DATASET_NAME
CALIBRATION_LOCATION = '../../../CalibrationMethods/' + DATASET_NAME
