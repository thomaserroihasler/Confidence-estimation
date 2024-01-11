from torch.nn import CrossEntropyLoss, MSELoss
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import sys
# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Other.Measures.definitions import LabelSmoothedCrossEntropyLoss
from Confidence_Estimation.Networks_and_predictors.Networks.definitions import VGG, ResNet18, SimpleCNN # Assuming this import from Networks.py
from Confidence_Estimation.Data.Data_sets.definitions import CustomTransformDataset
from Confidence_Estimation.Data.Data_sets.functions import Normalization
from Confidence_Estimation.Data.Data_transformations.definitions import DynamicDiffeomorphism
### META VARIABLES ###

LOSS_FUNCTIONS = {
    'Label_smoothed_cross-entropy': LabelSmoothedCrossEntropyLoss(epsilon=0.1),
    'Cross-entropy': CrossEntropyLoss(),
    'MSE': MSELoss()  # Not typically used for classification
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

### DATA SET ###

DATASET_NAME = 'MNIST' # Name of the dataset
NUMBER_OF_CLASSES = None #Number of classes to consider
SPLIT_SIZES = { "train": 5/6, "validation": 1/12, "test": 1/12} # Split sizes (usually for non-standard datasets)
SPLIT_SIZES = { "train": 6/7, "validation": 1/14, "test": 1/14} # Split sizes (usually for non-standard datasets)

#SPLIT_SIZES = None
#SPLIT_SIZES = { "train": 0.6, "validation": 0.2, "test": 0.2} # Split sizes (usually for non-standard datasets)

BASIC_TRANSFORMATIONS = transforms.Compose([transforms.ToTensor()]+ Normalization(DATASET_NAME))

CONFIG = { # Configurations for each dataset
    'MNIST': {
        'loader': datasets.MNIST,
        'normalization': BASIC_TRANSFORMATIONS,
        'input_dim': (1, 28, 28),
        'path': './data/MNIST',
        'classes': list(range(10))
    },
    'HAM-10000': {
        'loader': CustomTransformDataset,
        'transforms': BASIC_TRANSFORMATIONS,
        'input_dim': (3, 128, 128),
        'path': './Dataset/HAM10000_combined/images',
        'label_path': './Dataset/HAM10000_combined/Labels.txt',
        'classes': list(range(7))
    },
    'CIFAR-10': {
        'loader': datasets.CIFAR10,
        'transforms': BASIC_TRANSFORMATIONS,
        'input_dim': (3, 32, 32),
        'path': './data/CIFAR10',
        'classes': list(range(10))
    },
    'CIFAR-100': {
        'loader': datasets.CIFAR100,
        'transforms': BASIC_TRANSFORMATIONS,
        'input_dim': (3, 32, 32),
        'path': './data/CIFAR100',
        'classes': list(range(100))
    },
}

### NETWORK ###

NETWORK_NAME = 'SimpleCNN'
NETWORK_SAVE = 'save.pth'
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
TRAINING_NUMBER_OF_EPOCHS = 0

TRAINING_LOSS_FUNCTION = 'Cross-entropy'
TRAINING_BATCH_SIZE = 32

TRAINING_OPTIMIZER = 'SGD'
TRAINING_LEARNING_RATE = 0.01
TRAINING_MOMENTUM = 0.9
TRAINING_WEIGHT_DECAY = 0.0005

VALIDATION_ACCURACY_THRESHOLD = 95
EARLY_STOPPING = False  # Flag to indicate whether early stopping was triggered

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
TRAINING_DIFFEOMORPHISM_PARAMS = {"Temperature_scale": TRAINING_DIFFEOMORPHISM_TEMPERATURE_SCALE,  "c": TRAINING_DIFFEOMORPHISM_DEGREES_OF_FREEDOM}

TRAINING_ADDITIONAL_TRANSFORMATIONS = transforms.Compose([transforms.ToTensor()])
diffeo = DynamicDiffeomorphism(**TRAINING_DIFFEOMORPHISM_PARAMS)
#TRAINING_PARALLEL_TRANSFORMATIONS = [diffeo]
TRAINING_PARALLEL_TRANSFORMATIONS = None

### OUTPUT GENERATION ###

OUTPUT_GENERATION_H_FLIP = False
OUTPUT_GENERATION_V_FLIP = False
GENERATION_OUTPUT_RANDOM_CROP = False
GENERATION_OUTPUT_COLOR_JITTER = False
GENERATION_OUTPUT_ROTATION = False  # New Flag for rotation
GENERATION_OUTPUT_ROTATION_DEGREES = (-180, 180)  # Define the range for rotation
GENERATION_OUTPUT_USE_MIXUP = False  # Add a flag to turn on or off mixup
GENERATION_OUTPUT_MIXUP_ALPHA = 1.0  # Mixup interpolation coefficient
GENERATION_OUTPUT_USE_DIFFEOMORPHISM = False  # Flag to turn on or off the diffeomorphism transformations
GENERATION_OUTPUT_DIFFEOMORPHISM_DEGREES_OF_FREEDOM = 5
GENERATION_OUTPUT_DIFFEOMORPHISM_TEMPERATURE_SCALE = 1
GENERATION_OUTPUT_DIFFEOMORPHISM_PARAMS = {"Temperature_scale": GENERATION_OUTPUT_DIFFEOMORPHISM_TEMPERATURE_SCALE,  "c": GENERATION_OUTPUT_DIFFEOMORPHISM_DEGREES_OF_FREEDOM}

rotation_degrees = (-10, 10)
GENERATION_OUTPUT_ADDITIONAL_TRANSFORMATIONS = None
# transforms.Compose([
#     transforms.RandomRotation(rotation_degrees)
# ])
diffeo = DynamicDiffeomorphism(**GENERATION_OUTPUT_DIFFEOMORPHISM_PARAMS)
GENERATION_OUTPUT_PARALLEL_TRANSFORMATIONS = [diffeo]
GENERATION_OUTPUT_NUMBER_OF_TRANSFORMATIONS =  1
GENERATION_OUTPUT_BATCH_SIZE = 32
OUTPUT_SAVE = 'all_model_data.pth'

### CONFIDENCE ESTIMATOR VALIDATION

VALIDATION_NUMBER_OF_TRANSFORMATIONS = 10
VALIDATION_NUMBER_OF_EPOCHS = 25

VALIDATION_LOSS_FUNCTION = 'Cross-entropy'
VALIDATION_NUMBER_OF_BINS = None
VALIDATION_NUMBER_OF_POINTS_PER_BIN = 50
VALIDATION_BINNING = { 'num_bins': VALIDATION_NUMBER_OF_BINS, 'num_per_bin' : VALIDATION_NUMBER_OF_POINTS_PER_BIN}
VALIDATION_BATCH_SIZE = 32

VALIDATION_OPTIMIZER = 'SGD'
VALIDATION_LEARNING_RATE = 0.01
VALIDATION_MOMENTUM = 0.9
VALIDATION_WEIGHT_DECAY = 0.0005

TEST_ACCURACY_THRESHOLD = 95
EARLY_STOPPING = False  # Flag to indicate whether early stopping was triggered

VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL = 50
VALIDATION_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED = 50
VALIDATION_TEMPERATURE_SCALE = 1.0

### CONFIDENCE ESTIMATORS TEST ###

TEST_NUMBER_OF_TRANSFORMATIONS = 100
TEST_BATCH_SIZE = 32

TEST_LEARNING_RATE = 0.01
TEST_LOSS_FUNCTION = 'Cross-entropy'
TEST_NUMBER_OF_BINS = 30
TEST_NUMBER_OF_POINTS_PER_BIN = None
TEST_BINNING = { 'num_bins': TEST_NUMBER_OF_BINS, 'num_per_bin' : TEST_NUMBER_OF_POINTS_PER_BIN}
TEST_NUMBER_OF_EPOCHS = 100

TEST_NUMBER_OF_NEAREST_NEIGHBORS_NORMAL = 50
TEST_NUMBER_OF_NEAREST_NEIGHBORS_TRANSFORMED = 50

### FOLDER LOCATIONS ###

INPUT_LOCATION = '../../../Data/Inputs/'+DATASET_NAME+'/'+NETWORK_NAME
OUTPUT_LOCATION = '../../../Data/Outputs/'+DATASET_NAME+'/'+NETWORK_NAME
NETWORK_LOCATION =  '../../../Networks/'+DATASET_NAME+'/'+NETWORK_NAME
CALIBRATION_LOCATION = '../../../CalibrationMethods/'+DATASET_NAME+'/'+NETWORK_NAME
FINAL_DATA_LOCATION  = '../../../Data/Final_Data/'+DATASET_NAME+'/'+NETWORK_NAME

### FILE PATHS ###

INPUT_FILE_PATH  = INPUT_LOCATION
OUTPUT_FILE_PATH  = OUTPUT_LOCATION +'/'+ OUTPUT_SAVE
NETWORK_FILE_PATH = NETWORK_LOCATION+'/' + NETWORK_SAVE
