import sys
# Update the system path to include the directory for Confidence Estimation
new_path = sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Configurations.definitions import *

def create_optimizer(optimizer_name, model_params, lr, momentum=0, weight_decay=0):
    optimizer_class = OPTIMIZERS[optimizer_name]['optimizer']
    optimizer_args = OPTIMIZERS[optimizer_name]['args'](model_params, lr, momentum, weight_decay)
    return optimizer_class(model_params, **optimizer_args)

def Classes_to_consider(Number_of_classes):
    if Number_of_classes:
        classes_to_include = list(range(0, Number_of_classes))
    else:
        classes_to_include = None
    return classes_to_include