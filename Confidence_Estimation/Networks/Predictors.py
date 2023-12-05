import torch as tr
import torch.nn.functional as F
import torch.nn as nn

import torch as tr
import torch.nn.functional as F
import torch.nn as nn

#### PREDICTION BLOCKS ####

class Predictor(nn.Module):
    """
    Predictor: An abstract base class for prediction modules in neural networks.
    It defines a common interface for different prediction strategies.
    """
    def __init__(self):
        super(Predictor, self).__init__()

    def prediction_function(self, input):
        # Abstract method that needs to be implemented in subclasses
        raise NotImplementedError("Subclass must implement prediction function.")

    def forward(self, input):
        # Forward pass that uses the prediction function
        output = self.prediction_function(input)
        # Additional processing can be added here if needed
        return output

class MaximalLogitPredictor(Predictor):
    """
    MaximalLogitPredictor: A predictor that returns the class with the highest probability.
    It applies a softmax function to the input and returns the class with the highest probability.
    """
    def __init__(self):
        super(MaximalLogitPredictor, self).__init__()

    def prediction_function(self, input):
        # Applies softmax to input to get probabilities and returns the class with the highest probability
        probabilities = F.softmax(input, dim=1)
        _, argmax = tr.max(probabilities, dim=1)
        return argmax

class IdentityPredictor(Predictor):
    """
    IdentityPredictor: A predictor that simply returns the input as output.
    Useful for confidence estimators. It performs no modification to the input.
    """
    def __init__(self):
        super(IdentityPredictor, self).__init__()

    def prediction_function(self, input):
        # Directly returns the input without any modification
        return input
