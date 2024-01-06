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
        prediction, score = self.prediction_function(input)
        # Additional processing can be added here if needed
        return prediction, score

class MaximalLogitPredictor(Predictor):
    """
    MaximalLogitPredictor: A predictor that returns the class with the highest probability.
    It applies a softmax function to the input and returns the class with the highest probability.
    """
    def __init__(self):
        super(MaximalLogitPredictor, self).__init__()

    def prediction_function(self, input):
        # Applies softmax to input to get probabilities and returns the class with the highest probability
        probabilities = F.softmax(input, dim=-1)
        score , prediction = tr.max(probabilities, dim=-1)
        return prediction, score
