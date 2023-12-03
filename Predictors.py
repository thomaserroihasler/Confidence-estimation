import torch as tr
import torch.nn.functional as F
import torch.nn as nn

#### PREDICTION BLOCKS ####

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

    def prediction_function(self, input):
        raise NotImplementedError("Subclass must implement prediction function.")

    def forward(self, input):
        output = self.prediction_function(input)
        # Do additional processing here if needed
        return output

# standard classifier predictor

class MaximalLogitPredictor(Predictor):
    def __init__(self):
        super(MaximalLogitPredictor, self).__init__()

    def prediction_function(self, input):
        probabilities = F.softmax(input, dim=1)
        _, argmax = tr.max(probabilities, dim=1)
        return argmax

# useful for the confidence estimators

class IdentityPredictor(Predictor):
    def __init__(self):
        super(IdentityPredictor, self).__init__()

    def prediction_function(self, input):
        return input

