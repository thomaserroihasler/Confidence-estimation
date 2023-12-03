import torch.nn as nn
import torch as tr
from Stabilities_and_distributions import noise_relative_stability, coherency, coherency_probability

# Output based confidences

class Output_Confidence(nn.Module):
    def __init__(self,Probability = False):
        super(Output_Confidence, self).__init__()
        self.Probability = Probability
    def confidence_estimation(self,x):
        raise NotImplementedError("Subclasses must implement transform method")

    def probability_estimation(self,x):
        raise NotImplementedError("Subclasses must implement transform method")

    def forward(self, x):
        if(self.Probability):
            return self.probability_estimation(x)
        else:
            return self.confidence_estimation(x)

class Naive_confidence(Output_Confidence):
    def __init__(self,Probability = False):
        super(Naive_confidence, self).__init__(Probability)

    def confidence_estimation(self, x):
        Softmax = nn.Softmax(dim=1)
        x = Softmax(x)
        c = tr.max(x,1).values
        return c

    def probability_estimation(self, x):
        Softmax = nn.Softmax(dim=1)
        p = Softmax(x)
        return p


class TemperatureScaledConfidence(Output_Confidence):
    def __init__(self, temperature=1.0, Probability = False):
        super(TemperatureScaledConfidence, self).__init__(Probability)
        self.Temperature = nn.Parameter(tr.tensor(temperature))

    def confidence_estimation(self, x):
        softmax = nn.Softmax(dim=-1)
        p_T = softmax(x / self.Temperature)
        c = tr.max(p_T,dim = 1).values
        return c

    def probability_estimation(self, x):
        softmax = nn.Softmax(dim=-1)
        p = softmax(x / self.Temperature)
        return p

class AveragetemperatureScaledConfidence(Output_Confidence):
    def __init__(self,temperature=1.0,New_predictions = False,Probability = False):
        super(AveragetemperatureScaledConfidence, self).__init__(Probability)
        self.Temperature = nn.Parameter(tr.tensor(temperature))
        self.New_predictions = New_predictions

    def confidence_estimation(self, x):
        new_x = x[:, :-1, :]
        softmax = nn.Softmax(dim=-1)
        x_T = softmax(new_x / self.Temperature)
        p_T = tr.mean(x_T,dim=1)

        if (self.New_predictions):
            self.predicitions = tr.argmax(p_T,dim = -1)
        else:
            self.predicitions = tr.argmax(x[:, -1, :], dim=-1)
        indices_0 = tr.arange(p_T.size(0))
        c = p_T[indices_0.long(),(self.predicitions).long()]

        return c

    def probability_estimation(self, x):
        new_x = x[:, :-1, :]
        softmax = nn.Softmax(dim=-1)
        x_T = softmax(new_x / self.Temperature)
        p_T = tr.mean(x_T, dim=1)
        if (self.New_predictions):
            self.predicitions = tr.argmax(p_T, dim=-1)
        else:
            self.predicitions = tr.argmax(x[:, -1, :], dim=-1)
        indices_0 = tr.arange(p_T.size(0))
        c = p_T[indices_0.long(), (self.predicitions).long()]
        return p_T


class HybridtemperaturescaledConfidence(Output_Confidence):
    def __init__(self, temperature=1.0, temperature2=1.0, Probability = False):
        super(HybridtemperaturescaledConfidence, self).__init__(Probability)
        self.Temperature = nn.Parameter(tr.tensor(temperature))
        self.Temperature2 = nn.Parameter(tr.tensor(temperature2))

    def confidence_estimation(self, x):
        # split in 2 and apply the temperature scaling to each block
        x_1 = x[:,0:-1,0:-1]
        x_2 = x[:,-1,0:-1]
        predictions = x[:,-1,-1]
        softmax = nn.Softmax(dim=-1)
        p_x_1 = tr.mean(softmax(x_1 / self.Temperature), dim=1)
        p_x_2 = softmax(x_2 / self.Temperature2)
        p_x = 1/2 * (p_x_2 + p_x_1)
        indices_0 = tr.arange(p_x.size(0))
        c = p_x[indices_0.long(), (predictions).long()]
        return c

    def probability_estimation(self, x):
        x_1 = x[:, 0:-1, 0:-1]
        x_2 = x[:, -1, 0:-1]
        predictions = x[:, -1, -1]
        softmax = nn.Softmax(dim=-1)
        p_x_1 = tr.mean(softmax(x_1 / self.Temperature), dim=1)
        p_x_2 = softmax(x_2 / self.Temperature2)
        p_x = 1 / 2 * (p_x_2 + p_x_1)
        indices_0 = tr.arange(p_x.size(0))
        return p_x


class AveragetemperatureScaledProbability(Output_Confidence):
    def __init__(self, temperature=1.0):
        super(AveragetemperatureScaledProbability, self).__init__()
        self.Temperature = nn.Parameter(tr.tensor(temperature))

    # def confidence_estimation(self, x):
    #     predictions = x[:,0,-1]
    #     new_x = x[:,:,0:-1]
    #     softmax = nn.Softmax(dim=-1)
    #     t_x = softmax(new_x / self.Temperature)
    #     p_t = tr.mean(t_x,dim=1)
    #     indices_0 = tr.arange(p_t.size(0))
    #     c = p_t[indices_0.long(),(predictions).long()]
    #     return c

    def confidence_estimation(self, x):
        new_x = x[:, :, 0:-1]
        softmax = nn.Softmax(dim=-1)
        t_x = softmax(new_x / self.Temperature)
        p_t = tr.mean(t_x, dim=1)
        return p_t


# input based confidences

def Coherency_confidence(transformations, network, x):
    return coherency(transformations, network, x)

def Probability_coherency_confidences(transformations, network, x):
    return coherency_probability(transformations, network, x)

def Noise_relative_stability_confidence(transformations, network, x,normalization_function):
    return (noise_relative_stability(transformations, network, x))

# hybrid confidences

class hybrid_Confidence(nn.Module):
    def __init__(self):
        super(Output_Confidence, self).__init__()

    def hybridization(self,p1,p2):
        raise NotImplementedError("Subclasses must implement transform method")

    def forward(self, p1,p2):
        return self.hybridization(p1,p2)


class power_law_decay(nn.Module):
    def __init__(self,linear_parameter,power):
        super(power_law_decay, self).__init__()
        self.linear_parameter = linear_parameter
        self.power = power
    def hybridization(self,x):
        raise

    def forward(self, p1,p2):
        return self.hybridization(p1,p2)