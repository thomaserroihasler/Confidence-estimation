import torch as tr
import torch.nn as nn
import numpy as np
import sys
from sklearn.metrics import auc

new_path =  sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path
from Confidence_Estimation.Other.Useful_functions.definitions import handle_nan, Bin_edges, Single_bit_entropy_func

class TrueFalseMeasure(nn.Module):
    """
    TrueFalseMeasure class: computes accuracy as the fraction of elements in input1 and input2 that are equal.
    """
    def __init__(self):
        super(TrueFalseMeasure, self).__init__()

    def forward(self, input1, input2):
        # Compute the number of equal elements
        num_equal = (input1 == input2).sum().item()

        # Compute the total number of elements
        num_total = input1.numel()

        # Compute the fraction of elements that are equal
        true_false_measure = num_equal / num_total

        return true_false_measure

class CrossEntropy(nn.Module):
    """
    CrossEntropy class: calculates cross entropy loss assuming the inputs are already probabilities.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, predicted, target_pred):
        target = target_pred[:,0]
        # number of samples
        N = predicted.shape[0]

        # Convert the targets to one-hot encoding
        target_one_hot = tr.zeros_like(predicted)
        target_one_hot[tr.arange(N), target.long()] = 1

        # Compute cross entropy loss
        cross_entropy_loss = -tr.sum(target_one_hot * tr.log(predicted + 1e-9)) / N
        return cross_entropy_loss

class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    LabelSmoothedCrossEntropyLoss class: calculates Smoothed cross entropy.
    """
    def __init__(self, epsilon=0.1):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        n_classes = outputs.size(-1)
        log_preds = tr.nn.functional.log_softmax(outputs, dim=-1)
        loss = -log_preds.gather(dim=-1, index=targets.unsqueeze(-1))
        loss = loss.squeeze(-1) * (1-self.epsilon) - (self.epsilon / n_classes) * log_preds.sum(dim=-1)
        return loss.mean()

class ECE(nn.Module):
    """
    ECE class: calculates Expected Calibration Error (ECE) using defined bin edges.
    """
    def __init__(self, num_bins=None, num_per_bin=5):
        super(ECE, self).__init__()
        self.num_bins = num_bins
        self.num_per_bin = num_per_bin

    def forward(self, confidence, accuracy):

        sorted_indices = tr.argsort(confidence)
        confidence = confidence[sorted_indices]
        accuracy = accuracy[sorted_indices]
        self.bin_edges = Bin_edges(confidence, self.num_bins, self.num_per_bin)
        # Compute bin sizes, confidences, and accuracies
        bin_sizes, bin_confidences, bin_accuracies = self.calculate_bins(confidence, accuracy)

        # Calculate ECE loss
        ece_loss = ((bin_sizes.float() / len(confidence)) * tr.abs(bin_accuracies - bin_confidences)).sum()

        return ece_loss

    def calculate_bins(self, confidence, accuracy):
        # Bin sizes, confidences, and accuracies for each bin
        bin_sizes = tr.Tensor([((confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])).sum().item() for i in range(len(self.bin_edges)-1)])
        bin_confidences = tr.Tensor([confidence[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].mean().item() for i in range(len(self.bin_edges)-1)])
        bin_accuracies = tr.Tensor([accuracy[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].float().mean().item() for i in range(len(self.bin_edges)-1)])
        return bin_sizes, bin_confidences, bin_accuracies


class ECEWithProbabilities(nn.Module):
    """
    ECEWithProbabilities class: calculates Expected Calibration Error (ECE) using probabilities and actual predictions.
    """
    def __init__(self, num_bins=None, num_per_bin=5):
        super(ECEWithProbabilities, self).__init__()
        self.num_bins = num_bins
        self.num_per_bin = num_per_bin

    def forward(self, probabilities, target_pred):
        # Get the predicted class and confidence from probabilities
        labels = target_pred[:,0]
        predictions = target_pred[:,1]
        confidences = # probabilities along the direction of prediction
        accuracy = (predictions == labels).float()

        # Sort by confidence
        sorted_indices = tr.argsort(confidences)
        confidences = confidences[sorted_indices]
        accuracy = accuracy[sorted_indices]

        # Compute bin edges
        self.bin_edges = self._calculate_bin_edges(confidences)

        # Compute bin sizes, confidences, and accuracies
        bin_sizes, bin_confidences, bin_accuracies = self._calculate_bins(confidences, accuracy)

        # Calculate ECE loss
        ece_loss = ((bin_sizes.float() / len(confidences)) * tr.abs(bin_accuracies - bin_confidences)).sum()

        return ece_loss

    def _calculate_bin_edges(self, confidences):
        # Calculate the bin edges based on the confidence scores
        if self.num_bins is None:
            self.num_bins = int(len(confidences) / self.num_per_bin)
        bin_edges = tr.linspace(0, 1, steps=self.num_bins + 1)
        return bin_edges

    def _calculate_bins(self, confidences, accuracy):
        # Calculate bin sizes, confidences, and accuracies for each bin
        bin_sizes = tr.Tensor([((confidences > self.bin_edges[i]) & (confidences <= self.bin_edges[i+1])).sum().item() for i in range(len(self.bin_edges)-1)])
        bin_confidences = tr.Tensor([confidences[(confidences > self.bin_edges[i]) & (confidences <= self.bin_edges[i+1])].mean().item() for i in range(len(self.bin_edges)-1)])
        bin_accuracies = tr.Tensor([accuracy[(confidences > self.bin_edges[i]) & (confidences <= self.bin_edges[i+1])].mean().item() for i in range(len(self.bin_edges)-1)])
        return bin_sizes, bin_confidences, bin_accuracies



class MIE(nn.Module):
    """
    MIE class: calculates Expected Calibration Error (ECE) using defined bin edges.
    """
    def __init__(self, num_bins=None, num_per_bin=5):
        super(MIE, self).__init__()
        self.num_bins = num_bins
        self.num_per_bin = num_per_bin

    def forward(self, confidence, accuracy):
        sorted_indices = tr.argsort(confidence)
        confidence = confidence[sorted_indices]
        accuracy = accuracy[sorted_indices]
        self.bin_edges = Bin_edges(confidence, self.num_bins, self.num_per_bin)
        # Compute bin sizes, confidences, and accuracies
        bin_sizes, bin_confidences, bin_accuracies = self.calculate_bins(confidence, accuracy)

        # Calculate ECE loss
        mie_loss = ((bin_sizes.float() / len(confidence)) * Single_bit_entropy_func(bin_accuracies)).sum()

        return mie_loss

    def calculate_bins(self, confidence, accuracy):
        # Bin sizes, confidences, and accuracies for each bin
        bin_sizes = tr.Tensor([((confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])).sum().item() for i in range(len(self.bin_edges)-1)])
        bin_confidences = tr.Tensor([confidence[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].mean().item() for i in range(len(self.bin_edges)-1)])
        bin_accuracies = tr.Tensor([accuracy[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].float().mean().item() for i in range(len(self.bin_edges)-1)])
        return bin_sizes, bin_confidences, bin_accuracies




def AUROC(confidence_scores, accuracies):
    """
    Calculate the Area Under the Receiver Operating Characteristic (AUROC) curve.

    Args:
        confidence_scores: Model confidence scores.
        accuracies: Binary accuracy (0 or 1) for each score.
        num_thresholds: Number of thresholds to use for calculating the curve.

    Returns:
        False positive rate, true positive rate, and the AUROC value.
    """
    confidence_scores_np = confidence_scores.detach().cpu().numpy()
    accuracies_np = accuracies.detach().cpu().numpy()
    unique_scores = np.unique(confidence_scores_np)
    thresholds = sorted(unique_scores, reverse=False)
    tpr = handle_nan([np.mean(confidence_scores_np[accuracies_np == 1] >= t) for t in thresholds])
    fpr = handle_nan([np.mean(confidence_scores_np[accuracies_np == 0] >= t) for t in thresholds])
    return fpr, tpr, auc(fpr, tpr)

def AUPR(confidence_scores, accuracies):
    """
    Calculate the Area Under the Precision-Recall (AUPR) curve.

    Args:
        confidence_scores: Model confidence scores.
        accuracies: Binary accuracy (0 or 1) for each score.
        num_thresholds: Number of thresholds to use for calculating the curve.

    Returns:
        Recall, precision, and the AUPR value.
    """
    confidence_scores_np = confidence_scores.detach().cpu().numpy()
    accuracies_np = accuracies.detach().cpu().numpy()
    unique_scores = np.unique(confidence_scores_np)
    thresholds = sorted(unique_scores, reverse=False)
    precision = handle_nan([np.mean(accuracies_np[confidence_scores_np >= t]) for t in thresholds])
    recall = handle_nan([np.mean(confidence_scores_np[accuracies_np == 1] >= t) for t in thresholds])
    return recall, precision, auc(recall, precision)

def AURC(confidence_scores, accuracies):
    """
    Calculate the Area Under the Risk-Coverage (AURC) curve.

    Args:
        confidence_scores: Model confidence scores.
        accuracies: Binary accuracy (0 or 1) for each score.

    Returns:
        Risk-Coverage rate, error rate, and the AURC value.
    """
    confidence_scores_np = confidence_scores.detach().cpu().numpy()
    accuracies_np = accuracies.detach().cpu().numpy()
    unique_scores = np.unique(confidence_scores_np)
    thresholds = sorted(unique_scores, reverse=False)
    error = handle_nan([1- np.mean(accuracies_np[confidence_scores_np >= t]) for t in thresholds])
    rcr = handle_nan([np.sum(confidence_scores_np >= t) / len(confidence_scores_np) for t in thresholds])
    return rcr, error, auc(rcr, error)

def Brier_score(forecast_probabilities, actual_outcomes):
    """
    Calculate the Brier score for probabilistic forecasts.

    Args:
        forecast_probabilities: Probabilities forecast by the model.
        actual_outcomes: Actual outcomes (0 or 1).

    Returns:
        Brier score value.
    """
    forecast_probabilities_np = np.asarray(forecast_probabilities)
    actual_outcomes_np = np.asarray(actual_outcomes)
    brier_score = np.mean((forecast_probabilities_np - actual_outcomes_np) ** 2)
    return brier_score

class MSELossModule(nn.Module):
    """
    MSELossModule class: a wrapper for the Mean Squared Error loss function.
    """
    def __init__(self):
        super(MSELossModule, self).__init__()
        self.loss_function = nn.MSELoss()

    def forward(self, input, target):
        # Compute MSE loss between input and target
        loss = self.loss_function(input, target)
        return loss

class KLDivergenceModule(nn.Module):
    """
    KLDivergenceModule class: computes Kullback-Leibler divergence between input and target tensors.
    """
    def __init__(self):
        super(KLDivergenceModule, self).__init__()

    def forward(self, input, target):
        # Ensure the input and target tensors have the same shape
        assert input.shape == target.shape

        # Compute the KL-divergence between each pair of values in the input and target tensors
        kl_divs = tr.zeros_like(input)
        for i in range(input.shape[0]):
            p1 = tr.tensor([input[i], 1 - input[i]])
            p2 = tr.tensor([target[i], 1 - target[i]])
            kl_div = self.kl_div(p1, p2)
            kl_divs[i] = kl_div

        # Sum the KL-divergences and return the result
        total_kl_div = kl_divs.sum()
        return total_kl_div

    def kl_div(self, p1, p2):
        # Computes the Kullback-Leibler divergence between two probability distributions p1 and p2
        kl_div = tr.sum(p1 * tr.log(p1 / p2))
        return kl_div
