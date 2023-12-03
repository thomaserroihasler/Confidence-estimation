from DataProcessing import
from Functions import handle_nan
import torch as tr
import torch.nn as nn
import numpy as np
from sklearn.metrics import auc

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

    def forward(self, predicted, target):
        # number of samples
        N = predicted.shape[0]

        # Convert the targets to one-hot encoding
        target_one_hot = tr.zeros_like(predicted)
        target_one_hot[tr.arange(N), target.long()] = 1

        # Compute cross entropy loss
        cross_entropy_loss = -tr.sum(target_one_hot * tr.log(predicted + 1e-9)) / N
        return cross_entropy_loss

class ECE(nn.Module):
    """
    ECE class: calculates Expected Calibration Error (ECE) using defined bin edges.
    """
    def __init__(self, bin_edges):
        super(ECE, self).__init__()
        self.bin_edges = bin_edges
        self.num_bins = len(bin_edges) - 1

    def forward(self, confidence, accuracy):
        sorted_indices = tr.argsort(confidence)
        confidence = confidence[sorted_indices]
        accuracy = accuracy[sorted_indices]

        # Compute bin sizes, confidences, and accuracies
        bin_sizes, bin_confidences, bin_accuracies = self.calculate_bins(confidence, accuracy)

        # Calculate ECE loss
        ece_loss = ((bin_sizes.float() / len(confidence)) * tr.abs(bin_accuracies - bin_confidences)).sum()

        return ece_loss

    def calculate_bins(self, confidence, accuracy):
        # Bin sizes, confidences, and accuracies for each bin
        bin_sizes = tr.Tensor([((confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])).sum().item() for i in range(self.num_bins)])
        bin_confidences = tr.Tensor([confidence[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].mean().item() for i in range(self.num_bins)])
        bin_accuracies = tr.Tensor([accuracy[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].mean().item() for i in range(self.num_bins)])
        return bin_sizes, bin_confidences, bin_accuracies

class ECE_probability(nn.Module):
    """
    ECE_probability class: calculates Expected Calibration Error (ECE) without considering the edge case.
    """
    def __init__(self, num_bins=None, num_per_bin=50, New_prediction=False):
        super(ECE_probability, self).__init__()
        self.num_bins = num_bins
        self.num_per_bin = num_per_bin
        self.New_prediction = New_prediction

    def forward(self, outputs, labels):
        # Select probabilities tensor based on New_prediction
        probabilities, indices = outputs
        confidence = self.get_confidence(probabilities, indices)

        # Predicted labels based on probabilities
        predicted_labels = tr.argmax(probabilities, dim=-1)

        # Calculate accuracy and sort inputs by confidence
        accuracy = (predicted_labels == labels).float()
        confidence, bin_edges, accuracy = self.sort_and_bin(confidence, accuracy)

        # Compute ECE loss
        ece_loss = self.calculate_ece(confidence, accuracy, bin_edges)

        return ece_loss

    def get_confidence(self, probabilities, indices):
        # Extract confidence values from the predicted probabilities
        indices_0 = tr.arange(probabilities.size(0))
        return probabilities[indices_0.long(), indices.long()]

    def sort_and_bin(self, confidence, accuracy):
        # Sort confidence and determine bin edges
        confidence, sorted_indices = tr.sort(confidence)
        bin_edges = Binning_method(confidence, self.num_bins, self.num_per_bin)
        accuracy = accuracy[sorted_indices]
        return confidence, bin_edges, accuracy

    def calculate_ece(self, confidence, accuracy, bin_edges):
        # Calculate ECE loss using bin sizes, confidences, and accuracies
        N_bins = len(bin_edges) - 1
        bin_confidences, bin_accuracies, bin_sizes = [tr.zeros(N_bins) for _ in range(3)]

        for i in range(N_bins):
            mask = self.get_bin_mask(confidence, bin_edges, i)
            if mask.sum() > 0:
                bin_confidences[i] = confidence[mask].mean()
                bin_accuracies[i] = accuracy[mask].mean()
                bin_sizes[i] = mask.sum()

        bin_sizes = bin_sizes.float()
        ece_loss = (bin_sizes / len(confidence)).mul(tr.abs(bin_accuracies - bin_confidences)).sum()
        return ece_loss

    def get_bin_mask(self, confidence, bin_edges, i):
        # Create mask for each bin range
        if i == 0:
            return (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
        else:
            return (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])

# Similar structure and logic is applied for the ECE_no_edge and MIE classes.
# The ECE_no_edge class calculates ECE without considering edge cases, and the MIE class estimates Mutual Information.
# Various utility functions like get_2Dhjoint_and_edges, get_1Dhjoint_and_edges, get_bins, get_1Daccuracy, and get_2Daccuracy are provided to support these calculations.


def calculate_auroc(confidence_scores, accuracies):
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

def calculate_aupr(confidence_scores, accuracies):
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

def calculate_aurc(confidence_scores, accuracies):
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

def calculate_brier_score(forecast_probabilities, actual_outcomes):
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
