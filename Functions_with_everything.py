
import torch.nn as nn
import torch as tr
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
import math as mt
import torch as tr
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from DataProcessing import Binning_method
from abc import ABC, abstractmethod
from scipy.signal import argrelextrema

import datetime
from sklearn.kernel_ridge import KernelRidge


import torch as tr
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def handle_nan(arr):
    return np.nan_to_num(arr, nan=0)


def get_accuracy(loader, model,device):
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with tr.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)
            _, predicted = tr.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set model back to training mode
    return 100 * correct / total


def calculate_auroc(confidence_scores, accuracies, num_thresholds=10000):
    confidence_scores_np = confidence_scores.detach().cpu().numpy()
    accuracies_np = accuracies.detach().cpu().numpy()
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr = handle_nan([np.mean(confidence_scores_np[accuracies_np == 1] >= t) for t in thresholds])
    fpr = handle_nan([np.mean(confidence_scores_np[accuracies_np == 0] >= t) for t in thresholds])
    return fpr, tpr, auc(fpr, tpr)

def calculate_aupr(confidence_scores, accuracies, num_thresholds=10000):
    confidence_scores_np = confidence_scores.detach().cpu().numpy()
    accuracies_np = accuracies.detach().cpu().numpy()
    thresholds = np.linspace(0, 1, num_thresholds)
    precision = handle_nan([np.mean(accuracies_np[confidence_scores_np >= t]) for t in thresholds])
    recall = handle_nan([np.mean(confidence_scores_np[accuracies_np == 1] >= t) for t in thresholds])
    return recall, precision, auc(recall, precision)
# #
# def calculate_aurc(confidence_scores, accuracies, num_thresholds=100000):
#     confidence_scores_np = confidence_scores.detach().cpu().numpy()
#     accuracies_np = accuracies.detach().cpu().numpy()
#     thresholds = np.linspace(0, 1, num_thresholds)
#     error = handle_nan([1- np.mean(accuracies_np[confidence_scores_np >= t]) for t in thresholds])
#     rcr = handle_nan([np.sum(confidence_scores_np >= t)/(np.sum(confidence_scores_np >= t)+np.sum(confidence_scores_np < t)) for t in thresholds])
#     return rcr, error, auc(rcr, error)

def calculate_aurc(confidence_scores, accuracies):
    confidence_scores_np = confidence_scores.detach().cpu().numpy()
    accuracies_np = accuracies.detach().cpu().numpy()
    unique_scores = np.unique(confidence_scores_np)
    # Sort in descending order
    thresholds = sorted(unique_scores, reverse=False)
    error = handle_nan([1- np.mean(accuracies_np[confidence_scores_np >= t]) for t in thresholds])
    rcr = handle_nan([np.sum(confidence_scores_np >= t) / len(confidence_scores_np) for t in thresholds])
    return rcr, error, auc(rcr, error)

import numpy as np

def calculate_brier_score(forecast_probabilities, actual_outcomes):
    # Ensure that the inputs are NumPy arrays
    forecast_probabilities_np = np.asarray(forecast_probabilities)
    actual_outcomes_np = np.asarray(actual_outcomes)

    # Calculate the mean squared difference between predicted probabilities and actual outcomes
    brier_score = np.mean((forecast_probabilities_np - actual_outcomes_np) ** 2)

    return brier_score


#
# def calculate_auroc_and_plot(confidence_scores, accuracies):
#     # Detach the tensors and convert to NumPy arrays
#     confidence_scores_np = confidence_scores.detach().cpu().numpy()
#     accuracies_np = accuracies.detach().cpu().numpy()
#
#     # Calculate the ROC curve points
#     fpr, tpr, thresholds = roc_curve(accuracies_np, confidence_scores_np)
#     print(len(thresholds))
#     # Calculate the area under the ROC curve
#     auroc = auc(fpr, tpr)
#
#     # # Plot the ROC curve
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
#     # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#     # plt.xlim([0.0, 1.0])
#     # plt.ylim([0.0, 1.05])
#     # plt.xlabel('False Positive Rate')
#     # plt.ylabel('True Positive Rate')
#     # plt.title('Receiver Operating Characteristic')
#     # plt.legend(loc="lower right")
#     # plt.grid()
#     # plt.show()
#
#     return auroc  # Return the AUC value if needed


def Single_bit_entropy_func(a):
    return -1 * ((a * tr.log(a + 1e-5) + (1 - a) * tr.log((1 - a + 1e-5))))


def monte_carlo_integral(f, points):
    # Apply the function to each point and sum the results
    integral_approx = tr.sum(f(points))

    # Normalize by the number of points
    integral_approx /= len(points)

    return integral_approx


def simple_integral(f, x):
    y = f(x)

    dx = x[1:] - x[:-1]
    integral = tr.sum(y[:-1] * dx)
    return integral.item()


def simple_2D_integral(f, N):
    # Initialize a grid of N points in the interval [0, 1]^2
    x = tr.linspace(0, 1, N)
    y = tr.linspace(0, 1, N)

    # Create a 2D grid
    X, Y = tr.meshgrid(x, y)

    # Reshape the 2D grid to a batch of 2D points
    points = tr.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # Apply the function to each point and sum the results
    #print('point shape',(points).shape)
    integral_approx = tr.sum(f(points))
    #print(integral_approx)
    # Normalize by the number of points
    integral_approx /= N ** 2

    return integral_approx

def Integrate(x, y1, y2):
    product = y1 - y2
    integral = tr.trapz(product, x)
    return integral

# K Nearest Neighbors

def kNN(values, x, k):
    """Find the k nearest neighbors."""

    # Check type of values and x and convert to tensors if necessary
    if type(values) is list:
        values = tr.tensor(values)
    if type(x) is list:
        x = tr.tensor(x)

    if len(values.shape) == 1:
        values = values.unsqueeze(-1)
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)

    #print('values and x shape', values.shape, x.shape)

    # Expand dims to perform broadcasting
    values = values.unsqueeze(0)
    x = x.unsqueeze(1)

    #print('values and x shape', values.shape, x.shape)
    # Ensure the broadcasting is correctly done
    if values.shape[2] != x.shape[2]:
        raise ValueError("Last dimensions of values and x must be equal")

    # Compute the Euclidean distance using broadcasting
    dist = tr.sqrt(((values - x) ** 2).sum(-1))

    # Get the k nearest points
    distances, knn_indices = dist.topk(k, dim=1, largest=False, sorted=False)

    return knn_indices, values.squeeze()[knn_indices], distances


class FunctionApproximator(ABC, nn.Module):
    """
    Base class for function approximators.
    """

    def __init__(self, inputs, outputs, Remove=False):
        super(FunctionApproximator, self).__init__()

        self.inputs = self._squeeze_trivial_dims(inputs)
        self.outputs = self._squeeze_trivial_dims(outputs)
        self.Remove = Remove

    def _squeeze_trivial_dims(self, tensor):
        if tensor.dim() == 2 and (tensor.shape[0] == 1 or tensor.shape[1] == 1):
            tensor = tensor.squeeze()
        return tensor

    def _check_and_convert_tensor(self, x):
        if not isinstance(x, tr.Tensor):
            x = tr.tensor([x], dtype=tr.float32)

        if x.dtype != tr.float32:
            x = x.float()

        return x

    @abstractmethod
    def Forward(self, x):
        pass

    @abstractmethod
    def remove(self, x,result):
        pass

    def forward(self, x):
        x = self._check_and_convert_tensor(x)
        result = self.Forward(x)
        result =  self.remove(x,result)
        # if self.mini:
        #     mask = x < self.inputs.min()
        #     result[mask] = 0.0
        return result.flatten()

class Function_composition(nn.Module):
    """
    Function composition class.
    """

    def __init__(self, function_approximators, Remove=False):
        super(Function_composition, self).__init__()
        self.function_approximators = nn.ModuleList(function_approximators)
        self.Remove = Remove

    def forward(self, x):

        if self.Remove:
            mask = x <= self.function_approximators[-1].inputs.min()
            #print('in the mask range', self.function_approximators[-1].inputs.min(), x, mask)

        else:
            mask = tr.zeros_like(x, dtype=tr.bool)
            #print(mask)
        for function_approximator in self.function_approximators:
            x = (function_approximator(x)).float()

        x[mask] = 0.0

        return x

#         # Compute joint histogram
#         h_joint, _ = np.histogramdd((confidence1.numpy(), confidence2.numpy(), accuracy.numpy()),
#                                         bins=[conf1_bin_edges.numpy(), conf2_bin_edges.numpy(), self.acc_bin_edges.numpy()])

import numpy as np
import torch as tr

class Predictor(nn.Module):
    def __init__(self,New_prediction = False):
        super(Predictor, self).__init__()
        self.New_prediction = New_prediction
    @abstractmethod
    def predict(self, inputs):
        pass

    def forward(self,inputs,naive_inputs):
        if self.New_prediction:
            return self.predict(inputs)
        else:
            return self.predict(naive_inputs)

class ArgmaxPredictor(Predictor):
    def __init__(self, New_prediction = False):
        super(ArgmaxPredictor, self).__init__(New_prediction)

    def predict(self, inputs):
        return tr.argmax(inputs, dim=-1)

import numpy as np
import torch as tr
from torch import nn
import torch.nn.functional as F
#
class Box_accuracy(nn.Module):

    def __init__(self, confidence1, confidence2, accuracy, num_bins):
        super().__init__()
        self.num_bins = num_bins

        # Combine confidences and standardize
        confidences = tr.stack([confidence1, confidence2], dim=1).numpy()
        self.mean = np.mean(confidences, axis=0, keepdims=True)
        self.std = np.std(confidences, axis=0, keepdims=True)
        self.confidences = (confidences - self.mean) / self.std
        self.accuracy = accuracy

        # Compute the bin edges using the Binning_method function
        self.xedges = Binning_method(tr.tensor(self.confidences[:, 0]), num_bins=int(np.sqrt(self.num_bins)))
        self.yedges = Binning_method(tr.tensor(self.confidences[:, 1]), num_bins=int(np.sqrt(self.num_bins)))

        # Compute the bin number for each point
        self.binnumber = np.ravel_multi_index([
            np.clip(np.digitize(self.confidences[:, 0], self.xedges) - 1, 0, len(self.xedges)-2),
            np.clip(np.digitize(self.confidences[:, 1], self.yedges) - 1, 0, len(self.yedges)-2)
        ], (len(self.xedges)-1, len(self.yedges)-1))

        # Compute the average accuracy for each bin
        self.bin_accuracies = tr.tensor([
            accuracy[self.binnumber == i].mean().item() if np.sum(self.binnumber == i) > 0 else 0.0
            for i in range((len(self.xedges)-1) * (len(self.yedges)-1))
        ], dtype=tr.float32)


    def forward(self, x):
        # Standardize the input tensor x
        x = (x - self.mean) / self.std

        # Convert x to numpy array for querying with KDTree
        x_np = x.numpy()

        # Compute the bin number for each point in x
        binnumber = np.ravel_multi_index([
            np.clip(np.digitize(x_np[:, 0], self.xedges) - 1, 0, len(self.xedges)-2),
            np.clip(np.digitize(x_np[:, 1], self.yedges) - 1, 0, len(self.yedges)-2)
        ], (len(self.xedges)-1, len(self.yedges)-1))

        # Return the accuracy of each bin that each data point belongs to
        return self.bin_accuracies[binnumber]

# class Box_accuracy(nn.Module):
#
#     def __init__(self, confidence1, confidence2, accuracy, num_bins):
#         super().__init__()
#         self.num_bins = num_bins
#
#         # Combine confidences and standardize
#         confidences = tr.stack([confidence1, confidence2], dim=1)
#         self.mean = tr.mean(confidences, dim=0, keepdim=True)
#         self.std = tr.std(confidences, dim=0, keepdim=True)
#         self.confidences = (confidences - self.mean) / self.std
#         self.accuracy = accuracy
#
#         # Compute the bin edges using the Binning_method function
#         self.xedges = Binning_method(self.confidences[:, 0], num_bins=int(self.num_bins ** 0.5))
#         self.yedges = Binning_method(self.confidences[:, 1], num_bins=int(self.num_bins ** 0.5))
#
#         # Compute the bin number for each point
#         self.binnumber = self.ravel_multi_index(
#             tr.stack([
#                 self.digitize(self.confidences[:, 0], self.xedges),
#                 self.digitize(self.confidences[:, 1], self.yedges)
#             ]),
#             (len(self.xedges) - 1, len(self.yedges) - 1)
#         )
#
#         # Compute the average accuracy for each bin
#         self.bin_accuracies = tr.tensor([
#             accuracy[self.binnumber == i].mean().item() if tr.sum(self.binnumber == i) > 0 else 0.0
#             for i in range((len(self.xedges) - 1) * (len(self.yedges) - 1))
#         ])
#
#     def forward(self, x):
#         # Standardize the input tensor x
#         x = (x - self.mean) / self.std
#
#         # Compute the bin number for each point in x
#         binnumber = self.ravel_multi_index(
#             tr.stack([
#                 self.digitize(x[:, 0], self.xedges),
#                 self.digitize(x[:, 1], self.yedges)
#             ]),
#             (len(self.xedges) - 1, len(self.yedges) - 1)
#         )
#
#         # Return the accuracy of each bin that each data point belongs to
#         return self.bin_accuracies[binnumber]
#
#     @staticmethod
#     def digitize(x, bins):
#         return tr.bucketize(x, bins, right=False)
#
#     @staticmethod
#     def ravel_multi_index(coords, dims):
#         strides = tr.cumprod(tr.tensor([1] + list(dims[::-1][:-1])), dim=0)[::-1]
#         return tr.sum(coords * strides, dim=0)


#
#
# class Box_accuracy(tr.nn.Module):
#     """
#     Box_accuracy class: A function approximator using histogram binning for 3D inputs.
#     """
#
#     def __init__(self, confidence1, confidence2, accuracy, num_bins, num_per_bin = None):
#         super(Box_accuracy, self).__init__()
#
#         # Standardize the confidences
#         self.conf1_mean = confidence1.mean()
#         self.conf1_std = confidence1.std()
#         confidence1 = (confidence1 - self.conf1_mean) / self.conf1_std
#
#         self.conf2_mean = confidence2.mean()
#         self.conf2_std = confidence2.std()
#         confidence2 = (confidence2 - self.conf2_mean) / self.conf2_std
#
#         # Compute bin edges
#         conf1_bin_edges = Binning_method(confidence1, num_bins, num_per_bin)
#         conf2_bin_edges = Binning_method(confidence2, num_bins, num_per_bin)
#
#         # Calculate h_joint here using input data
#         h_joint = self._calculate_histogramdd(confidence1, confidence2, accuracy, conf1_bin_edges, conf2_bin_edges)
#
#         self.bin_edges = [conf1_bin_edges.clone().detach(), conf2_bin_edges.clone().detach()]
#         self.h_joint = tr.tensor(h_joint)
#         self.h_joint_avg = self.h_joint[:, :, 1] / (tr.sum(self.h_joint, dim=2)+1e-7)
#         print(self.bin_edges)
#
#     def forward(self, x):
#         x = self._check_and_convert_tensor(x)
#
#         # Standardize x using the stored means and standard deviations
#         x[:, 0] = (x[:, 0] - self.conf1_mean) / self.conf1_std
#         x[:, 1] = (x[:, 1] - self.conf2_mean) / self.conf2_std
#
#         print(x.shape)
#         i = tr.bucketize(x[:, 0], self.bin_edges[0], right=False)
#         j = tr.bucketize(x[:, 1], self.bin_edges[1], right=False)
#
#         i = tr.clamp(i, max=self.h_joint_avg.shape[0] - 1)
#         j = tr.clamp(j, max=self.h_joint_avg.shape[1] - 1)
#         print(i, j)
#         return self.h_joint_avg[i, j]
#
#     def _check_and_convert_tensor(self, x):
#         if not isinstance(x, tr.Tensor):
#             x = tr.tensor(x, dtype=tr.float32)
#
#         if x.dtype != tr.float32:
#             x = x.float()
#
#         return x
#
#     def _calculate_histogramdd(self, confidence1, confidence2, accuracy, conf1_bin_edges, conf2_bin_edges):
#         # Convert input tensors to numpy
#         confidence1 = confidence1.numpy()
#         confidence2 = confidence2.numpy()
#         accuracy = accuracy.numpy()
#
#         # Use numpy.histogramdd to calculate the joint histogram
#         h_joint, _ = np.histogramdd((confidence1, confidence2, accuracy),bins=(conf1_bin_edges, conf2_bin_edges, np.array([0., 0.5, 1.])))
#         return h_joint


# VoronoiAccuracy class definition


from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
#from shapely.geometry import Polygon
#
#
# def voronoi_finite_polygons_2d(vor, radius=None):
#     """
#     Reconstruct infinite Voronoi regions in a 2D diagram to finite
#     regions.
#
#     Parameters
#     ----------
#     vor : Voronoi
#         Input diagram
#     radius : float, optional
#         Distance to 'points at infinity'.
#
#     Returns
#     -------
#     regions : list of tuples
#         Indices of vertices in each revised Voronoi regions.
#     vertices : list of tuples
#         Coordinates for revised Voronoi vertices. Same as coordinates
#         of input vertices, with 'points at infinity' appended to the
#         end.
#
#     """
#
#     if vor.points.shape[1] != 2:
#         raise ValueError("Requires 2D input")
#
#     new_regions = []
#     new_vertices = vor.vertices.tolist()
#
#     center = vor.points.mean(axis=0)
#     if radius is None:
#         radius = vor.points.ptp().max()
#
#     # Construct a map containing all ridges for a given point
#     all_ridges = {}
#     for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
#         all_ridges.setdefault(p1, []).append((p2, v1, v2))
#         all_ridges.setdefault(p2, []).append((p1, v1, v2))
#
#     # Reconstruct infinite regions
#     for p1, region in enumerate(vor.point_region):
#         vertices = vor.regions[region]
#
#         if all(v >= 0 for v in vertices):
#             # finite region
#             new_regions.append(vertices)
#             continue
#
#         # reconstruct a non-finite region
#         ridges = all_ridges[p1]
#         new_region = [v for v in vertices if v >= 0]
#
#         for p2, v1, v2 in ridges:
#             if v2 < 0:
#                 v1, v2 = v2, v1
#             if v1 >= 0:
#                 # finite ridge: already in the region
#                 continue
#
#             # Compute the missing endpoint of an infinite ridge
#             t = vor.points[p2] - vor.points[p1]  # tangent
#             t /= np.linalg.norm(t)
#             n = np.array([-t[1], t[0]])  # normal
#
#             midpoint = vor.points[[p1, p2]].mean(axis=0)
#             direction = np.sign(np.dot(midpoint - center, n)) * n
#             far_point = vor.vertices[v2] + direction * radius
#
#             new_region.append(len(new_vertices))
#             new_vertices.append(far_point.tolist())
#
#         # sort region counterclockwise
#         vs = np.asarray([new_vertices[v] for v in new_region])
#         c = vs.mean(axis=0)
#         angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
#         new_region = np.array(new_region)[np.argsort(angles)]
#
#         # finish
#         new_regions.append(new_region.tolist())
#
#     return new_regions, np.asarray(new_vertices)


from scipy.spatial import Voronoi

from sklearn.cluster import KMeans
from scipy.spatial import Voronoi

#
# class Voronoi_accuracy(nn.Module):
#
#     def __init__(self, confidence1, confidence2, accuracy, num_bins, N_vor):
#         super().__init__()
#         self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
#         self.num_bins = num_bins
#         self.N_vor = N_vor
#
#         # Combine confidences and standardize
#         confidences = tr.stack([confidence1, confidence2], dim=1).numpy()
#         self.mean = np.mean(confidences, axis=0, keepdims=True)
#         self.std = np.std(confidences, axis=0, keepdims=True)
#         self.confidences = confidences_standardized = (confidences - self.mean) / self.std
#         self.accuracy = accuracy
#
#         self.trees = []
#         self.bin_accuracies_list = []
#
#         # Run multiple Voronoi and KMeans simulations
#         for _ in range(self.N_vor):
#             # KMeans for binning
#             kmeans = KMeans(n_clusters=self.num_bins, n_init=10).fit(self.confidences)
#             vor = Voronoi(kmeans.cluster_centers_)
#             # Create a KDTree with the Voronoi centroids
#             tree = cKDTree(kmeans.cluster_centers_)
#             self.trees.append(tree)
#
#             # Assign each point to a bin according to Voronoi tessellation
#             _, bins = tree.query(confidences_standardized)
#
#             # Compute the average accuracy for each bin
#             bin_accuracies = tr.tensor([accuracy[bins == i].mean().item() for i in range(self.num_bins)],
#                                        dtype=tr.float32)
#             self.bin_accuracies_list.append(bin_accuracies)
#
#     def forward(self, x):
#         # Standardize the input tensor x
#         x = (x - self.mean) / self.std
#
#         # Convert x to numpy array for querying with KDTree
#         x_np = x.numpy()
#
#         avg_bin_accuracies = 0
#
#         # Loop over all simulations and get the accuracy for each point in x
#         for i in range(self.N_vor):
#             # Assign each point in x to a bin according to Voronoi tessellation
#             _, bins = self.trees[i].query(x_np)
#
#             # Add the accuracy of each bin that each data point belongs to
#             avg_bin_accuracies += self.bin_accuracies_list[i][bins]
#
#         # Return the average accuracy
#         return avg_bin_accuracies / self.N_vor
#
#
#     def plot_voronoi_tesselation(self, vor, points, accuracies):
#
#         #plt.figure(figsize=(10, 10))
#         voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_colors='orange')
#
#         # Create a KDTree with the Voronoi centroids
#         tree = cKDTree(vor.points)
#
#         # Assign each point to a bin according to Voronoi tessellation
#         _, bins = tree.query(points)
#
#         # Compute the average accuracy for each bin
#         bin_accuracies = np.array([accuracies[bins == i].mean().item() for i in range(len(vor.points))])
#
#         new_regions, new_vertices = voronoi_finite_polygons_2d(vor)
#
#
#         # Get Voronoi polygons
#         polygons = [Polygon(new_vertices[region]) for region in new_regions]
#
#         # Continue as before...
#
#         # Get Voronoi polygons
#         #polygons = [vor.vertices[region] for region in vor.regions if region and -1 not in region]
#
#         # Create a colormap
#         cmap = plt.cm.viridis
#
#         # Plot filled polygons
#         for polygon, accuracy in zip(polygons, bin_accuracies):
#             plt.fill(*zip(*polygon.exterior.coords), color=cmap(accuracy))
#
#         # Scatter colors based on Voronoi cell
#         #scatter_colors = [cmap(b) for b in bins]
#
#         # Generate color list
#         colors = ['blue' if acc else 'red' for acc in accuracies]
#
#         plt.scatter(points[:, 0], points[:, 1], c=colors, edgecolor='black', alpha=0.9)
#
#         # Add the centroids
#         # Add the centroids
#         centroids = np.array([polygon.centroid.coords[0] for polygon in polygons])
#         plt.scatter(centroids[:, 0], centroids[:, 1], c='white', marker='x')
#
#         # Expanding the x and y limits of the plot
#         plt.xlim(points[:, 0].min() , points[:, 0].max())
#         plt.ylim(points[:, 1].min() , points[:, 1].max())
#
#         # Show the colorbar
#         plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
#         plt.title("Voronoi tessellation with cells colored by bin accuracy")
#         plt.show()
#         plt.close()
#
#
#
# class Voronoi_accuracy(tr.nn.Module):
#     def __init__(self, confidence1, confidence2, accuracy, num_bins):
#         super(Voronoi_accuracy, self).__init__()
#
#         # Combine confidences
#         confidences = tr.stack([confidence1, confidence2], dim=1).numpy()
#
#         # Compute mean and std of confidences
#         self.mean = np.mean(confidences, axis=0, keepdims=True)
#         self.std = np.std(confidences, axis=0, keepdims=True)
#
#         # Standardize confidences
#         confidences_standardized = (confidences - self.mean) / self.std
#
#         # KMeans for seeds of Voronoi
#         kmeans = KMeans(n_clusters=num_bins, n_init=10).fit(confidences_standardized)
#
#         # Voronoi for binning and Create a KDTree with the Voronoi centroids
#         self.tree = cKDTree(kmeans.cluster_centers_)
#
#         # Assign each point to a bin according to Voronoi tessellation
#         _, bins = self.tree.query(confidences_standardized)
#
#         # Compute the average accuracy for each bin
#         self.bin_accuracies = tr.tensor([accuracy[bins == i].mean().item() for i in range(num_bins)], dtype=tr.float32)
#
#     def forward(self, confidence_vector):
#         # Combine and standardize confidences
#         confidences = confidence_vector.numpy()
#         confidences_standardized = (confidences - self.mean) / self.std
#
#         # Assign each point to a bin according to Voronoi tessellation
#         _, bins = self.tree.query(confidences_standardized)
#
#         # Get the average accuracy for each bin
#         accuracies = self.bin_accuracies[bins]
#
#         return accuracies


class NearestNeighborModule(nn.Module):
    def __init__(self, input_tensor, accuracy_tensor):
        super(NearestNeighborModule, self).__init__()
        # Ensure that input_tensor and accuracy_tensor are torch tensors.
        if not tr.is_tensor(input_tensor):
            input_tensor = tr.tensor(input_tensor)
        if not tr.is_tensor(accuracy_tensor):
            accuracy_tensor = tr.tensor(accuracy_tensor)

        assert input_tensor.shape == accuracy_tensor.shape, "input_tensor and accuracy_tensor must be of the same size"

        # Initialize tensors as buffers to make sure they are moved to the correct device.
        self.register_buffer('input_tensor', input_tensor)
        self.register_buffer('accuracy_tensor', accuracy_tensor)

    def forward(self, x):
        # Ensure x is a tensor.
        if not tr.is_tensor(x):
            x = tr.tensor(x)

        # Compute distances to all points in the input tensor for each x.
        # This computes a tensor of distances for each x.
        distances = tr.abs(self.input_tensor.unsqueeze(0) - x.unsqueeze(1))

        # Get the indices of the smallest distance for each x.
        min_distance_indices = tr.argmin(distances, dim=1)

        # Return the associated accuracies.
        return self.accuracy_tensor[min_distance_indices]


class best_fit_polynomial(FunctionApproximator):
    """
    Polynomial function approximator.
    """

    def __init__(self, inputs, outputs, dimension):
        super(best_fit_polynomial, self).__init__(inputs, outputs)
        self.dimension = dimension
        self.coefficients = best_fit_polynomial_coefficients(self.inputs, self.outputs, self.dimension)

    def forward(self, x):
        vandermonde_matrix = tr.stack([x ** i for i in range(self.dimension + 1)], dim=1)
        vandermonde_matrix = vandermonde_matrix.to(self.coefficients.dtype)
        return tr.matmul(vandermonde_matrix, self.coefficients)


class KNearest_neighbors(FunctionApproximator):
    """
    K Nearest Neighbors function approximator.
    """

    def __init__(self, inputs, outputs, K, Remove=False):
        super(KNearest_neighbors, self).__init__(inputs, outputs, Remove)
        self.K = K

    def Forward(self, x):
        y_k_nearest = self.outputs[kNN(self.inputs, x, self.K)[0]]
        return tr.mean(y_k_nearest, dim=1)



class KernelRidgeApproximator(FunctionApproximator):
    def __init__(self, inputs, outputs, alpha=.01, kernel='rbf', Remove=False):
        super(KernelRidgeApproximator, self).__init__(inputs, outputs, Remove)

        self.krr = KernelRidge(kernel=kernel, alpha=alpha)

        # Fit the model
        self.krr.fit(inputs.cpu().numpy(), outputs.cpu().numpy())

    def Forward(self, x):
        y_pred = self.krr.predict(x.cpu().numpy())
        return tr.tensor(y_pred, dtype=tr.float32)

class Kernel(FunctionApproximator):
    """
    Base class for function approximators.
    """

    def __init__(self, inputs, outputs,Normalization_dimension = 0, Remove=False):
        super(Kernel, self).__init__(inputs, outputs,Remove)
        self.weights  = None
        self.Normalization_dimension = Normalization_dimension
    @abstractmethod

    def weights_comp(self, x):
        pass

    def Weights(self, x):
        self.weights_comp(x)
        if (self.Normalization_dimension!= None):
            self.weights = self.weights/ (tr.sum(self.weights, dim=self.Normalization_dimension, keepdim=True) + 10 ** -7)

    def Forward(self, x):
        #print('confidences',x)
        self.Weights(x)
        #print(self.weights.T)
        weighted_outputs = tr.matmul(self.weights.T, self.outputs.view(-1, 1).float())
        return weighted_outputs

class GaussianKernel(Kernel):
    """
    Gaussian Kernel function approximator.
    """

    def __init__(self, inputs, outputs, bandwidth = 0.1,Normalization_dimension = 0, Remove=False):
        super(GaussianKernel, self).__init__(inputs, outputs, Normalization_dimension,Remove)
        self.bandwidth = bandwidth

    @abstractmethod
    def compute_bandwidth(self, x):
        pass

    def weights_comp(self, x):
        self.compute_bandwidth(x)
        temp_x = x.double()  # Convert to float64
        temp_inputs = self.inputs.double()  # Convert to float64
        distances = tr.cdist(temp_inputs.view(-1, 1), temp_x.view(-1, 1)).float()

        # If bandwidth is not a tensor, convert it into a tensor with the same shape as distances
        if not isinstance(self.bandwidth, tr.Tensor):
            self.bandwidth = tr.full_like(distances, self.bandwidth)
        else:
            # If bandwidth is a tensor, make sure it has the same shape as distances
            assert self.bandwidth.shape == distances.shape, "Bandwidth tensor must have the same shape as distances."

        # Identify where bandwidth is zero
        zero_bandwidth = (self.bandwidth == 0)

        # Compute weights using the Gaussian formula, this will create inf where bandwidth is zero
        self.weights = tr.exp(-distances.pow(2) / (2 * self.bandwidth ** 2))

        # Correct the weights where bandwidth was zero
        # Set to 1 where distance is zero, and 0 otherwise
        self.weights[zero_bandwidth] = (distances == 0).float()[zero_bandwidth]

        return self.weights

class KNNGaussianKernel(GaussianKernel):
    """
    Adaptive Gaussian Kernel function approximator.
    """

    def __init__(self, inputs, outputs, k,Normalization_dimension = 0, Remove=False):

        super(KNNGaussianKernel, self).__init__(inputs, outputs, None ,Normalization_dimension, Remove)
        self.k = k

        #self.bandwidth = None

    def remove(self,x,result):
        if self.Remove:
            mask = (x <= self.inputs.min())
            result[mask] = 0
            #print('in the mask range', mask)
        return result

    def compute_bandwidth(self, x):
        _, _, distances = kNN(self.inputs, x, self.k)
        kth_nearest_distances = distances[:, -1]  # Get the k-th nearest distance
        #print('input, x, distances, kthnn, ',self.inputs.shape, x.shape,distances.shape,kth_nearest_distances.shape)
        self.bandwidth = kth_nearest_distances.unsqueeze(0).repeat(len(self.inputs),1)


class BivariateGaussianKernel(FunctionApproximator):
    """
    Bivariate Gaussian Kernel function approximator.
    """

    def __init__(self, inputs, outputs,K = 20,Normalization_dimension = 0,  Remove=False ):
        assert inputs.dim() == 2 and inputs.shape[1] == 2, "Inputs should be a 2D tensor with shape (n, 2)"
        super(BivariateGaussianKernel, self).__init__(inputs, outputs, Remove)
        self.constant_bandwidth = 0.
        self.k = K
        self.Normalization_dimension = Normalization_dimension
    @abstractmethod
    def compute_bandwidth(self, x):
        pass

    def remove(self,x,result):
        if self.Remove:
            #print(self.inputs.min(dim = 0)[0][0])
            mask_1 = (x[:,0] <= self.inputs.min(dim = 0)[0][0])
            mask_2 = (x[:,1] <= self.inputs.min(dim=0)[0][1])
            result[mask_1] = 0
            result[mask_2] = 0
            #print('in the mask range', mask)
        return result

    def remove_bad_values(self, x):
        #print(x.shape)
        temp_x = x.double()  # Convert to float64
        temp_inputs = self.inputs.double()  # Convert to float64

        distances = tr.cdist(temp_inputs, temp_x).float()
        #print(distances.shape)
        # Get the indices of the N nearest neighbours

        _, indices = distances.topk(self.k, dim=0, largest=False)
        #print(indices.shape)
        # Compute the mean position of the N nearest neighbours
        #print(self.inputs[indices].shape)
        mean_positions = tr.mean(self.inputs[indices], dim=0)
        #print(x.shape, mean_positions.shape)
        # Compute the distances between x and the mean positions
        diff = x - mean_positions
        #print('difference shape', diff.shape)
        # Set the semi-major axes lengths

        # Assuming self.inputs is a PyTorch tensor of shape (n, 2)
        min_values, _ = tr.min(self.inputs, dim=0)
        max_values, _ = tr.max(self.inputs, dim=0)

        #print("Difference values for each direction: ",  )

        a, b = max_values - min_values  # modify these values as required
        #print(a,b)
        # If the point lies outside the ellipse, set the bandwidth to 0.0
        #print('condition_shape', ((diff[:, 0] / a) ** 2 + (diff[:, 1] / b) ** 2 > 1).shape)
        self.bandwidth[((diff[:, 0] / a) ** 2 + (diff[:, 1] / b) ** 2 > 0.1)] = self.constant_bandwidth

    def Forward(self, x):

        self.compute_bandwidth(x)
        if self.Remove:
            self.remove_bad_values(x)

        #print('normal bandwidth', self.bandwidth.shape)
        # distances for each dimension

        temp_x = x.double()  # Convert to float64
        temp_inputs = self.inputs.double()  # Convert to float64

        distances = tr.cdist(temp_inputs, temp_x).float()

        #print('distances in double gaussian', distances.shape)
        # normalize the weights for each dimension
        # reshape bandwidth tensor to match dimensions with distances
        self.bandwidth = self.bandwidth.view(1, -1)

        weights = tr.exp(-distances.pow(2) / (2 * self.bandwidth ** 2))
        print(weights.shape)
        if (self.Normalization_dimension!= None):
            weights = weights / (tr.sum(weights, dim=self.Normalization_dimension, keepdim=True) + 10 ** -7)  # normalize the weights
        else:
            weights = weights
        #print(weights.shape)
        # compute the weighted outputs
        #print('what am I doing here',weights.T.shape, self.outputs.view(-1, 1).shape)
        weighted_outputs = tr.matmul(weights.T, self.outputs.view(-1, 1).float())
        return weighted_outputs



class GeneralizedBivariateGaussianKernel(FunctionApproximator):
    """
    Generalized Bivariate Gaussian Kernel function approximator.
    """

    def __init__(self, inputs, outputs, min=False):
        assert inputs.dim() == 2 and inputs.shape[1] == 2, "Inputs should be a 2D tensor with shape (n, 2)"
        super(GeneralizedBivariateGaussianKernel, self).__init__(inputs, outputs, min)

    @abstractmethod
    def compute_bandwidth(self, x):
        pass

    def Forward(self, x):
        # Compute bandwidths separately for each dimension
        #print(x,self.inputs)

        self.compute_bandwidth(x)
        #print(self.bandwidth.shape)
        #print('the bandwidth', tr.sort(self.bandwidth[:, 1], dim=0))

        # Compute differences for each dimension separately

        diff = (self.inputs.unsqueeze(1).double() - x.unsqueeze(0).double()).float()

        # normalize the weights for each dimension
        # compute the weights using the separate bandwidths
        weights = tr.exp(-0.5 * (diff/ self.bandwidth.unsqueeze(0).expand(self.inputs.shape[0],-1,-1)) ** 2)
        print('weights',weights.shape)
        weights = tr.prod(weights, dim=2)
        weights = weights / (tr.sum(weights, dim=0, keepdim=True)+10**-7)  # normalize the weights
        print(weights.shape)
        # compute the weighted outputs
        print()
        weighted_outputs = tr.matmul(weights.T, self.outputs.view(-1, 1))
        return weighted_outputs


import torch as tr
#
#
# class NormalizedBivariateGaussianKernel(BivariateGaussianKernel):
#     """
#     Bivariate Gaussian Kernel function approximator with input normalization.
#     """
#
#     def __init__(self, inputs, outputs, K=20, Normalization_dimension=0, Remove=False):
#         # Apply logarithmic transformation
#         log_inputs = tr.log(1 - inputs + 1e-8)
#
#         # Normalize the inputs by subtracting the mean and dividing by the standard deviation
#         self.inputs_mean = tr.mean(log_inputs, dim=0, keepdim=True)
#         self.inputs_std = tr.std(log_inputs, dim=0, keepdim=True)
#
#         normalized_inputs = (log_inputs - self.inputs_mean) / self.inputs_std
#
#         # Call the superclass constructor with the normalized inputs
#         super(NormalizedBivariateGaussianKernel, self).__init__(normalized_inputs, outputs, K, Normalization_dimension,Remove)
#
#     def Forward(self, x):
#         # Apply logarithmic transformation
#         log_x = tr.log(1 - x + 1e-8)
#
#         # Normalize the input x in the same way as the training inputs
#         normalized_x = (log_x - self.inputs_mean) / self.inputs_std
#
#         # Call the superclass Forward method with the normalized input
#         return super(NormalizedBivariateGaussianKernel, self).Forward(normalized_x)



class NormalizedBivariateGaussianKernel(BivariateGaussianKernel):
    """
    Bivariate Gaussian Kernel function approximator with input normalization.
    """

    def __init__(self, inputs, outputs,K = 20,Normalization_dimension = 0,  Remove=False):
        # Normalize the inputs by subtracting the mean and dividing by the standard deviation
        self.inputs_mean = tr.mean(inputs, dim=0, keepdim=True)
        self.inputs_std = tr.std(inputs, dim=0, keepdim=True)

        #normalized_inputs = (inputs - self.inputs_mean) / self.inputs_std
        normalized_inputs = tr.log(1-inputs+1e-8)
        # Call the superclass constructor with the normalized inputs
        super(NormalizedBivariateGaussianKernel, self).__init__(normalized_inputs, outputs, K,Normalization_dimension, Remove)

    def Forward(self, x):
        # Normalize the input x in the same way as the training inputs
        #normalized_x = (x - self.inputs_mean) / self.inputs_std
        normalized_x = tr.log(1 - x+1e-8)
        # Call the superclass Forward method with the normalized input
        return super(NormalizedBivariateGaussianKernel, self).Forward(normalized_x)

class GeneralizedNormalizedBivariateGaussianKernel(GeneralizedBivariateGaussianKernel):
    """
    Bivariate Gaussian Kernel function approximator with input normalization.
    """

    def __init__(self, inputs, outputs, min=False):
        # Normalize the inputs by subtracting the mean and dividing by the standard deviation
        self.inputs_mean = tr.mean(inputs, dim=0, keepdim=True)
        self.inputs_std = tr.std(inputs, dim=0, keepdim=True)

        normalized_inputs = (inputs - self.inputs_mean) / self.inputs_std

        # Call the superclass constructor with the normalized inputs
        super(GeneralizedNormalizedBivariateGaussianKernel, self).__init__(normalized_inputs, outputs, min)

    def Forward(self, x):
        # Normalize the input x in the same way as the training inputs
        normalized_x = (x - self.inputs_mean) / self.inputs_std

        # Call the superclass Forward method with the normalized input
        return super(GeneralizedNormalizedBivariateGaussianKernel, self).Forward(normalized_x)

# class NormalizedRadialBasisKernel(NormalizedBivariateGaussianKernel):
#     """
#     Radial Basis Function kernel with normalized inputs and constant bandwidth.
#     """
#
#     def __init__(self, inputs, outputs, bandwidth=0.5, min=False):
#         # Store the constant bandwidth
#         self.constant_bandwidth = bandwidth
#
#         # Call the superclass constructor
#         super(NormalizedRadialBasisKernel, self).__init__(inputs, outputs, min)
#
#     def compute_bandwidth(self, x):
#         # Ignore the input x and always return the constant bandwidth
#         self.bandwidth = tr.tensor([self.constant_bandwidth for _ in range(self.inputs.shape[0])])
#class NormalizedRadialBasisKernel(BivariateGaussianKernel):


class MinimumBandwidthKernel(NormalizedBivariateGaussianKernel):
    """
    Minimum Bandwidth Kernel function approximator.
    """

    def __init__(self, inputs, outputs, K=20, Normalization_dimension=0, Remove=False):
        super(MinimumBandwidthKernel, self).__init__(inputs, outputs, K, Normalization_dimension, Remove)

    def compute_bandwidth(self, x):
        temp_x = x.double()  # Convert to float64
        temp_inputs = self.inputs.double()  # Convert to float64

        distances = tr.cdist(temp_inputs, temp_x).float()

        # Get the k-th smallest distances for each point
        k_distances, _ = distances.topk(self.k, dim=0, largest=False)

        # Get the minimum of the k-th smallest distances
        #print(k_distances.shape)
        min_k_distance = tr.mean(k_distances[-1,:], dim=0).item()
        #print(min_k_distance)
        # Duplicate the minimum k-th distance for each point to use as the bandwidth
        self.bandwidth = min_k_distance*tr.ones_like(x[:,0])

        #print(self.bandwidth.shape)
class NormalizedRadialBasisKernel(NormalizedBivariateGaussianKernel):
    """
    Radial Basis Function kernel with normalized inputs and variable bandwidth.
    """

    def __init__(self, inputs, outputs,K = 20,Normalization_dimension = 0,  Remove=False):
        # Store k for k-nearest neighbors
        # Call the superclass constructor
        super(NormalizedRadialBasisKernel, self).__init__(inputs, outputs, K,Normalization_dimension, Remove)
        self.k = K
    def compute_bandwidth(self, x):
        # Compute k nearest neighbors distances for each point

        _, _, distances = kNN(self.inputs, x, self.k)
        kth_nearest_distances = distances[:, -1]  # Get the k-th nearest distance
        #print('input, x, distances, kthnn, ', self.inputs.shape, x.shape, distances.shape, kth_nearest_distances.shape)
        self.bandwidth = kth_nearest_distances

class GeneralizedNormalizedRadialBasisKernel(GeneralizedNormalizedBivariateGaussianKernel):
    """
    Radial Basis Function kernel with normalized inputs and variable bandwidth per dimension.
    """

    def __init__(self, inputs, outputs, k=[10, 10], min=False):
        # Store k for k-nearest neighbors, must be a list of two elements
        assert len(k) == 2, "k should be a list with two elements"
        self.k = k

        # Call the superclass constructor
        super(GeneralizedNormalizedRadialBasisKernel, self).__init__(inputs, outputs, min)

    def compute_bandwidth(self, x):
        # Compute k nearest neighbors distances for each point and each dimension
        self.bandwidth = tr.empty_like(x)

        for i in range(x.shape[1]):
            #print(self.inputs,x)
            _, _, distances = kNN(self.inputs[:, i].unsqueeze(-1), x[:, i].unsqueeze(-1), self.k[i])
            self.bandwidth[:, i] = distances[:, -1]  # Get the k-th nearest distance
            self.bandwidth[tr.abs(self.bandwidth) < 10 ** -7] = 10 ** -7


class Moving_average(FunctionApproximator):
    def __init__(self, inputs, outputs, window_size=5,min= False):
        super(Moving_average, self).__init__(inputs, outputs,min)
        self.window_size = window_size

    def simple_moving_average(self, y):
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[self.window_size:] - cumsum[:-self.window_size]) / float(self.window_size)

    def Forward(self, x):
        # Find the index of the input x in self.inputs
        index = (self.inputs - x).abs().argmin()

        # Calculate the simple moving average for the outputs
        smooth_outputs = self.simple_moving_average(self.outputs)

        # Adjust the index for the smoothed outputs (since it loses (window_size-1) elements)
        index = max(0, index - (self.window_size - 1))

        # Return the smoothed output value corresponding to the input x
        return smooth_outputs[index]

class LinearizedFunctionApproximator(FunctionApproximator):
    def __init__(self, inputs, outputs,min= False):
        super(LinearizedFunctionApproximator, self).__init__(inputs, outputs,min)
        self.linearized_outputs = self.linearize()

    def numerical_derivative(self, y):
        dy = np.gradient(y)
        return dy

    def find_local_extrema(self, dy):
        local_maxima = np.argwhere((dy[:-1] > 0) & (dy[1:] < 0)).flatten()
        local_minima = np.argwhere((dy[:-1] < 0) & (dy[1:] > 0)).flatten()
        return local_maxima, local_minima

    def linearize(self):
        dy = self.numerical_derivative(self.outputs.numpy())
        local_maxima, local_minima = self.find_local_extrema(dy)

        extrema_indices = np.sort(np.concatenate((local_maxima, local_minima)))
        extrema_values = self.outputs.numpy()[extrema_indices]

        linearized_outputs = np.interp(self.inputs.numpy(), self.inputs.numpy()[extrema_indices], extrema_values)

        return tr.tensor(linearized_outputs)

    def Forward(self, x):
        x = x.numpy()
        y = np.interp(x, self.inputs.numpy(), self.linearized_outputs.numpy())
        return tr.tensor(y, dtype=tr.float32)


class Bivariate_density_approximator(FunctionApproximator):
    """
    Bivariate Gaussian Kernel function approximator.
    """

    def __init__(self, inputs, outputs, mini=False, N=20):
        assert inputs.dim() == 2 and inputs.shape[1] == 2, "Inputs should be a 2D tensor with shape (n, 2)"
        super(BivariateGaussianKernel, self).__init__(inputs, outputs, mini)
        self.constant_bandwidth = 0.
        self.min_bandwidth = 0.0
        self.k = N

    @abstractmethod
    def compute_bandwidth(self, x):
        pass

    def remove_bad_values(self, x):
        #print(x.shape)
        temp_x = x.double()  # Convert to float64
        temp_inputs = self.inputs.double()  # Convert to float64

        distances = tr.cdist(temp_inputs, temp_x).float()
        #print(distances.shape)
        # Get the indices of the N nearest neighbours

        _, indices = distances.topk(self.k, dim=0, largest=False)
        #print(indices.shape)
        # Compute the mean position of the N nearest neighbours
        #print(self.inputs[indices].shape)
        mean_positions = tr.mean(self.inputs[indices], dim=0)
        #print(x.shape, mean_positions.shape)
        # Compute the distances between x and the mean positions
        diff = x - mean_positions
        #print('difference shape', diff.shape)
        # Set the semi-major axes lengths

        # Assuming self.inputs is a PyTorch tensor of shape (n, 2)
        min_values, _ = tr.min(self.inputs, dim=0)
        max_values, _ = tr.max(self.inputs, dim=0)

        print("Difference values for each direction: ", )

        a, b = max_values - min_values  # modify these values as required
        print(a, b)
        # If the point lies outside the ellipse, set the bandwidth to 0.0
        #print('condition_shape', ((diff[:, 0] / a) ** 2 + (diff[:, 1] / b) ** 2 > 1).shape)
        self.bandwidth[((diff[:, 0] / a) ** 2 + (diff[:, 1] / b) ** 2 > 0.05)] = self.constant_bandwidth

    def Forward(self, x):
        self.compute_bandwidth(x)
        if self.mini:
            self.remove_bad_values(x)

        #print('normal bandwidth', self.bandwidth.shape)
        # distances for each dimension

        temp_x = x.double()  # Convert to float64
        temp_inputs = self.inputs.double()  # Convert to float64

        distances = tr.cdist(temp_inputs, temp_x).float()

        #print('distances in double gaussian', distances.shape)
        # normalize the weights for each dimension
        # reshape bandwidth tensor to match dimensions with distances
        self.bandwidth = self.bandwidth.view(1, -1)

        weights = tr.exp(-distances.pow(2) / (2 * self.bandwidth ** 2))
        # compute the weighted outputs
        weighted_outputs = tr.matmul(weights.T, self.outputs.view(-1, 1))
        return weighted_outputs




#
# class GeneralizedNormalizedRadialBasisKernel(GeneralizedBivariateGaussianKernel):
#     """
#     Radial Basis Function kernel with normalized inputs and variable bandwidth per dimension.
#     """
#
#     def __init__(self, inputs, outputs, k=10, min=False):
#         # Store k for k-nearest neighbors
#         self.k = k
#
#         # Call the superclass constructor
#         super(GeneralizedNormalizedRadialBasisKernel, self).__init__(inputs, outputs, min)
#
#     def compute_bandwidth(self, x):
#         # Compute k nearest neighbors distances for each point and each dimension
#         self.bandwidth = tr.empty_like(x)
#         for i in range(x.shape[1]):
#             _, _, distances = kNN(self.inputs[:, i].unsqueeze(-1), x[:, i].unsqueeze(-1), self.k)
#             self.bandwidth[:, i] = distances[:, -1]  # Get the k-th nearest distance

#

# class AdaptiveGaussianKernelFromMin(AdaptiveGaussianKernel):
#     """
#     Adaptive Gaussian Kernel function approximator that returns zero for any element in x smaller than the smallest self.input value.
#     """
#
#     def __init__(self, inputs, outputs, k):
#         super(AdaptiveGaussianKernelFromMin, self).__init__(inputs, outputs, k)
#
#     def Forward(self, x):
#         # Compute the bandwidth for the adaptive Gaussian kernel
#         self.compute_bandwidth(x)
#
#         # Call the Forward method of the parent class (AdaptiveGaussianKernel)
#         result = super(AdaptiveGaussianKernelFromMin, self).Forward(x)
#
#         # Create a mask for the elements in x smaller than the smallest self.input value
#         min_input = tr.min(self.inputs)
#         mask = x < min_input
#
#         # Set the masked output elements to zero
#         result[mask] = 0
#
#         return result
#
# class First_Point_Gaussian_Kernel(GaussianKernel):
#     """
#     First Point Gaussian Kernel function approximator.
#     """
#
#     def __init__(self, inputs, outputs, bandwidth, time_function):
#         super(First_Point_Gaussian_Kernel, self).__init__(inputs, outputs, None)
#         self.time_function = time_function
#
#
#     def Forward(self, x):
#         self.inputs = self.inputs.view(-1,1)
#         self.outputs = self.outputs.view(-1,1)
#         x = x.view(-1,1)
#
#         distances = tr.cdist(self.inputs, x, p=2)
#         adjusted_bandwidth = self.time_function()
#         adjusted_bandwidth[mask] = adjusted_bandwidth[mask] * self.time_function(x[mask] / x_0)
#         weights = tr.exp(-distances.pow(2) / (2 * adjusted_bandwidth.repeat(self.inputs.shape[0], 1).view(self.inputs.shape[0], -1) ** 2))
#         weights = weights / tr.sum(weights, dim=0)
#         weighted_outputs = weights * self.outputs
#         self.inputs = tr.squeeze(self.inputs, dim=1)
#         self.outputs = tr.squeeze(self.outputs, dim=1)
#         x = tr.squeeze(x,dim = 1)
#         return tr.sum(weighted_outputs, dim=0)

def piecewise_best_fit_polynomial(confidences, accuracies, split):
    """
    Piecewise best fit polynomial function.
    """

    # Find the index of the largest confidence smaller than split
    i = (confidences < split).nonzero(as_tuple=True)[0][-1]

    # Separate data into two parts based on the index i
    confidences1 = confidences[:i + 1]
    accuracies1 = accuracies[:i + 1]

    confidences2 = confidences[i + 1:]
    accuracies2 = accuracies[i + 1:]

    # Fit best_fit_polynomial for each part
    f1 = best_fit_polynomial(confidences1, accuracies1, confidences1.size(0) - 1)
    f2 = best_fit_polynomial(confidences2, accuracies2, confidences2.size(0) - 1)

    # Get data points between x_i and x_i+1
    xi = confidences[i]
    xi1 = confidences[i + 1]
    mask_middle = (confidences >= xi) & (confidences <= xi1)
    confidences_middle = confidences[mask_middle]
    accuracies_middle = accuracies[mask_middle]
    new_points = tr.tensor([f1(confidences_middle[0].item()), f2(confidences_middle[1].item())])

    # Fit a degree 3 polynomial for the interval between x_i and x_i+1
    f_middle = best_fit_polynomial(confidences_middle, new_points, 1)

    # Define piecewise function
    def piecewise_f(x):
        if not isinstance(x, tr.Tensor):
            x = tr.tensor([x])

        mask1 = (0 <= x) & (x <= xi)
        mask2 = (xi < x) & (x < xi1)
        mask3 = (xi1 <= x) & (x <= 1)

        result = tr.zeros_like(x)
        result[mask1] = f1(x[mask1])
        result[mask2] = f_middle(x[mask2])
        result[mask3] = f2(x[mask3])

        return result

    return piecewise_f


class Piecewise_constant(FunctionApproximator):
    """
    Piecewise constant function approximator.
    """

    def __init__(self, inputs, outputs,min=False):
        super(Piecewise_constant, self).__init__(inputs, outputs,min)
        self.inputs, indices = tr.sort(self.inputs)
        self.outputs = self.outputs[indices]

    def Forward(self, x):
        x = self._check_and_convert_tensor(x)
        lower_indices, _ = Nearest_lower_and_upper_bound_in_sorted_list(self.inputs, x)
        return self.outputs[lower_indices]


class Piecewise_linear(FunctionApproximator):
    """
    Piecewise linear function approximator.
    """

    def __init__(self, inputs, outputs, min = False):
        super(Piecewise_linear, self).__init__(inputs, outputs,  min)
        self.inputs, indices = tr.sort(self.inputs)
        self.outputs = self.outputs[indices]

    def Forward(self, x):
        x = self._check_and_convert_tensor(x)
        Indices = Nearest_lower_and_upper_bound_in_sorted_list(self.inputs, x)[1]
        lower_indices = Indices[:, 0]
        upper_indices = Indices[:, 1]

        # Create masks for edge and non-edge cases
        edge_lower_mask = lower_indices == -1
        edge_upper_mask = upper_indices == len(self.inputs)
        non_edge_mask = ~edge_lower_mask & ~edge_upper_mask

        # Update lower and upper indices for non-edge cases
        lower_indices = lower_indices[non_edge_mask]
        upper_indices = upper_indices[non_edge_mask]

        lower_inputs = self.inputs[lower_indices]
        upper_inputs = self.inputs[upper_indices]
        lower_outputs = self.outputs[lower_indices]
        upper_outputs = self.outputs[upper_indices]

        # Linear interpolation for non-edge cases
        alpha = (x[non_edge_mask] - lower_inputs) / (upper_inputs - lower_inputs)
        interpolated_outputs = tr.zeros_like(x)
        alpha = alpha.double()
        #print(alpha.dtype,lower_outputs.dtype,upper_outputs.dtype)

        interpolated_outputs[non_edge_mask] = ((1 - alpha) * lower_outputs + alpha * upper_outputs).float()

        # Handle edge cases
        # Lower edge case - interpolate between (0, 0) and first input-output pair
        lower_alpha = x[edge_lower_mask] / self.inputs[0]
        interpolated_outputs[edge_lower_mask] = lower_alpha * self.outputs[0]

        # Upper edge case - interpolate between last input-output pair and (1, 1)
        upper_alpha = (x[edge_upper_mask] - self.inputs[-1]) / (1 - self.inputs[-1])
        interpolated_outputs[edge_upper_mask] = (1 - upper_alpha) * self.outputs[-1] + upper_alpha

        return interpolated_outputs


#### NETWORK LOSSES/MEASURES ####

### output measures ###

### prediction measures ###

class TrueFalseMeasure(nn.Module):
    """
    TrueFalseMeasure class: a simple +1/+0 loss used to compute accuracy.
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

#### CONFIDENCE LOSSES/MEASURES ####

### output measures ###
import torch as tr
import torch.nn as nn
import torch as tr
import torch.nn as nn

class CrossEntropy(nn.Module):
    """
    CrossEntropy class: calculates cross entropy loss assuming the inputs are already probabilities.
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, predicted, target):

        """
        :param predicted: Predicted probabilities
        :param target: Actual class labels
        :return: Cross entropy loss
        """

        # number of samples
        N = predicted.shape[0]

        # Convert the targets to one-hot encoding
        target_one_hot = tr.zeros_like(predicted)
        target_one_hot[tr.arange(N), target.long()] = 1

        # Compute cross entropy loss
        cross_entropy_loss = -tr.sum(target_one_hot * tr.log(predicted + 1e-9)) / N
        return cross_entropy_loss

# 
# class CrossEntropy(nn.Module):
#     """
#     CrossEntropy class: calculates cross entropy loss assuming the inputs are already probabilities.
#     """
# 
#     def __init__(self, New_prediction=False):
#         super(CrossEntropy, self).__init__()
#         self.New_prediction = New_prediction
# 
#     def forward(self, predicted1, predicted2, target):
#         """
#         :param predicted1, predicted2: Predicted probabilities
#         :param target: Actual class labels
#         :return: Cross entropy loss
#         """
#         # Choose the prediction probabilities based on the New_prediction flag
#         if self.New_prediction:
#             predicted = predicted2
#         else:
#             predicted = predicted1
# 
#         # number of samples
#         N = predicted.shape[0]
# 
#         # Convert the targets to one-hot encoding
#         target_one_hot = tr.zeros_like(predicted)
#         target_one_hot[tr.arange(N), target] = 1
# 
#         # Compute cross entropy loss
#         cross_entropy_loss = -tr.sum(target_one_hot * tr.log(predicted + 1e-9)) / N
#         return cross_entropy_loss


## ECE ##
class ECE(nn.Module):
    """
    ECE class: calculates Edge defined Expected Calibration Error (ECE).
    """
    def __init__(self, bin_edges):
        super(ECE, self).__init__()
        self.bin_edges = bin_edges
        self.num_bins = len(bin_edges) - 1

    def forward(self, confidence, accuracy):
        sorted_indices = tr.argsort(confidence)
        confidence = confidence[sorted_indices]
        accuracy = accuracy[sorted_indices]

        bin_sizes = tr.Tensor([
            ((confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])).sum().item()
            for i in range(self.num_bins)
        ])

        bin_confidences = tr.Tensor([
            confidence[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].mean().item()
            for i in range(self.num_bins)
        ])

        bin_accuracies = tr.Tensor([
            accuracy[(confidence > self.bin_edges[i]) & (confidence <= self.bin_edges[i+1])].mean().item()
            for i in range(self.num_bins)
        ])

        ece_loss = ((bin_sizes.float() / len(confidence)) * tr.abs(bin_accuracies - bin_confidences)).sum()

        return ece_loss


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
        probabilities =  outputs[0]
        indices = outputs[1]
        # Calculate confidence
        indices_0 = tr.arange(probabilities.size(0))
        confidence = probabilities[indices_0.long(),(indices).long()]

        # Predicted labels based on probabilities
        predicted_labels = tr.argmax(probabilities, dim=-1)

        # Calculate accuracy
        accuracy = (predicted_labels == labels).float()

        # Sort inputs by confidence
        confidence, sorted_indices = tr.sort(confidence)
        bin_edges = Binning_method(confidence, self.num_bins, self.num_per_bin)
        accuracy = accuracy[sorted_indices]

        N_bins = len(bin_edges) - 1

        # Compute bin sizes, confidences, and accuracies
        bin_confidences = tr.zeros(N_bins)
        bin_accuracies = tr.zeros(N_bins)
        bin_sizes = tr.zeros(N_bins)

        for i in range(N_bins):
            if i == 0:
                mask = ((confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1]))
            else:
                mask = ((confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1]))

            if mask.sum() > 0:
                bin_confidences[i] = confidence[mask].mean()

                bin_accuracies[i] = accuracy[mask].mean()
                bin_sizes[i] = mask.sum()

        # Calculate ECE loss
        bin_sizes = bin_sizes.float()
        ece_loss = (bin_sizes / len(confidence)).mul(tr.abs(bin_accuracies - bin_confidences)).sum()

        return ece_loss


# ECE revised without edge

class ECE_no_edge(nn.Module):
    """
    ECE_no_edge class: calculates Expected Calibration Error (ECE) without considering the edge case.
    """
    def __init__(self, num_bins=None, num_per_bin=50):
        super(ECE_no_edge, self).__init__()
        self.num_bins = num_bins
        self.num_per_bin = num_per_bin

    def forward(self, confidence, accuracy):
        accuracy = accuracy.float()
        # Sort inputs by confidence
        confidence, sorted_indices = tr.sort(confidence)
        # print(confidence.shape)
        # print(self.num_bins)
        bin_edges = Binning_method(confidence, self.num_bins, self.num_per_bin)
        accuracy = accuracy[sorted_indices]
        #print(bin_edges)
        N_bins = len(bin_edges) - 1
        #print(N_bins)
        # Compute bin sizes, confidences, and accuracies
        bin_confidences = tr.zeros(N_bins)
        bin_accuracies = tr.zeros(N_bins)
        bin_sizes = tr.zeros(N_bins)

        for i in range(N_bins):
            if i == 0:
                mask = ((confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1]))
            else:
                mask = ((confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1]))

            if mask.sum() > 0:
                bin_confidences[i] = confidence[mask].mean()

                bin_accuracies[i] = accuracy[mask].mean()
                bin_sizes[i] = mask.sum()
        #print('confidences and accuracies of bins',bin_confidences,bin_accuracies)
        # Calculate ECE loss
        bin_sizes = bin_sizes.float()
        ece_loss = (bin_sizes / len(confidence)).mul(tr.abs(bin_accuracies - bin_confidences)).sum()

        return ece_loss
## MUTUAL INFORMATION ##

class MIE (nn.Module):
    """
    MIE class: Edge defined Mutual Information estimator.
    """
    def __init__(self, conf_bin_edges):
        super().__init__()
        self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
        self.conf_bin_edges = conf_bin_edges

    def forward(self, confidence, accuracy):
        # Compute histogram of accuracy
        h_acc, _ = tr.histogram(accuracy, bins=self.acc_bin_edges)

        # Compute joint histogram
        h_joint, _, _ = tr.histogram2d(confidence, accuracy, bins=[self.conf_bin_edges, self.acc_bin_edges])

        # Compute probabilities
        p_acc = h_acc / tr.sum(h_acc)
        p_joint = h_joint / tr.sum(h_joint)

        # Compute conditional probabilities
        p_acc_given_conf = p_joint / (tr.sum(p_joint, axis=1, keepdim=True)+1e-8)

        # Compute entropies
        print(p_acc)
        h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
        h_acc_given_conf_est = -tr.sum(p_joint * tr.log(p_acc_given_conf + 1e-8))

        # Compute mutual information
        mi_est = h_acc_est - h_acc_given_conf_est

        return mi_est
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
#from shapely.geometry import Polygon

#
# def voronoi_finite_polygons_2d(vor, radius=None):
#     """
#     Reconstruct infinite Voronoi regions in a 2D diagram to finite
#     regions.
#
#     Parameters
#     ----------
#     vor : Voronoi
#         Input diagram
#     radius : float, optional
#         Distance to 'points at infinity'.
#
#     Returns
#     -------
#     regions : list of tuples
#         Indices of vertices in each revised Voronoi regions.
#     vertices : list of tuples
#         Coordinates for revised Voronoi vertices. Same as coordinates
#         of input vertices, with 'points at infinity' appended to the
#         end.
#
#     """
#
#     if vor.points.shape[1] != 2:
#         raise ValueError("Requires 2D input")
#
#     new_regions = []
#     new_vertices = vor.vertices.tolist()
#
#     center = vor.points.mean(axis=0)
#     if radius is None:
#         radius = vor.points.ptp().max()
#
#     # Construct a map containing all ridges for a given point
#     all_ridges = {}
#     for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
#         all_ridges.setdefault(p1, []).append((p2, v1, v2))
#         all_ridges.setdefault(p2, []).append((p1, v1, v2))
#
#     # Reconstruct infinite regions
#     for p1, region in enumerate(vor.point_region):
#         vertices = vor.regions[region]
#
#         if all(v >= 0 for v in vertices):
#             # finite region
#             new_regions.append(vertices)
#             continue
#
#         # reconstruct a non-finite region
#         ridges = all_ridges[p1]
#         new_region = [v for v in vertices if v >= 0]
#
#         for p2, v1, v2 in ridges:
#             if v2 < 0:
#                 v1, v2 = v2, v1
#             if v1 >= 0:
#                 # finite ridge: already in the region
#                 continue
#
#             # Compute the missing endpoint of an infinite ridge
#             t = vor.points[p2] - vor.points[p1]  # tangent
#             t /= np.linalg.norm(t)
#             n = np.array([-t[1], t[0]])  # normal
#
#             midpoint = vor.points[[p1, p2]].mean(axis=0)
#             direction = np.sign(np.dot(midpoint - center, n)) * n
#             far_point = vor.vertices[v2] + direction * radius
#
#             new_region.append(len(new_vertices))
#             new_vertices.append(far_point.tolist())
#
#         # sort region counterclockwise
#         vs = np.asarray([new_vertices[v] for v in new_region])
#         c = vs.mean(axis=0)
#         angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
#         new_region = np.array(new_region)[np.argsort(angles)]
#
#         # finish
#         new_regions.append(new_region.tolist())
#
#     return new_regions, np.asarray(new_vertices)
#
#
# class MIE3_Voronoi(nn.Module):
#     def __init__(self, num_bins, N_vor):
#         super().__init__()
#         self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
#         self.num_bins = num_bins
#         self.N_vor = N_vor
#
#     def forward(self, confidence1, confidence2, accuracy):
#         confidences = tr.stack([confidence1, confidence2], dim=1).numpy()
#         mean = np.mean(confidences, axis=0, keepdims=True)
#         std = np.std(confidences, axis=0, keepdims=True)
#         confidences_standardized = (confidences - mean) / std
#
#         trees = []
#         bin_accuracies_list = []
#         h_joint_list = []
#
#         for _ in range(self.N_vor):
#             kmeans = KMeans(n_clusters=self.num_bins, n_init=10).fit(confidences_standardized)
#             vor = Voronoi(kmeans.cluster_centers_)
#             tree = cKDTree(kmeans.cluster_centers_)
#             trees.append(tree)
#
#             _, bins = tree.query(confidences_standardized)
#             bin_accuracies = np.array([accuracy[bins == i].mean().item() for i in range(self.num_bins)])
#             bin_accuracies_list.append(bin_accuracies)
#
#             accuracy = accuracy.type(self.acc_bin_edges.dtype)
#             h_acc, _ = np.histogram(accuracy.numpy(), bins=self.acc_bin_edges.numpy())  # added .numpy()
#             h_joint, xedges, yedges = np.histogram2d(bins, accuracy.numpy(), bins=[np.arange(self.num_bins + 1) - 0.5,
#                                                                                    self.acc_bin_edges.numpy()])
#             h_joint_list.append(h_joint)
#
#         mi_est = 0
#         for i in range(self.N_vor):
#             h_acc_tensor = tr.from_numpy(h_acc)  # changed variable name
#             h_joint = tr.from_numpy(h_joint_list[i])
#
#             p_acc = h_acc_tensor / (tr.sum(h_acc_tensor) + 1e-8)  # used h_acc_tensor
#             p_joint = h_joint / (tr.sum(h_joint) + 1e-8)
#
#             p_acc_given_bins = p_joint / (tr.sum(p_joint, dim=1, keepdim=True) + 1e-8)
#
#             h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
#             h_acc_given_bins_est = -tr.sum(p_joint * tr.log(p_acc_given_bins + 1e-8))
#
#             mi_est += h_acc_est - h_acc_given_bins_est
#
#         return mi_est / self.N_vor
#
#     #
# #
# # class MIE3_Voronoi(nn.Module):
# #     """
# #     MIE3_Voronoi class: Mutual Information estimator using Voronoi binning for three variables.
# #     """
# #     def __init__(self, num_bins):
# #         super().__init__()
# #         self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
# #         self.num_bins = num_bins
# #
# #     def forward(self, confidence1, confidence2, accuracy):
# #         # Combine confidences and standardize
# #         confidences = tr.stack([confidence1, confidence2], dim=1).numpy()
# #         mean = np.mean(confidences, axis=0, keepdims=True)
# #         std = np.std(confidences, axis=0, keepdims=True)
# #         confidences_standardized = (confidences - mean) / std
# #
# #         # KMeans for binning
# #         kmeans = KMeans(n_clusters=self.num_bins, n_init=10).fit(confidences_standardized)
# #
# #         # Create Voronoi tessellation
# #         vor = Voronoi(kmeans.cluster_centers_)
# #
# #         # Plot the Voronoi tesselation
# #         #self.plot_voronoi_tesselation(vor, confidences_standardized, accuracy)
# #         # Create a KDTree with the Voronoi centroids
# #         tree = cKDTree(kmeans.cluster_centers_)
# #
# #         # Assign each point to a bin according to Voronoi tessellation
# #         _, bins = tree.query(confidences_standardized)
# #
# #         # Compute the average accuracy for each bin
# #         bin_accuracies = np.array([accuracy[bins == i].mean().item() for i in range(self.num_bins)])
# #
# #         # Compute histogram of accuracy
# #         accuracy = accuracy.type(self.acc_bin_edges.dtype)
# #         h_acc, _ = np.histogram(accuracy, bins=self.acc_bin_edges)
# #
# #         # Compute joint histogram
# #
# #         #print(np.arange(self.num_bins + 1))
# #         h_joint, xedges, yedges = np.histogram2d(bins, accuracy.numpy(), bins=[np.arange(self.num_bins+1)-0.5, self.acc_bin_edges.numpy()])
# #
# #
# #         # Convert numpy arrays to tensors
# #         h_acc = tr.from_numpy(h_acc)
# #         h_joint = tr.from_numpy(h_joint)
# #
# #         # Compute probabilities
# #         p_acc = h_acc / (tr.sum(h_acc) + 1e-8)
# #         p_joint = h_joint / (tr.sum(h_joint)+ 1e-8)
# #
# #         # Compute conditional probabilities
# #         p_acc_given_bins = p_joint / (tr.sum(p_joint, dim=1, keepdim=True)+ 1e-8)
# #
# #         # Compute entropies
# #         h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
# #         print('The accuracy Entropy', h_acc_est)
# #         h_acc_given_bins_est = -tr.sum(p_joint * tr.log(p_acc_given_bins + 1e-8))
# #
# #         # Compute mutual information
# #         mi_est = h_acc_est - h_acc_given_bins_est
# #
# #         return mi_est
# #
# #         # Compute joint histogram
#
#
#     def plot_voronoi_tesselation(self, vor, points, accuracies):
#
#         #plt.figure(figsize=(10, 10))
#         voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_colors='orange')
#
#         # Create a KDTree with the Voronoi centroids
#         tree = cKDTree(vor.points)
#
#         # Assign each point to a bin according to Voronoi tessellation
#         _, bins = tree.query(points)
#
#         # Compute the average accuracy for each bin
#         bin_accuracies = np.array([accuracies[bins == i].mean().item() for i in range(len(vor.points))])
#
#         new_regions, new_vertices = voronoi_finite_polygons_2d(vor)
#
#
#         # Get Voronoi polygons
#         polygons = [Polygon(new_vertices[region]) for region in new_regions]
#
#         # Continue as before...
#
#         # Get Voronoi polygons
#         #polygons = [vor.vertices[region] for region in vor.regions if region and -1 not in region]
#
#         # Create a colormap
#         cmap = plt.cm.viridis
#
#         # Plot filled polygons
#         for polygon, accuracy in zip(polygons, bin_accuracies):
#             plt.fill(*zip(*polygon.exterior.coords), color=cmap(accuracy))
#
#         # Scatter colors based on Voronoi cell
#         #scatter_colors = [cmap(b) for b in bins]
#
#         # Generate color list
#         colors = ['blue' if acc else 'red' for acc in accuracies]
#
#         plt.scatter(points[:, 0], points[:, 1], c=colors, edgecolor='black', alpha=0.9)
#
#         # Add the centroids
#         # Add the centroids
#         centroids = np.array([polygon.centroid.coords[0] for polygon in polygons])
#         plt.scatter(centroids[:, 0], centroids[:, 1], c='white', marker='x')
#
#         # Expanding the x and y limits of the plot
#         plt.xlim(points[:, 0].min() , points[:, 0].max())
#         plt.ylim(points[:, 1].min() , points[:, 1].max())
#
#         # Show the colorbar
#         plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
#         plt.title("Voronoi tessellation with cells colored by bin accuracy")
#         plt.show()
#         plt.close()
#

#
# class MIE3_no_edge(nn.Module):
#     """
#     MIE3_no_edge class: Mutual Information estimator without considering the edge case for three variables.
#     """
#     def __init__(self, num_bins, num_per_bin):
#         super().__init__()
#         self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
#         self.num_bins = num_bins
#         self.num_per_bin = num_per_bin
#
#     def forward(self, confidence1, confidence2, accuracy):
#         # Compute bin edges for confidence values and third variable
#         conf1_bin_edges = Binning_method(confidence1, self.num_bins, self.num_per_bin)
#         conf2_bin_edges = Binning_method(confidence2, self.num_bins, self.num_per_bin)
#
#         # Compute histogram of accuracy
#         accuracy = accuracy.type(self.acc_bin_edges.dtype)
#         h_acc, _ = np.histogram(accuracy, bins=self.acc_bin_edges)
#
#         # Compute joint histogram
#         h_joint, _ = np.histogramdd((confidence1.numpy(), confidence2.numpy(), accuracy.numpy()),
#                                         bins=[conf1_bin_edges.numpy(), conf2_bin_edges.numpy(), self.acc_bin_edges.numpy()])
#
#         # Convert numpy arrays to tensors
#         h_acc = tr.from_numpy(h_acc)
#         h_joint = tr.from_numpy(h_joint)
#
#         # Compute probabilities
#         p_acc = h_acc / tr.sum(h_acc)
#         p_joint = h_joint / tr.sum(h_joint)
#
#         # Compute conditional probabilities
#         p_acc_given_conf1_and_conf2 = p_joint / (tr.sum(p_joint, dim=(1,2), keepdim=True)+ 1e-8)
#
#         # Compute entropies
#         h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
#         h_acc_given_conf1_and_conf2_est = -tr.sum(p_joint * tr.log(p_acc_given_conf1_and_conf2 + 1e-8))
#
#         # Compute mutual information
#         mi_est = h_acc_est - h_acc_given_conf1_and_conf2_est
#
#         return mi_est

import numpy as np
import matplotlib.pyplot as plt


class MIE3_kernel(nn.Module):
    """
    MIE3_no_edge class: Mutual Information estimator without considering the edge case for three variables.
    """

    def __init__(self, kernel_acc, test_set):
        super().__init__()
        self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
        self.test_set = test_set
        self.kernel_acc = kernel_acc

    def forward(self, accuracy):
        # Compute bin edges for confidence values and third variable

        # Compute histogram of accuracy
        accuracy = accuracy.type(self.acc_bin_edges.dtype)
        h_acc, _ = np.histogram(accuracy, bins=self.acc_bin_edges)

        # Convert numpy arrays to tensors
        h_acc = tr.from_numpy(h_acc)

        p_acc = h_acc / tr.sum(h_acc)
        #print(p_acc)
        h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
        #print(h_acc_est)
        #print('accuracy entropy', h_acc_est)

        # Define a function to compute the conditional entropy
        def entropy_func(x):
            #print('x shape', x.shape)
            Accuracy = self.kernel_acc(x)
            return -1 * ((Accuracy * tr.log(Accuracy + 1e-5) + (1 - Accuracy) * tr.log((1 - Accuracy + 1e-5))))

        # Estimate the conditional entropy using Monte Carlo integral
        h_acc_given_conf_and_third_est = self.monte_carlo_2D_integral(entropy_func)
        #print('conditional entropy', h_acc_given_conf_and_third_est)
        # Compute mutual information
        mi_est = h_acc_est - h_acc_given_conf_and_third_est
        return mi_est

    def monte_carlo_2D_integral(self, f):
        # Generate random points in the square [0,1] x [0,1]

        points = self.test_set
        #print('the points are',points, ' and there are '+str(len(points)))
        # Apply the function to each point and sum the results
        integral_approx = tr.sum(f(points))

        # Normalize by the number of points
        integral_approx /= len(self.test_set)

        return integral_approx

class MIE3_no_edge(nn.Module):
    """
    MIE3_no_edge class: Mutual Information estimator without considering the edge case for three variables.
    """

    def __init__(self, num_bins, num_per_bin):
        super().__init__()
        self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
        self.num_bins = num_bins
        self.num_per_bin = num_per_bin

    def forward(self, confidence, accuracy, third_variable):
        # Compute bin edges for confidence values and third variable
        conf_bin_edges = Binning_method(confidence, self.num_bins, self.num_per_bin)
        third_var_bin_edges = Binning_method(third_variable, self.num_bins, self.num_per_bin)

        # Compute histogram of accuracy
        accuracy = accuracy.type(self.acc_bin_edges.dtype)
        h_acc, _ = np.histogram(accuracy, bins=self.acc_bin_edges)

        # Compute joint histogram
        h_joint, _ = np.histogramdd((confidence.numpy(), accuracy.numpy(), third_variable.numpy()),
                                    bins=[conf_bin_edges.numpy(), self.acc_bin_edges.numpy(),
                                          third_var_bin_edges.numpy()])

        # Convert numpy arrays to tensors
        h_acc = tr.from_numpy(h_acc)
        h_joint = tr.from_numpy(h_joint)

        # Compute probabilities
        p_acc = h_acc / (tr.sum(h_acc)+ 1e-8)
        p_joint = tr.sum(h_joint, dim=1) / (tr.sum(h_joint)+ 1e-8)

        # Compute conditional probabilities
        conditional_acc = (h_joint[:, 1, :]) / (tr.sum(h_joint, dim=1) + 1e-8)

        # Compute entropies
        h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
        h_acc_given_conf_and_third_est = -tr.sum(p_joint * (
                    conditional_acc * tr.log(conditional_acc + 1e-8) + (1 - conditional_acc) * tr.log(
                (1 - conditional_acc) + 1e-8)))

        # Compute mutual information
        mi_est = h_acc_est - h_acc_given_conf_and_third_est
        #
        # # Scatter plot of confidence vectors with accuracy as color
        # plt.scatter(confidence, third_variable, c=accuracy, cmap='viridis', label='Accuracy')
        # plt.colorbar()
        # plt.title('Scatter plot of Confidence Vectors')
        # plt.xlabel('Confidence')
        # plt.ylabel('Third Variable')
        # plt.show()
        return mi_est


def get_2Dhjoint_and_edges(confidence, accuracy, third_variable,num_bins,num_per_bin):
    acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
    # Compute bin edges for confidence values and third variable
    conf_bin_edges = Binning_method(confidence, num_bins, num_per_bin)
    third_var_bin_edges = Binning_method(third_variable, num_bins, num_per_bin)

    # Compute histogram of accuracy
    accuracy = accuracy.type(acc_bin_edges.dtype)
    h_acc, _ = np.histogram(accuracy, bins=acc_bin_edges)

    # Compute joint histogram
    h_joint, _ = np.histogramdd((confidence.numpy(), accuracy.numpy(), third_variable.numpy()),
                                bins=[conf_bin_edges.numpy(), acc_bin_edges.numpy(),
                                      third_var_bin_edges.numpy()])
    return h_joint, conf_bin_edges, third_var_bin_edges

def get_1Dhjoint_and_edges(confidence, accuracy,num_bins,num_per_bin):
    acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
    # Compute bin edges for confidence values and third variable
    conf_bin_edges = Binning_method(confidence, num_bins, num_per_bin)

    # Compute histogram of accuracy
    accuracy = accuracy.type(acc_bin_edges.dtype)
    h_acc, _ = np.histogram(accuracy, bins=acc_bin_edges)

    # Compute joint histogram
    h_joint, _, _ = np.histogram2d(confidence.numpy(), accuracy.numpy(), bins=[conf_bin_edges.numpy(), acc_bin_edges.numpy()])
    return h_joint, conf_bin_edges


def get_bins(values, bin_edges):
    # Expand dims for broadcasting
    expanded_values = values.unsqueeze(-1)
    expanded_bin_edges = bin_edges.unsqueeze(0)

    # Create boolean mask
    mask = expanded_values >= expanded_bin_edges

    # Find the last False in each row
    bins = tr.sum(mask, dim=-1) - 1

    # If the bin value is above the number of bins, set it to 9
    bins = tr.where(bins >= bin_edges.shape[0] - 1, bin_edges.shape[0] - 2, bins)

    return bins


def get_1Daccuracy(confidence_value, conf_bin_edges, h_joint):
    # Get bin index for confidence_value

    conf_bin_index = get_bins(confidence_value,conf_bin_edges)

    # Retrieve the histogram counts for the corresponding square and compute accuracy
    bin_counts= []
    for N_n, n in enumerate(conf_bin_index):
        bin_counts +=  [tr.tensor(h_joint[n, :])]

    bin_counts = tr.stack(bin_counts)

    total_count = tr.sum(bin_counts,dim = 1)

    accuracy_bin_count = bin_counts[:,1]
    accuracy = accuracy_bin_count / (total_count+1e-8)

    return accuracy


def get_2Daccuracy(confidence_value, third_variable_value, conf_bin_edges, third_var_bin_edges, h_joint):
    # Get bin index for confidence_value

    conf_bin_index = get_bins(confidence_value,conf_bin_edges)

    # Get bin index for third_variable_value
    third_var_bin_index = get_bins(third_variable_value, third_var_bin_edges)
    # Retrieve the histogram counts for the corresponding square and compute accuracy
    bin_counts= []
    for N_n, n in enumerate(conf_bin_index):
        bin_counts +=  [tr.tensor(h_joint[n, :, third_var_bin_index[N_n]])]

    bin_counts = tr.stack(bin_counts)

    total_count = tr.sum(bin_counts,dim = 1)

    accuracy_bin_count = bin_counts[:,1]
    accuracy = accuracy_bin_count / (total_count+1e-8)

    return accuracy

#
# class MIE3_no_edge(nn.Module):
#     """
#     MIE3_no_edge class: Mutual Information estimator without considering the edge case for three variables.
#     """
#     def __init__(self, num_bins, num_per_bin):
#         super().__init__()
#         self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
#         self.num_bins = num_bins
#         self.num_per_bin = num_per_bin
#
#     def forward(self, confidence, accuracy, third_variable):
#         # Compute bin edges for confidence values and third variable
#         conf_bin_edges = Binning_method(confidence, self.num_bins, self.num_per_bin)
#         third_var_bin_edges = Binning_method(third_variable, self.num_bins, self.num_per_bin)
#
#         # Compute histogram of accuracy
#         accuracy = accuracy.type(self.acc_bin_edges.dtype)
#         h_acc, _ = np.histogram(accuracy, bins=self.acc_bin_edges)
#
#         # Compute joint histogram
#         h_joint, _ = np.histogramdd((confidence.numpy(), accuracy.numpy(), third_variable.numpy()),
#                                         bins=[conf_bin_edges.numpy(), self.acc_bin_edges.numpy(), third_var_bin_edges.numpy()])
#
#         # plt.imshow(np.sum(h_joint,axis=1), cmap='hot', interpolation='nearest')
#         # plt.colorbar(label='Value')
#         # plt.show()
#         # Convert numpy arrays to tensors
#         h_acc = tr.from_numpy(h_acc)
#         h_joint = tr.from_numpy(h_joint)
#
#         # Compute probabilities
#
#         p_acc = h_acc / tr.sum(h_acc)
#         p_joint = h_joint / tr.sum(h_joint)
#         print('p_joint_shape',p_joint.shape)
#         # Compute conditional probabilities
#
#         p_acc_given_conf_and_third = p_joint / (tr.sum(p_joint, dim=(1), keepdim=True)+ 1e-8)
#         # Compute entropies
#         h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
#         h_acc_given_conf_and_third_est = -tr.sum(p_joint * tr.log(p_acc_given_conf_and_third + 1e-8))
#
#         # Compute mutual information
#         mi_est = h_acc_est - h_acc_given_conf_and_third_est
#
#         return mi_est

class MIE_no_edge(nn.Module):
    """
    MIE_no_edge class: Mutual Information estimator without considering the edge case.
    """
    def __init__(self, num_bins, num_per_bin):
        super().__init__()
        self.acc_bin_edges = tr.tensor([0.0, 0.5, 1.0])
        self.num_bins = num_bins
        self.num_per_bin = num_per_bin

    def forward(self, confidence, accuracy):
        # Compute bin edges for confidence values
        conf_bin_edges = Binning_method(confidence, self.num_bins, self.num_per_bin)
        #print(conf_bin_edges)
        # Compute histogram of accuracy
        accuracy = accuracy.type(self.acc_bin_edges.dtype)
        h_acc, _ = np.histogram(accuracy, bins=self.acc_bin_edges)

        # Compute joint histogram
        h_joint, _, _ = np.histogram2d(confidence.cpu().detach().numpy(), accuracy.cpu().detach().numpy(), bins=[conf_bin_edges.cpu().detach().numpy(), self.acc_bin_edges.cpu().detach().numpy()])
        #print(h_joint.shape)
        # Convert numpy arrays to tensors
        h_acc = tr.from_numpy(h_acc)
        h_joint = tr.from_numpy(h_joint)

        # Compute probabilities
        p_acc = h_acc / tr.sum(h_acc)
        print(p_acc)
        p_joint = h_joint / tr.sum(h_joint)
        # Compute conditional probabilities

        p_acc_given_conf = p_joint / (tr.sum(p_joint, axis=1, keepdim=True)+ 1e-8)

        # Compute entropies
        h_acc_est = -tr.sum(p_acc * tr.log(p_acc + 1e-8))
        h_acc_given_conf_est = -tr.sum(p_joint * tr.log(p_acc_given_conf + 1e-8))

        # Compute mutual information
        mi_est = (h_acc_est - h_acc_given_conf_est)
        print(h_acc_est)
        return mi_est

### prediction measures ###

class MSELossModule(nn.Module):
    """
    MSELossModule class: a wrapper for the Mean Squared Error loss function.
    """
    def __init__(self):
        super(MSELossModule, self).__init__()
        self.loss_function = nn.MSELoss()

    def forward(self, input, target):
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
            # Compute the KL-divergence between the two probabilities
            p1 = tr.tensor([input[i], 1 - input[i]])
            p2 = tr.tensor([target[i], 1 - target[i]])
            kl_div = self.kl_div(p1, p2)

            # Store the KL-divergence for the current pair of probabilities
            kl_divs[i] = kl_div

        # Sum the KL-divergences and return the result
        total_kl_div = kl_divs.sum()
        return total_kl_div

    def kl_div(self, p1, p2):
        """
        Computes the Kullback-Leibler divergence between two probability distributions p1 and p2.
        """
        kl_div = tr.sum(p1 * tr.log(p1 / p2))
        return kl_div



def Nearest_lower_and_upper_bound_in_sorted_list(X, x):
    """
    Find the largest element smaller than each element in x in X and
    the smallest element larger than x in X.
    """
    # Ensure both input tensors are 1-dimensional
    assert x.dim() == 1, "x must be a 1D tensor"
    assert X.dim() == 1, "X must be a 1D tensor"

    # Ensure X is sorted
    assert tr.all(X[:-1] <= X[1:]), "X must be a sorted tensor"

    # Store the original data type of X
    original_dtype = X.dtype

    x = x.to(tr.float32)
    X = X.to(tr.float32)

    indices = tr.searchsorted(X, x)
    X = tr.cat((X, tr.tensor([X[-1]], dtype=tr.float32)))
    upper_bound_values = X[indices]
    X = tr.cat((tr.tensor([X[0]],  dtype=tr.float32), X))
    lower_bound_values = X[indices]
    bound_indices = tr.stack([indices-1, indices], dim=-1)
    bound_values = tr.stack([lower_bound_values, upper_bound_values], dim=-1)
    bound_values = bound_values.to(original_dtype)
    return bound_values, bound_indices

def Nearest_lower_and_upper_bound(values, x):
    values, indices = tr.sort(values)
    return Nearest_lower_and_upper_bound_in_sorted_list(values, x)





def x_transform(x, scale=0.0001):
    # Compute the transformation on torch tensors
    if not isinstance(x, tr.Tensor):
        x = tr.tensor(x)
    return tr.log(tr.cos(tr.pi * x / (2 * (1 + scale)))) / np.log(np.cos(np.pi * 1 / (2 * (1 + scale))))

def x_inverse_transform(x, scale=0.0001):
    if not isinstance(x, tr.Tensor):
        x = tr.tensor(x)

    return  tr.arccos(tr.exp(np.log(np.cos(np.pi * 1 / (2 * (1 + scale))))*x))*2*(1+scale)/tr.pi

def inverse_transform_formatter(x, _, scale=0.0001):
    if not isinstance(x, tr.Tensor):
        x = tr.tensor(x)
    return  f"{x_inverse_transform(x,scale):.1f}"

def inverse_log_formatter(value, _):
    #print('value',value)
    return  f"{np.arccos(np.exp(np.log(np.cos(np.pi*1/(2*(1+0.0001))))*value))*2*(1+0.0001)/np.pi:.1f}"

#
def find_scale_and_smooth(confidences,N_p = 50,log_step_size = 0.1):
    value = 0
    breaking_condition = False
    scale = 10 ** value
    smooth = 1
    while True:
            # Perform operations or actions inside the loop for each value

            transformed_confidences = x_transform(confidences, scale)
            hist = tr.histc(transformed_confidences, bins=20)

            smooth = 20*tr.min(hist)/N_p
            max_location = tr.argmax(hist)
            #print(max_location)

            if (max_location != 19):
                breaking_condition = True
            # Example action: Print the current value
            #print(value)
            # Check breaking condition
            if breaking_condition:
                    break
            # Update the value for the next iteration
            value -= log_step_size
            scale = 10 ** value
            #print(value)
    return scale, smooth


def convexify(x, f, Figure_path, max_iterations=1000):
    X = x.clone()
    fx = f(X)

    # Calculate the integral of the original function
    original_integral = simple_integral(f, X)

    # Plot the original function
    plt.plot(X.numpy(), fx.numpy(), label="Original function" + str(original_integral))

    edge_values = (X[0], X[-1], fx[0], fx[-1])
    print(X, fx)
    iteration = 0

    best_approx = None
    best_integral = float('inf')
    prev_local_minima_indices = None

    while iteration < max_iterations:
        convex_approx = create_convex_approximation(X, fx)
        fx = convex_approx(X)
        mask = ~tr.isnan(fx)
        X = X[mask]
        fx = fx[mask]
        local_minima_indices = find_local_minima_indices(X.numpy(), fx.numpy())

        # Calculate the integral of the current approximation
        integral = simple_integral(convex_approx, X)

        print('the minimum indices are', local_minima_indices)
        plt.plot(X.numpy(), fx.numpy(), label=f"Iteration {iteration}" + " int = " + str(integral))

        # Check if the current approximation has a smaller integral than the previous best
        if integral < best_integral:
            best_approx = convex_approx
            best_integral = integral
            print("the best integral is", iteration)

        # Compare the current local minima indices with the previous iteration
        if (prev_local_minima_indices is not None and np.array_equal(local_minima_indices,
                                                                     prev_local_minima_indices)) or len(
                local_minima_indices) == 0:
            break

        prev_local_minima_indices = local_minima_indices
        iteration += 1

    plt.legend()

    # Save the plot with current date and time as the filename
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"convexify_plot_{current_time}.png"
    plt.savefig(Figure_path + filename)
    plt.close()

    # If the original function has a smaller integral, return it instead of the convex approximation
    if original_integral < best_integral:
        return f
    else:
        return best_approx


# The rest of the code remains unchanged

def create_convex_approximation(x, fx):
    x_np, fx_np = x.numpy(), fx.numpy()
    local_minima_indices = find_local_minima_indices(x_np, fx_np)

    local_minima_indices = np.concatenate(([0], local_minima_indices, [len(x_np) - 1]))

    def convex_approximation(z):
        z = tr.tensor(z, dtype=tr.float32)
        if z.dim() == 0:
            z = z.unsqueeze(0)
        z_np = z.numpy()

        indices = np.searchsorted(x_np[local_minima_indices], z_np, side="right") - 1
        indices = np.clip(indices, 0, len(local_minima_indices) - 2)

        minima_x = x_np[local_minima_indices[indices]]
        minima_fx = fx_np[local_minima_indices[indices]]

        alpha = (z_np - minima_x) / (x_np[local_minima_indices[indices + 1]] - minima_x)
        print(alpha)
        alpha = np.clip(alpha, 0, 1)

        interpolated_fx = (1 - alpha) * minima_fx + alpha * fx_np[local_minima_indices[indices + 1]]

        return tr.tensor(interpolated_fx, dtype=tr.float32)

    return convex_approximation

def find_local_minima_indices(x, fx):
    minima_indices = []
    for i in range(len(x)):
        if i == 0:
            if fx[i] < fx[i + 1]:
                minima_indices.append(i)
        elif i == len(x) - 1:
            if fx[i] < fx[i - 1]:
                minima_indices.append(i)
        else:
            if fx[i - 1] > fx[i] and fx[i] < fx[i + 1]:
                minima_indices.append(i)
    return np.array(minima_indices)


def local_minima_from_samples(f, x_values):
    fx_values = f(x_values)

    # Find the indices of the local minima
    minima_indices = argrelextrema(fx_values.numpy(), np.less)

    # Retrieve the x values corresponding to the minima
    local_minima_x = x_values[minima_indices]
    return local_minima_x


# Time functions
class Time_Function(ABC):
    """Base class for time functions."""
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # Check if input is a tensor and convert it if needed
        if not isinstance(x, tr.Tensor):
            x = tr.tensor([x], dtype=tr.float32)

        # Check if values are in the range [0, 1]
        if (x < 0).any() or (x > 1).any():
            raise ValueError("Inputs must be between 0 and 1")
        else:
            return self._forward(x)

    @abstractmethod
    def _forward(self, x):
        pass

class Polynomial_Time_Function(Time_Function):
    """Class for polynomial time functions."""
    def __init__(self, p):
        super().__init__()
        self.p = p

    def _forward(self, x):
        return x ** self.p

class Exponential_Time_Function(Time_Function):
    """Class for exponential time functions."""

    def __init__(self, Amplitude, decay_rate):
        super().__init__()
        self.Amplitude = Amplitude
        self.decay_rate = decay_rate

    def _forward(self, x):
        return self.Amplitude * tr.exp(-self.decay_rate * x)


class Exponential_Time_Function_linear_decay(Time_Function):
    """Class for exponential time functions."""

    def __init__(self, Amplitude, beginning_rate, final_rate):
        super().__init__()
        self.Amplitude = Amplitude
        self.beginning_rate = beginning_rate
        self.final_rate = final_rate


    def Current_rate(self, x):
        decay_rate = self.beginning_rate + x * (self.final_rate - self.beginning_rate)
        return decay_rate

    def _forward(self, x):
        return self.Amplitude * tr.exp(-(self.Current_rate(x)) * x)


class Custom_Time_Function(Time_Function):
    """Class for custom time functions."""
    def __init__(self):
        super().__init__()

    def _forward(self, x):
        peak_x = 0.1
        left_function = lambda x: 0.03*(1 - (1 * (x - peak_x) ** 2) / (peak_x ** 2))
        right_function = lambda x: 0.03*(1 - (1 * (x - peak_x) ** 2) / ((1 - peak_x) ** 2))

        left_mask = (x <= peak_x)
        right_mask = (x > peak_x)

        result = tr.zeros_like(x)
        result[left_mask] = left_function(x[left_mask])
        result[right_mask] = right_function(x[right_mask])

        return result

# Polynomial best fit and the coefficients (Least squares)

def best_fit_polynomial_coefficients(inputs, outputs, degree):
    """
    Find the best-fit polynomial of the specified degree for the given input-output data.
    """
    # Check tensor dimensions
    assert inputs.dim() == 1, "inputs must be a 1D tensor"
    assert outputs.dim() == 1, "outputs must be a 1D tensor"
    assert inputs.shape[0] == outputs.shape[0], "inputs and outputs must have the same length"

    # Construct the Vandermonde matrix
    vandermonde_matrix = tr.stack([inputs ** i for i in range(degree + 1)], dim=1)

    # Solve the least-squares problem using tr.linalg.lstsq
    coefficients = tr.linalg.lstsq(vandermonde_matrix, outputs.view(-1, 1)).solution
    coefficients = coefficients.view(-1)  # Reshape the solution to the appropriate size

    return coefficients
