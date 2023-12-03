from DataProcessing import Binning_method
from Functions import kNN, Nearest_lower_and_upper_bound_in_sorted_list

import torch as tr
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from sklearn.kernel_ridge import KernelRidge

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

