import numpy as np
import torch as tr

def handle_nan(arr):
    """
    Replace NaN values in the array with zeros
    """
    return np.nan_to_num(arr, nan=0)

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
    """
    Finds closest value greater and lower than x.
    """

    values, indices = tr.sort(values)
    return Nearest_lower_and_upper_bound_in_sorted_list(values, x)

def get_accuracy(loader, model, device):
    """
    Calculate the accuracy of a model on a given dataset.

    Args:
        loader: DataLoader for the dataset.
        model: Neural network model to evaluate.
        device: The device (CPU or GPU) to perform calculations on.

    Returns:
        Accuracy as a percentage.
    """
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode to disable dropout, etc.
    with tr.no_grad():  # Disable gradient calculations
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the specified device
            outputs = model(images)
            _, predicted = tr.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set the model back to training mode
    return 100 * correct / total

def Single_bit_entropy_func(a):
    """
    Calculate the single-bit entropy of a probability value.

    Args:
        a: Probability value.

    Returns:
        Single-bit entropy value.
    """
    return -1 * ((a * tr.log(a + 1e-5) + (1 - a) * tr.log((1 - a + 1e-5))))

def monte_carlo_integral(f, points):
    """
    Calculate an integral using the Monte Carlo method.

    Args:
        f: Function to integrate.
        points: Random points used for the Monte Carlo approximation.

    Returns:
        Approximated integral value.
    """
    integral_approx = tr.sum(f(points))
    integral_approx /= len(points)
    return integral_approx

def simple_integral(f, x):
    """
    Calculate an integral using the trapezoidal rule.

    Args:
        f: Function to integrate.
        x: Points at which to evaluate the function.

    Returns:
        Approximated integral value.
    """
    y = f(x)
    dx = x[1:] - x[:-1]
    integral = tr.sum(y[:-1] * dx)
    return integral.item()

def simple_2D_integral(f, N):
    """
    Calculate a 2D integral over a square domain [0,1]x[0,1] using the Monte Carlo method.

    Args:
        f: Function to integrate.
        N: Number of points along each axis.

    Returns:
        Approximated integral value.
    """
    x = tr.linspace(0, 1, N)
    y = tr.linspace(0, 1, N)
    X, Y = tr.meshgrid(x, y)
    points = tr.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    integral_approx = tr.sum(f(points))
    integral_approx /= N ** 2
    return integral_approx

def Integrate(x, y1, y2):
    """
    Calculate the integral of the difference between two functions over a given range.

    Args:
        x: Points at which the functions are evaluated.
        y1: Values of the first function.
        y2: Values of the second function.

    Returns:
        Integral value of the difference between the two functions.
    """
    product = y1 - y2
    integral = tr.trapz(product, x)
    return integral

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






