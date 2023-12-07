import numpy as np
import torch as tr
import matplotlib.pyplot as plt
from PIL import Image
import math as mt

def heavyside(x):
    """Applies the heavy side function to x."""
    return (x >= 0).float()

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

def Bin_edges(data, num_bins=None, num_per_bin=5):
    """ Bin data into a specified number of bins. """
    data = data.to(dtype=tr.float64)  # Convert to higher precision
    data = tr.sort(data).values  # Sort the data

    # Determine bin edges based on quantiles
    if num_bins is not None:
        # Fixed number of bins
        quantiles = tr.linspace(0, 1, num_bins + 1).to(dtype=tr.float64)[1:-1]  # Exclude 0 and 1
        bin_edges = tr.quantile(data, quantiles)
    else:
        # Determine the number of bins based on num_per_bin
        num_bins = max(mt.ceil(len(data) / num_per_bin), 1)
        quantiles = tr.linspace(0, 1, num_bins + 1).to(dtype=tr.float64)[1:-1]  # Exclude 0 and 1
        bin_edges = tr.quantile(data, quantiles)

    # Finalize bin edges
    unique_edges = tr.unique(bin_edges)  # Remove duplicates
    sorted_edges = tr.sort(unique_edges).values  # Sort edges
    sorted_edges = tr.cat((tr.tensor([data.min()]), sorted_edges, tr.tensor([data.max()])))  # Add min and max
    return sorted_edges

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


def show_tensor(tensor, filename):
    """ Convert a PyTorch tensor to an image and display/save it. """
    array = tensor.cpu().numpy()  # Convert tensor to NumPy array
    dims = len(array.shape)  # Get the number of dimensions

    # Create and display the image
    if dims == 1:
        # 1D tensor to grayscale image
        image = np.zeros((1, array.shape[0]), dtype=np.uint8)
        image[0, :] = (array * 255).astype(np.uint8)
    elif dims == 3:
        # 3D tensor to color image
        image = np.transpose((array * 255).astype(np.uint8), (1, 2, 0))
    else:
        # 2D tensor to black and white image
        image = (array * 255).astype(np.uint8)

    plt.imshow(image, cmap='gray' if dims == 1 or dims == 2 else None)
    plt.axis('off')
    plt.show()

    # Save the image if filename is provided
    if filename is not None:
        image = Image.fromarray(image)
        image.save(filename)

def sample_points_on_sphere(num_points, dim, device):
    """Sample uniformly distributed points on a hypersphere of the given dimension."""
    vec = tr.randn(num_points, dim, device=device)
    vec /= vec.norm(dim=1, keepdim=True)
    return vec

def generate_noises_on_sphere(T, N_t, shape, device):
    """Generate N_t noises of dimension d for each element in T with norm equal to T[i]
       and uniformly distributed on the sphere of dimension d."""
    T = T.reshape(-1, 1, 1).to(device)  # reshape T to (N_b, 1, 1)
    tensor_shape_as_tensor = tr.tensor(shape, dtype=tr.int64, device=device)
    d = tensor_shape_as_tensor.prod()
    N_b = T.shape[0]
    sphere_points = sample_points_on_sphere(N_t, d, device)  # sample sphere points
    sphere_points = sphere_points.reshape(1, N_t, d).to(device)  # reshape to (1, N_t, d)
    sphere_points = sphere_points.expand(T.shape[0], -1, -1)  # expand to (N_b, N_t, d)
    noises = T * sphere_points  # scale sphere points by T
    assert noises.numel() == d * N_b * N_t


    return noises.view(N_b, N_t, *shape)

def shuffle_and_split(tensor,permutation):
    total_samples = tensor.size(0)
    # Apply the permutation to shuffle the data
    shuffled_tensor = tensor[permutation]
    # Find the midpoint
    midpoint = total_samples // 2
    # Split the tensor into two parts
    return shuffled_tensor[:midpoint], shuffled_tensor[midpoint:]

def print_device_name():
    device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
    print('the current device is', device)
    return device

def plot_sample_images(loader, title, n=5):
    """Plots n sample images from the given data loader"""
    dataiter = iter(loader)
    images, labels = dataiter.next()
    fig, axes = plt.subplots(1, n, figsize=(10, 3))

    for i, ax in enumerate(axes):
        if images[i].shape[0] == 1:  # if grayscale
            ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        else:  # if RGB
            ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')

    plt.suptitle(title)
    plt.show()