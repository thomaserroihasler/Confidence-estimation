import sys
import math as mt
import torch as tr
import torch.nn as nn

new_path =  sys.path[0].split("Confidence_Estimation")[0] + "Confidence_Estimation"
sys.path[0] = new_path

from Confidence_Estimation.Other.Useful_functions.definitions import generate_noises_on_sphere

class Transformation(nn.Module):
    """
    Base class for transformations, requiring subclasses to implement the transform method.
    """
    def __init__(self):
        super(Transformation, self).__init__()

    def transform(self, x):
        # Subclasses should implement this method
        raise NotImplementedError("Subclasses must implement transform method")

    def forward(self, x):
        # Applies the transformation to the input x
        y = self.transform(x)
        return y

class Temperature_scale(Transformation):
    """
    Temperature scaling class for output scaling.
    Initialize with a temperature parameter.
    """
    def __init__(self, temperature=1.0):
        super(Temperature_scale, self).__init__()
        self.Temperature = nn.Parameter(tr.tensor(temperature))  # Temperature parameter

    def transform(self, x):
        # Apply temperature scaling to the input x
        y = x / self.Temperature
        return y

class LinearTransformation(Transformation):
    """
    Simple linear transformation block with weight and bias parameters.
    Initialize with input and output feature dimensions.
    """
    def __init__(self, in_features, out_features):
        super(LinearTransformation, self).__init__()
        self.weight = nn.Parameter(tr.randn(out_features, in_features))  # Weight parameter
        self.bias = nn.Parameter(tr.randn(out_features))  # Bias parameter

    def transform(self, x):
        # Apply linear transformation to the input x
        y = tr.matmul(x, self.weight.t()) + self.bias
        return y

class ReshapeTransform:
    """
    Class for reshaping a tensor to a specified shape.
    Initialize with the new shape for the tensor.
    """
    def __init__(self, new_shape):
        self.new_shape = new_shape  # New shape for the tensor

    def __call__(self, img):
        # Reshape the input tensor to the new shape
        return img.view(self.new_shape)

## Stardard image noises

class Noise(Transformation):
    """
    Base class for noise transformations. Requires subclasses to implement the local_noise method.
    """
    def __init__(self, pixel_wise_parameters=0):
        super(Noise, self).__init__()
        self.pixel_wise_parameters = pixel_wise_parameters

    def local_noise(self, x):
        # Subclasses should implement this method
        raise NotImplementedError("Subclasses must implement local_noise method")

    def transform(self, x):
        # Applies local noise to the input x
        noise = self.local_noise(x)
        y = x + noise
        return y

class Homogenous_Noise(Noise):
    """
    Homogenous_Noise: Implements homogenous noise transformation.
    """
    def __init__(self, pixel_wise_parameters):
        super(Homogenous_Noise, self).__init__(pixel_wise_parameters)

    def local_noise(self, x):
        # Generates and returns homogenous noise
        return tr.randn_like(x) * self.pixel_wise_parameters

#### General diffeomorphisms
class Diffeomorphism(Transformation):
    """
    Diffeomorphism: Base class for diffeomorphism transformations.
    """
    def __init__(self):
        super(Diffeomorphism, self).__init__()

    def deform(self, x):
        # Subclasses should implement this method
        raise NotImplementedError("Subclasses must implement deform method")

    def transform(self, x):
        # Applies deformation to the input x
        x = self.deform(x)
        return x

class DynamicDiffeomorphism(Diffeomorphism):
    """
    DynamicDiffeomorphism: Implements dynamic diffeomorphism transformations.
    """

    def __init__(self, Temperature_scale, c,Noise = False, seed=None, interp='linear'):
        super(DynamicDiffeomorphism, self).__init__()
        self.T = Temperature_scale
        self.c = c
        self.interp = interp
        self.seed = seed
        self.Noise = Noise  # Added Noise attribute

    def generate_fields(self, batch_size, device):
        # Creating a batch of random fields with an additional dimension for batch size
        F_x = tr.randn(batch_size, self.c, self.c, device=device)
        F_y = tr.randn(batch_size, self.c, self.c, device=device)
        return F_x, F_y

    def scalar_field_modes(self, n, m, field, device):
        x = tr.linspace(0, 1, n, dtype=field.dtype, device=device)
        k = tr.arange(1, m + 1, dtype=field.dtype, device=device)
        i, j = tr.meshgrid(k, k)
        r = (i.pow(2) + j.pow(2)).sqrt()
        e = (r < m + 0.5) / r
        s = tr.sin(mt.pi * x[:, None] * k[None, :])
        return e, s

    def scalar_field(self, n, m, F, device):
        # Adapted to handle a batch of fields (3D tensors)
        e, s = self.scalar_field_modes(n, m, F[0], device)  # Only use the first field to get modes
        # We use batch matrix multiplication to handle the batch dimension
        c = F * e.unsqueeze(0)  # Adding batch dimension to e
        tensor= tr.einsum('bij,xi,yj->byx', c, s, s)
        return tensor

    def typical_displacement(self, n):
        if isinstance(self.c, (float, int)):
            log = mt.log(self.c)
        else:
            log = self.c.log()
        return n * (mt.pi * self.T * log) ** .5 / 2

    def deform(self, x):

        device = x.device
        batch_size = x.shape[0]
        F_x, F_y = self.generate_fields(batch_size, device)

        n = x.shape[-1]
        assert x.shape[-2] == n, 'Image(s) should be square.'

        u = self.scalar_field(n, self.c, F_x, device)
        v = self.scalar_field(n, self.c, F_y, device)

        dx = (self.T / self.typical_displacement(n)**2) ** 0.5 * u * n
        dy = (self.T / self.typical_displacement(n)**2) ** 0.5 * v * n

        # Now we need to apply the displacement to each image in the batch

        #deformed_images = tr.stack([self.remap(x[i], dx[i], dy[i], device) for i in range(batch_size)])
        deformed_images = self.remap(x, dx, dy, device)

        if self.Noise:
            noise_levels = tr.norm(x - deformed_images, p=2, dim=(1, 2, 3), keepdim=True).squeeze()
            noises_on_sphere = generate_noises_on_sphere(noise_levels, 1, x[0].shape, device).reshape_as(x)
            deformed_images = x + noises_on_sphere


        return deformed_images
    #
    # def  remap(self, a, dx, dy, device):
    #     # switch first and second dimension of a:
    #
    #     """
    #     :param a: Tensor of shape [..., y, x]
    #     :param dx: Tensor of shape [y, x]
    #     :param dy: Tensor of shape [y, x]
    #     :param interp: interpolation method
    #     """
    #     n, m = a.shape[-2:]
    #     #assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'
    #
    #     y, x = tr.meshgrid(tr.arange(n, dtype=dx.dtype, device=device),
    #                           tr.arange(m, dtype=dx.dtype, device=device))
    #
    #     xn = (x - dx).clamp(0, m - 1)
    #     yn = (y - dy).clamp(0, n - 1)
    #
    #     xf = xn.floor().long()
    #     yf = yn.floor().long()
    #     xc = xn.ceil().long()
    #     yc = yn.ceil().long()
    #
    #     xv = xn - xf
    #     yv = yn - yf
    #
    #     return (1 - yv) * (1 - xv) * a[..., yf, xf] + (1 - yv) * xv * a[..., yf, xc] + yv * (1 - xv) * a[..., yc, xf] + yv * xv * a[..., yc, xc]

    def remap(self, a, dx, dy, device):
        a = tr.transpose(a,1,0)
        b, n, m = a.shape[-3:]

        assert dx.shape[-2:] == (n, m) and dy.shape[-2:] == (n, m), 'Image(s) and displacement fields shapes should match.'
        dtype = dx.dtype

        y, x = tr.meshgrid(tr.arange(n, dtype=dtype, device=device),
                           tr.arange(m, dtype=dtype, device=device),
                           indexing='ij')
        B = tr.arange(b).view(b, 1, 1).expand(b, n, m)
        xn = (x - dx).clamp(0, m - 1)
        yn = (y - dy).clamp(0, n - 1)

        if self.interp == 'linear':
            xf = xn.floor().long()
            yf = yn.floor().long()
            xc = xn.ceil().long()
            yc = yn.ceil().long()
            xv = xn - xf
            yv = yn - yf
            temp = (1 - yv) * (1 - xv) * a[...,B, yf, xf] + (1 - yv) * xv * a[..., B,yf, xc] + yv * (1 - xv) * a[...,B, yc, xf] + yv * xv * a[..., B,yc, xc]
            a =  tr.transpose(a,1,0)
            temp = tr.transpose(temp,1,0)
            return temp
