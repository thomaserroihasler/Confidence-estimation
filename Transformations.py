import math as mt
import functools
import torch as tr
import torch.nn as nn

from Functions import heavyside

class Transformation(nn.Module):
    """ Base class for transformations, requiring subclasses to implement the transform method. """
    def __init__(self):
        super(Transformation, self).__init__()

    def transform(self, x):
        # Subclasses should implement this method
        raise NotImplementedError("Subclasses must implement transform method")

    def forward(self, x):
        """ Applies the transformation to the input x """
        y = self.transform(x)
        return y

class Normalization(nn.Module):
    """ Normalization function applying a relative stability transformation. """
    def __init__(self, a, b):
        """ Initialize with parameters a and b for the normalization. """
        super(Normalization, self).__init__()
        self.a = a  # Parameter a
        self.b = b  # Parameter b

    def transform(self, x):
        """ Apply the normalization transformation """
        C = tr.exp(1.0 / (self.b * x) - 1.0) - 1.0
        return 1.0 / (1.0 + tr.exp(-C)) ** self.a

    def forward(self, x):
        """ Apply the normalization transformation to x """
        y = self.transform(x)
        return y

class Temperature_scale(Transformation):
    """ Temperature scaling class for output scaling. """
    def __init__(self, temperature=1.0):
        """ Initialize with a temperature parameter. """
        super(Temperature_scale, self).__init__()
        self.Temperature = nn.Parameter(tr.tensor(temperature))  # Temperature parameter

    def transform(self, x):
        """ Apply temperature scaling to the input x """
        y = x / self.Temperature
        return y

class LinearTransformation(Transformation):
    """ Simple linear transformation block with weight and bias parameters. """
    def __init__(self, in_features, out_features):
        """ Initialize with input and output feature dimensions. """
        super(LinearTransformation, self).__init__()
        self.weight = nn.Parameter(tr.randn(out_features, in_features))  # Weight parameter
        self.bias = nn.Parameter(tr.randn(out_features))  # Bias parameter

    def transform(self, x):
        """ Apply linear transformation to the input x """
        y = tr.matmul(x, self.weight.t()) + self.bias
        return y

class ReshapeTransform:
    """ Class for reshaping a tensor to a specified shape. """
    def __init__(self, new_shape):
        """ Initialize with the new shape for the tensor. """
        self.new_shape = new_shape  # New shape for the tensor

    def __call__(self, img):
        """ Reshape the input tensor to the new shape"""
        return img.view(self.new_shape)

## Stardard image noises

class Noise(Transformation):

    def __init__(self, pixel_wise_parameters = 0):
        super(Noise, self).__init__()
        self.pixel_wise_parameters = pixel_wise_parameters

    def local_noise(self,x):
        raise NotImplementedError("Subclasses must implement transform method")

    def transform(self, x):
        noise = self.local_noise(x)
        y = x + noise
        return y

class Homogenous_Noise(Noise):
    def __init__(self,pixel_wise_parameters):
        super(Homogenous_Noise, self).__init__(pixel_wise_parameters)

    def local_noise(self, x):
        return tr.randn_like(x) * self.pixel_wise_parameters

#### General diffeomorphisms

class Diffeomorphism(Transformation):
    def __init__(self):
        super(Diffeomorphism, self).__init__()

    def deform(self, x):
        raise NotImplementedError("Subclasses must implement deform method")

    def transform(self, x):
        x = self.deform(x)
        return x

class DynamicDiffeomorphism(Diffeomorphism):
    def __init__(self, T, c,Noise = False, seed=None, interp='linear'):
        super(DynamicDiffeomorphism, self).__init__()
        self.T = T
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
        #print('e and s',e.shape,s.shape)
        # We use batch matrix multiplication to handle the batch dimension
        c = F * e.unsqueeze(0)  # Adding batch dimension to e
        #print(c.shape,F.shape,e.unsqueeze(0).shape)
        #print('sinshape',s.shape)
        tensor= tr.einsum('bij,xi,yj->byx', c, s, s)
        #print('tensor shape',tensor.shape)
        return tensor

    def deform(self, x):

        device = x.device
        batch_size = x.shape[0]
        F_x, F_y = self.generate_fields(batch_size, device)

        n = x.shape[-1]
        assert x.shape[-2] == n, 'Image(s) should be square.'

        u = self.scalar_field(n, self.c, F_x, device)
        v = self.scalar_field(n, self.c, F_y, device)
        #print('u and v shapes', u.shape,v.shape)
        #dx should be of shape [batch_size, channel_num, height width]
        dx = (self.T / typical_displacement(1, self.c, n)**2) ** 0.5 * u * n
        dy = (self.T / typical_displacement(1, self.c, n)**2) ** 0.5 * v * n

        # Now we need to apply the displacement to each image in the batch
        deformed_images = tr.stack([self.remap(x[i], dx[i], dy[i], device) for i in range(batch_size)])
        #deformed_images = self.remap(x, dx, dy, device)

        if self.Noise:
            #print(x.shape,deformed_images.shape)
            noise_levels = tr.norm(x - deformed_images, p=2, dim=(1, 2, 3), keepdim=True).squeeze()

            #print(noise_levels.shape)
            #print(x.shape)
            noises_on_sphere = generate_noises_on_sphere(noise_levels, 1, x[0].shape, device).reshape_as(x)
            #print(deformed_images.shape,noises_on_sphere.shape)
            deformed_images = x + noises_on_sphere

        return deformed_images

    def remap(self, a, dx, dy, device):
        n, m = a.shape[-2:]
        #print(dx.shape,(n, m))
        assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'
        dtype = dx.dtype

        y, x = tr.meshgrid(tr.arange(n, dtype=dtype, device=device),
                           tr.arange(m, dtype=dtype, device=device),
                           indexing='ij')
        xn = (x - dx).clamp(0, m - 1)
        yn = (y - dy).clamp(0, n - 1)

        if self.interp == 'linear':
            xf = xn.floor().long()
            yf = yn.floor().long()
            xc = xn.ceil().long()
            yc = yn.ceil().long()

            xv = xn - xf
            yv = yn - yf
            temp = (1 - yv) * (1 - xv) * a[..., yf, xf] + (1 - yv) * xv * a[..., yf, xc] + yv * (1 - xv) * a[..., yc, xf] + yv * xv * a[..., yc, xc]
            return temp


class Standard_diffeomorphism(Diffeomorphism):
    def __init__(self, T, F_x, F_y, seed = None, interp='linear'):

        super(Standard_diffeomorphism, self).__init__()

        self.T = T

        # Check if F_x and F_y are square 2D tensors of the same size
        if len(F_x.shape) != 2 or len(F_y.shape) != 2:
            raise ValueError("F_x and F_y must be 2D tensors")
        if F_x.shape != F_y.shape or F_x.shape[0] != F_x.shape[1] or F_y.shape[0] != F_y.shape[1]:
            raise ValueError("F_x and F_y must be square tensors of the same size")

        self.F_x = F_x
        self.F_y = F_y
        self.interp = interp
        self.seed = seed

    def scalar_field_modes(n, m, dtype=tr.float64, device='cpu'):
        """
        sqrt(1 / Energy per mode) and the modes
        """
        x = tr.linspace(0, 1, n, dtype=dtype, device=device)
        k = tr.arange(1, m + 1, dtype=dtype, device=device)
        i, j = tr.meshgrid(k, k)
        r = (i.pow(2) + j.pow(2)).sqrt()
        e = (r < m + 0.5) / r
        s = tr.sin(mt.pi * x[:, None] * k[None, :])
        return e, s

    def deform(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) != 4:
            raise ValueError("Input must be a 3D or 4D tensor")

        n = x.shape[-1]
        assert x.shape[-2] == n, 'Image(s) should be square.'

        device = x.device.type

        # if self.seed is not None:
        #     tr.manual_seed(self.seed)

        u = scalar_field(n, self.F_x.shape[0], self.F_x, device)  # [n,n]
        v = scalar_field(n, self.F_y.shape[0], self.F_y, device)  # [n,n]
        dx = self.T ** 0.5 * u * n
        dy = self.T ** 0.5 * v * n
        return self.remap(x, dx, dy).contiguous()

    def remap(self, a, dx, dy):
        """
        :param a: Tensor of shape [..., y, x]
        :param dx: Tensor of shape [y, x]
        :param dy: Tensor of shape [y, x]
        :param interp: interpolation method
        """
        n, m = a.shape[-2:]
        assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'

        dtype = dx.dtype
        device = dx.device.type

        y, x = tr.meshgrid(tr.arange(n, dtype=dtype, device=device), tr.arange(m, dtype=dtype, device=device),
                              indexing='ij')

        xn = (x - dx).clamp(0, m - 1)
        yn = (y - dy).clamp(0, n - 1)

        if self.interp == 'linear':
            xf = xn.floor().long()
            yf = yn.floor().long()
            xc = xn.ceil().long()
            yc = yn.ceil().long()

            xv = xn - xf
            yv = yn - yf

            #print(xv.shape,yv.shape,a.shape)
            return (1 - yv) * (1 - xv) * a[..., yf, xf] + (1 - yv) * xv * a[..., yf, xc] + yv * (1 - xv) * a[
                ..., yc, xf] + yv * xv * a[..., yc, xc]

    def temperature_range(self, n, cut):
        if isinstance(cut, (float, int)):
            log = mt.log(cut)
        else:
            log = cut.log()
        T1 = 1 / (mt.pi * n ** 2 * log)
        T2 = 4 / (mt.pi ** 3 * cut ** 2 * log)
        return T1, T2

    def typical_displacement(self, T, cut, n):
        if isinstance(cut, (float, int)):
            log = mt.log(cut)
        else:
            log = cut.log()
        return n * (mt.pi * T * log) ** .5 / 2

    def get_cutoff(self):
        return self.F_x.shape[0]

# DEFINE FOR A DATA_POINT

# Useful functions for unit_transformations:

def Total_color(x: tr.Tensor):
    return round((tr.round(x).sum() / x.numel()).item())

#### Special transformations

class unit_Transformation(Transformation):
    def __init__(self,Total_norm):
        super(unit_Transformation, self).__init__()
        self.total_norm = Total_norm
        self.pixel_norm  = None

    def Average_color(self,x):
        return x.mean(dim = -1).round()

    def Average_Average_color(self, x):
        return self.Average_color(x.round())

    def pixel_direction(self, x):
        raise NotImplementedError("Subclasses must implement d method")

    def get_pixel_norm(self, x):

        if (self.pixel_norm == None):
            self.pixel_norm = self.total_norm/x.shape[-1]*tr.randint(0, 2, (x.shape[-1],)).float()

        else:
            if (self.pixel_norm.shape != x.shape[-1]):
                self.pixel_norm = self.total_norm/x.shape[-1]*tr.randint(0, 2, (x.shape[-1],)).float()

        A = self.pixel_norm

        return  A

    def rounder(self, x):
        return x - tr.floor(x)

    def transform(self, x):
        # Seperate between the single input case and the batch of inputs case
        avg_color = self.Average_Average_color(x)
        pixel_norm = self.get_pixel_norm(x)

        if (len(x.shape)>1):
            pixel_norm =  pixel_norm.unsqueeze(0).repeat(x.shape[0], 1)

        pixel_dir = self.pixel_direction(x)


        x = x + pixel_norm * pixel_dir
        x = self.rounder(x)
        return x

class Black_and_white_noise(unit_Transformation):
    def __init__(self):
        super(Black_and_white_noise, self).__init__()
        #self.M = M

    def pixel_direction(self, x):
        random_tensor = tr.rand(x.shape)
        return (random_tensor >= 0.5).float() * 2 - 1


class Black_and_white_diffeomorphism(unit_Transformation):
    def __init__(self, Total_norm):
        super(Black_and_white_diffeomorphism, self).__init__(Total_norm=Total_norm)

    def pixel_direction(self, x):
        return 2 * (1 - (heavyside(x - 1 / 4) + heavyside(x - 3 / 4)))+tr.sign(x-1/2)


@functools.lru_cache()

def scalar_field_modes(n, m, dtype=tr.float64, device='cpu'):
    """
    sqrt(1 / Energy per mode) and the modes
    """
    x = tr.linspace(0, 1, n, dtype=dtype, device=device)
    k = tr.arange(1, m + 1, dtype=dtype, device=device)
    i, j = tr.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m + 0.5) / r
    s = tr.sin(mt.pi * x[:, None] * k[None, :])
    return e, s

def non_zero_mode(n, m, dtype=tr.float64, device='cpu'):
    """
    sqrt(1 / Energy per mode) and the modes
    """
    x = tr.linspace(0, 1, n, dtype=dtype, device=device)
    k = tr.arange(1, m + 1, dtype=dtype, device=device)
    i, j = tr.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m + 0.5)
    return e

def scalar_field(n, m, F, device='cpu'):
    """
    random scalar field of size nxn made of the first m modes
    """
    e, s = scalar_field_modes(n, m, dtype=tr.get_default_dtype(), device=device)
    c = F * e
    return tr.einsum('ij,xi,yj->yx', c, s, s)

def deform(image,T, F_x,F_y, interp='linear', seed=None):
    """
    1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
    2. Apply tau to `image`
    :param img Tensor: square image(s) [..., y, x]
    :param T float: temperature
    :param cut int: high frequency cutoff
    """

    n = image.shape[-1]
    assert image.shape[-2] == n, 'Image(s) should be square.'

    device = image.device.type

    # if seed is not None:
    #     tr.manual_seed(seed)

    # Sample dx, dy
    # u, v are defined in [0, 1]^2
    # dx, dx are defined in [0, n]^2
    u = scalar_field(n, F_x.shape[0], F_x, device)  # [n,n]
    v = scalar_field(n, F_y.shape[0], F_y, device)  # [n,n]
    dx = T ** 0.5 * u * n
    dy = T ** 0.5 * v * n
    # Apply tau
    #print(np.sqrt((dx**2+dy**2).mean()))
    return remap(image, dx, dy, interp).contiguous()

def remap(a, dx, dy, interp):
    """
    :param a: Tensor of shape [..., y, x]
    :param dx: Tensor of shape [y, x]
    :param dy: Tensor of shape [y, x]
    """
    n, m = a.shape[-2:]
    #     assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'

    dtype = dx.dtype
    device = dx.device.type
    B = a.shape[0]

    y, x = tr.meshgrid(tr.arange(n, dtype=dtype, device=device), tr.arange(m, dtype=dtype, device=device),
                          indexing='ij')

    xn = (x + dx).clamp(0, m - 1)
    yn = (y + dy).clamp(0, n - 1)

    if interp == 'linear':
        xf = xn.floor().long()
        yf = yn.floor().long()
        xc = xn.ceil().long()
        yc = yn.ceil().long()

        xv = (xn - xf).unsqueeze(1)
        yv = (yn - yf).unsqueeze(1)

        return (1 - yv) * (1 - xv) * a[tr.arange(B)[:, None, None], ..., yf, xf].permute(0, 3, 1, 2) + (
                    1 - yv) * xv * a[tr.arange(B)[:, None, None], ..., yf, xc].permute(0, 3, 1, 2) + yv * (1 - xv) * \
               a[tr.arange(B)[:, None, None], ..., yc, xf].permute(0, 3, 1, 2) + yv * xv * a[
                   tr.arange(B)[:, None, None], ..., yc, xc].permute(0, 3, 1, 2)

def temperature_range(n, cut):
    """
    Define the range of allowed temperature
    for given image size and cut.
    """
    if isinstance(cut, (float, int)):
        log = mt.log(cut)
    else:
        log = cut.log()
    T1 = 1 / (mt.pi * n ** 2 * log)
    T2 = 4 / (mt.pi ** 3 * cut ** 2 * log)
    return T1, T2

def typical_displacement(T, cut, n):
    if isinstance(cut, (float, int)):
        log = mt.log(cut)
    else:
        log = cut.log()
    return n * (mt.pi * T * log) ** .5 / 2

class ImageDiffeomorphismOLD(Diffeomorphism):
    def __init__(self, Temperature, cut_off):
        super(ImageDiffeomorphismOLD, self).__init__()
        self.Temperature = Temperature
        self.cut_off = cut_off

    def deform(self, x):
        return deform(x, self.Temperature,self.cut_off,x.device)


class ImageDiffeomorphism(Diffeomorphism):
    def __init__(self, Fourier_Coefficents):
        super(ImageDiffeomorphism, self).__init__()
        self.Fourier_Coefficents = Fourier_Coefficents
    def deform(self, x):
        return deform(x, self.Fourier_Coefficents[0],self.Fourier_Coefficents[1],x.device)














#
#
# """
# Computes diffeomorphism of 2D images in pytorch
# """
# @functools.lru_cache()
# def scalar_field_modes(n, m, dtype=torch.float64, device='cpu'):
#     """
#     sqrt(1 / Energy per mode) and the modes
#     """
#     x = torch.linspace(0, 1, n, dtype=dtype, device=device)
#     k = torch.arange(1, m + 1, dtype=dtype, device=device)
#     i, j = torch.meshgrid(k, k)
#     r = (i.pow(2) + j.pow(2)).sqrt()
#     e = (r < m + 0.5) / r
#     s = torch.sin(math.pi * x[:, None] * k[None, :])
#     return e, s
#
# def non_zero_mode(n, m, dtype=torch.float64, device='cpu'):
#     """
#     sqrt(1 / Energy per mode) and the modes
#     """
#     x = torch.linspace(0, 1, n, dtype=dtype, device=device)
#     k = torch.arange(1, m + 1, dtype=dtype, device=device)
#     i, j = torch.meshgrid(k, k)
#     r = (i.pow(2) + j.pow(2)).sqrt()
#     e = (r < m + 0.5)
#     return e
#
# def scalar_field(n, m, B,C, device='cpu'):
#     """
#     random scalar field of size nxn made of the first m modes
#     """
#     e, s = scalar_field_modes(n, m, dtype=torch.get_default_dtype(), device=device)
#     c = C * e
#     return torch.einsum('bij,xi,yj->byx', c, s, s)
#
# def scalar_fieldOLD(n, m,C, device='cpu'):
#     """
#     random scalar field of size nxn made of the first m modes
#     """
#     e, s = scalar_field_modes(n, m, dtype=torch.get_default_dtype(), device=device)
#     c = C * e
#     #print(c)
#     return torch.einsum('ij,xi,yj->yx', c, s, s)
#
#
# def scalar_field_OLDOLD(n, m, device='cpu'):
#     """
#     random scalar field of size nxn made of the first m modes
#     """
#     e, s = scalar_field_modes(n, m, dtype=torch.get_default_dtype(), device=device)
#     C = torch.randn(m, m, device=device)
#     c = C * e
#     #print(c)
#     return torch.einsum('ij,xi,yj->yx', c, s, s)
#
#
# def deform_OLDOLD(image, T, cut, interp='linear'):
#     """
#     1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
#     2. Apply tau to `image`
#     :param img Tensor: square image(s) [..., y, x]
#     :param T float: temperature
#     :param cut int: high frequency cutoff
#     """
#     n = image.shape[-1]
#     assert image.shape[-2] == n, 'Image(s) should be square.'
#
#     device = image.device.type
#
#     # Sample dx, dy
#     # u, v are defined in [0, 1]^2
#     # dx, dx are defined in [0, n]^2
#     u = scalar_field_OLDOLD(n, cut, device)  # [n,n]
#     v = scalar_field_OLDOLD(n, cut, device)  # [n,n]
#     dx = T ** 0.5 * u * n
#     dy = T ** 0.5 * v * n
#
#     # Apply tau
#     return remap(image, dx, dy, interp).contiguous()
#
# def deformOLD(image, T, cut,C, interp='linear'):
#     """
#     1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
#     2. Apply tau to `image`
#     :param img Tensor: square image(s) [..., y, x]
#     :param T float: temperature
#     :param cut int: high frequency cutoff
#     """
#     n = image.shape[-1]
#     assert image.shape[-2] == n, 'Image(s) should be square.'
#     device = image.device.type
#     # Sample dx, dy
#     # u, v are defined in [0, 1]^2
#     # dx, dx are defined in [0, n]^2
#     u = scalar_fieldOLD(n, cut,C, device)  # [n,n]
#     v = scalar_fieldOLD(n, cut,C, device)  # [n,n]
#     dx = T ** 0.5 * u * n
#     dy = T ** 0.5 * v * n
#
#     # Apply tau
#     return remapOLD(image, dx, dy, interp).contiguous()
#
# def deform(image, T, cut,C, interp='linear', seed=None):
#     """
#     1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
#     2. Apply tau to `image`
#     :param img Tensor: square image(s) [..., y, x]
#     :param T float: temperature
#     :param cut int: high frequency cutoff
#     """
#     n = image.shape[-1]
#     assert image.shape[-2] == n, 'Image(s) should be square.'
#
#     device = image.device.type
#     B = image.shape[0]
#
#
#     if seed is not None:
#         torch.manual_seed(seed)
#
#     # Sample dx, dy
#     # u, v are defined in [0, 1]^2
#     # dx, dx are defined in [0, n]^2
#     u = scalar_field(n, cut, B,C, device)  # [n,n]
#     v = scalar_field(n, cut, B,C, device)  # [n,n]
#     dx = T ** 0.5 * u * n
#     dy = T ** 0.5 * v * n
#     # Apply tau
#     #print(np.sqrt((dx**2+dy**2).mean()))
#     return remap(image, dx, dy, interp).contiguous()
#
# def remapOLD(a, dx, dy, interp):
#     """
#     :param a: Tensor of shape [..., y, x]
#     :param dx: Tensor of shape [y, x]
#     :param dy: Tensor of shape [y, x]
#     :param interp: interpolation method
#     """
#     n, m = a.shape[-2:]
#     assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'
#     y, x = torch.meshgrid(torch.arange(n, dtype=dx.dtype, device = dx.device.type), torch.arange(m, dtype=dx.dtype, device = dx.device.type))
#
#     xn = (x - dx).clamp(0, m-1)
#     yn = (y - dy).clamp(0, n-1)
#
#     if interp == 'linear':
#         xf = xn.floor().long()
#         yf = yn.floor().long()
#         xc = xn.ceil().long()
#         yc = yn.ceil().long()
#
#         xv = xn - xf
#         yv = yn - yf
#
#         return (1-yv)*(1-xv)*a[..., yf, xf] + (1-yv)*xv*a[..., yf, xc] + yv*(1-xv)*a[..., yc, xf] + yv*xv*a[..., yc, xc]
#
#     if interp == 'gaussian':
#         # can be implemented more efficiently by adding a cutoff to the Gaussian
#         sigma = 0.4715
#
#         dx = (xn[:, :, None, None] - x)
#         dy = (yn[:, :, None, None] - y)
#
#         c = (-dx**2 - dy**2).div(2 * sigma**2).exp()
#         c = c / c.sum([2, 3], keepdim=True)
#
#         return (c * a[..., None, None, :, :]).sum([-1, -2])
#
# def remap(a, dx, dy, interp):
#     """
#     :param a: Tensor of shape [..., y, x]
#     :param dx: Tensor of shape [y, x]
#     :param dy: Tensor of shape [y, x]
#     """
#     n, m = a.shape[-2:]
#     #     assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'
#
#     dtype = dx.dtype
#     device = dx.device.type
#     B = a.shape[0]
#
#     y, x = torch.meshgrid(torch.arange(n, dtype=dtype, device=device), torch.arange(m, dtype=dtype, device=device),
#                           indexing='ij')
#
#     xn = (x + dx).clamp(0, m - 1)
#     yn = (y + dy).clamp(0, n - 1)
#
#     if interp == 'linear':
#         xf = xn.floor().long()
#         yf = yn.floor().long()
#         xc = xn.ceil().long()
#         yc = yn.ceil().long()
#
#         xv = (xn - xf).unsqueeze(1)
#         yv = (yn - yf).unsqueeze(1)
#
#         return (1 - yv) * (1 - xv) * a[torch.arange(B)[:, None, None], ..., yf, xf].permute(0, 3, 1, 2) + (
#                     1 - yv) * xv * a[torch.arange(B)[:, None, None], ..., yf, xc].permute(0, 3, 1, 2) + yv * (1 - xv) * \
#                a[torch.arange(B)[:, None, None], ..., yc, xf].permute(0, 3, 1, 2) + yv * xv * a[
#                    torch.arange(B)[:, None, None], ..., yc, xc].permute(0, 3, 1, 2)
#
#
#
# def temperature_range(n, cut):
#     """
#     Define the range of allowed temperature
#     for given image size and cut.
#     """
#     if isinstance(cut, (float, int)):
#         log = math.log(cut)
#     else:
#         log = cut.log()
#     T1 = 1 / (math.pi * n ** 2 * log)
#     T2 = 4 / (math.pi ** 3 * cut ** 2 * log)
#     return T1, T2
#
# def typical_displacement(T, cut, n):
#     if isinstance(cut, (float, int)):
#         log = math.log(cut)
#     else:
#         log = cut.log()
#     return n * (math.pi * T * log) ** .5 / 2
#
# def TransVectors(X,temperature,cut_off, parameters, diffeo):
#     diffeo = torch.tensor(diffeo, dtype = torch.float32)
#     X_new = torch.clone(X)
#     norms = []
#     for i in range(0,len(X_new)):
#         for j in range(0,len(parameters[i])):
#             X_new[i] += parameters[i,j]*diffeo[j]
#         norms += [torch.norm(X[i] - X_new[i]).item()]
#     return X_new, norms
#
#
# def TransImages(X, T, cut, C_list):
#     device = X.device.type
#     X_new = torch.clone(X).to(device)
#     X_new = deform(X, T, cut, C_list, interp='linear')
#     norms = [torch.norm(X-X_new).item()]
#     return X_new.contiguous(), norms
#
#
# def TransImagesOLD(X,T,cut,C_list):
#     device = X.device.type
#     X_new = torch.clone(X).to(device)
#     norms = []
#
#     for i in range(0,X.shape[0]):
#         X_new[i] =  deform(X[i], T, cut ,C_list[i], interp='linear')
#         norms +=[torch.norm(X[i]-X_new[i]).item()]
#
#     return X_new.contiguous(), norms
#
# class Transformations:
#     def __init__(self, temperature, cut_off, parameters ,Model,Data_type,dimension,index_of_discriminating_factor, diffeo):
#
#         self.Model = Model
#         self.Data_type = Data_type
#         self.dimension = dimension
#         self.index_of_discriminating_factor = index_of_discriminating_factor
#         self.diffeo = diffeo
#
#         self.temperature = temperature
#         self.cut_off = cut_off
#         self.parameters = parameters
#
#     def Diffeomorphism(self, input):
#         if (self.Data_type == "Images"):
#             return TransImages(input, self.temperature, self.cut_off, self.parameters)
#         else:
#             return TransVectors(input, self.temperature,self.cut_off, self.parameters, self.diffeo)
#
#     def Noise(self, input, norms):
#         X_new = torch.clone(input)
#         delta = torch.rand(input.shape)
#         for i in range(0,len(input)):
#             delta[i] *= norms[i]/torch.norm(delta[i])
#             X_new[i] += delta[i]
#         return X_new + delta, norms
#
# def Diffeo_shape(diffeo_shape,cut_off):
#     diffeo_shape = np.array(diffeo_shape)
#     total_parameters = 0
#     for i in range(0, len(diffeo_shape)):
#         diffeo_shape[i] *= (cut_off / diffeo_shape[i])
#         total_parameters += diffeo_shape[i]
#     diffeo_shape = diffeo_shape.tolist()
#     diffeo_shape = tuple(diffeo_shape)
#     return diffeo_shape, total_parameters
#
# def diffeo_distr(batch_size,diffeo_shape,device):
#     total = 1
#     for p in diffeo_shape:
#         total *= p
#     loc = torch.zeros(int(batch_size*total)).to(device)
#     covariance_matrix = torch.eye(int(batch_size*total)).to(device)
#     distr = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix, precision_matrix=None, scale_tril=None, validate_args=None)
#     return distr
# #
# def diffeo_parameters(distr,shape,diffeo_shape,device, temperature):
#     para = distr.sample().to(device)
#     para = torch.reshape(para, shape)
#     if(len(diffeo_shape)==2):
#         for p in para:
#             for i in range(0, diffeo_shape[0]):
#                 for j in range(0, diffeo_shape[1]):
#                     p[i, j] *= mt.sqrt(temperature / ((i + 1) ** 2 + (j + 1) ** 2))
#     else:
#         para = torch.zeros(shape)
#     return para