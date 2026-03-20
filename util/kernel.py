import typing
import torch
import numpy as np
# noinspection PyPep8Naming
import torch.nn.functional as F
from .jacobi_ext import jacobi

__use_cpu: bool = False

def get_device(use_cpu: bool = __use_cpu) -> torch.device:
    return torch.device('cuda') if (not use_cpu and torch.cuda.is_available()) else torch.device('cpu')

__device: torch.device = get_device()

"""
Masked discrete Poisson equation (with arbitray Dirchilet boundary condition): 
        (I - bc_mask) A x = (I - bc_mask) f
             bc_mask    x =        bc_value  
Note for simplicity, 
    we preprocess bc_value s.t. bc_value == bc_mask bc_value, 
    and we have the exterior band of x be zero (trivial boundary condition). 
 
Masked Jacobi update: Step matrix P = -4I
        x' = (I - bc_mask) ( (I - P^-1 A) x                        + P^-1 f ) + bc_value
           = (I - bc_mask) ( F.conv2d(x, jacobi_kernel, padding=1) - 0.25 f ) + bc_value
           = util.jacobi_step(x)
"""

jacobi_kernel = torch.tensor([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]], dtype=torch.float32
                             ).view(1, 1, 3, 3).to(__device) / 4.0

laplace_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32
                              ).view(1, 1, 3, 3).to(__device)

# Half-weight restriction
restriction_kernel = torch.tensor([[0, 1, 0],
                                   [1, 4, 1],
                                   [0, 1, 0]], dtype=torch.float32
                                  ).view(1, 1, 3, 3).to(__device) / 8.0

# P = 20I, A is  the biharmonic kernel.
# Jacobi update: u_{k+1} = (1/20) * ( u_k * kernel + f )
biharmonic_jacobi_kernel = torch.tensor([
    [0,  0,  1,  0,  0],
    [0,  2, -8,  2,  0],
    [1, -8,  0, -8,  1],
    [0,  2, -8,  2,  0],
    [0,  0,  1,  0,  0]], dtype = torch.float32).view(1, 1, 5, 5).to(__device) / -20

biharmonic_kernel = torch.tensor([
    [0,  0,  1,  0,  0],
    [0,  2, -8,  2,  0],
    [1, -8,  20, -8,  1],
    [0,  2, -8,  2,  0],
    [0,  0,  1,  0,  0]], dtype = torch.float32).view(1, 1, 5, 5).to(__device) 

poisson_kernel_5 = torch.tensor([
    [0,  0,  -1,  0,   0],
    [0,  0,  16,  0,   0],
    [-1, 16, -60, 16, -1],
    [0,  0,  16,  0,   0],
    [0,  0,  -1,  0,   0]], dtype = torch.float32).view(1, 1, 5, 5).to(__device) 

def initial_guess(bc_value: torch.Tensor, bc_mask: torch.Tensor, initialization: str) -> torch.Tensor:
    """
    Assemble the initial guess of solution.
    """
    if initialization == 'random':
        routine = torch.rand_like   # U[0, 1)
    elif initialization == 'zero':
        routine = torch.zeros_like
    else:
        raise NotImplementedError

    return (1 - bc_mask) * routine(bc_value) + bc_value


def jacobi_step(x: torch.Tensor, bc_value: torch.Tensor, bc_mask: torch.Tensor, f: typing.Optional[torch.Tensor]):
    """
    One iteration step of masked Jacobi iterative solver.
    """
    # y = laplace_jacobi(x)
    y = F.conv2d(x, jacobi_kernel, padding=1)

    if f is not None:
        y = y - 0.25 * f

    return (1 - bc_mask) * y + bc_value

def jacobi_step_sparse(x: torch.Tensor, bc_value: torch.Tensor, bc_mask: torch.Tensor, f: typing.Optional[torch.Tensor], iters = 1):
    return jacobi.jacobi_step(x, bc_value, bc_mask, f, iters)

def biharmonic_jacobi_step(x: torch.Tensor, bc_value: torch.Tensor, bc_mask: torch.Tensor, f: typing.Optional[torch.Tensor]):
    """
    One iteration step of masked biharmonic iterative solver.  
    """
    omega = 0.2  # weighting to improve stability
    y = biharmonic_jacobi(x)
    # y = F.conv2d(x, biharmonic_jacobi_kernel, padding=2)

    if f is not None:
        # y = 0.05 * f - y
        y = y + 0.05 * f
    # else:
    #     # If f is zero: -y
    #     y = -y

    y = omega * y + (1 - omega) * x
    return (1 - bc_mask) * y + bc_value

    # Debugging and testing with 5x5 poisson kernel
    # y = F.conv2d(x, poisson_jacobi_kernel_5, padding=2)

    # if f is not None:
    #     # y = 0.05 * f - y
    #     y = y + (-1/60) * f

    # return (1 - bc_mask) * y + bc_value

def downsample2x(x: torch.Tensor) -> torch.Tensor:
    """
    Bilinear 2x-downsampling of an image of size 2^N + 1 is essentially direct injection.
    E.g., 257 -> 129 -> 65 -> ...

    Note: torch.nn.UpsamplingBilinear2d is deprecated testcase favor of interpolate.
    It is equivalent to nn.functional.interpolate(..., mode='bilinear', align_corners=True).
    """
    new_size = (x.size(-1) - 1) // 2 + 1
    y = F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)
    return y


def upsample2x(x: torch.Tensor) -> torch.Tensor:
    """
    Bilinear 2x-upsampling of an image of size 2^N + 1.
    E.g., 65 -> 129 -> 257 -> ...

    Note: torch.nn.UpsamplingBilinear2d is deprecated testcase favor of interpolate.
    It is equivalent to nn.functional.interpolate(..., mode='bilinear', align_corners=True).
    """
    new_size = x.size(-1) * 2 - 1
    y = F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)
    return y


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Vector norm on each batch.
    Note: We only deal with cases where channel == 1!
    :param x: (batch_size, channel, image_size, image_size)
    :return: (batch_size,)
    """
    y = x.view(x.size(0), -1)
    return (y * y).sum(dim=1).sqrt()


def absolute_residue(x: torch.Tensor,
                     bc_mask: torch.Tensor,
                     f: typing.Optional[torch.Tensor],
                     reduction: str = 'norm',
                     biharmonic_problem: bool = False) -> torch.Tensor:
    """
    For a linear system Ax = f,
    the absolute residue is r = f - Ax,
    the absolute residual (norm) error eps = ||f - Ax||.
    """
    # eps of size (batch_size, channel (1), image_size, image_size)
    if biharmonic_problem:
        # eps = F.conv2d(x, biharmonic_kernel, padding=2)
        eps = biharmonic(x)
    else:
        # eps = F.conv2d(x, laplace_kernel, padding=1)
        eps = laplacian(x)

    if f is not None:
        eps = eps - f

    eps = eps * (1 - bc_mask)
    eps = eps.view(eps.size(0), -1)            # of size (batch_size, image_size ** 2)

    if reduction == 'norm':
        error = norm(eps)                      # of size (batch_size,)
    elif reduction == 'mean':
        error = torch.abs(eps).mean(dim=1)     # of size (batch_size,)
    elif reduction == 'max':
        error = torch.abs(eps).max(dim=1)[0]   # of size (batch_size,)
    elif reduction == 'none':
        error = -eps                           # of size (batch_size, image_size ** 2)
    else:
        raise NotImplementedError

    return error


def relative_residue(x: torch.Tensor,
                     bc_value: torch.Tensor,
                     bc_mask: torch.Tensor,
                     f: typing.Optional[torch.Tensor], biharmonic_problem=False) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    For a linear system Ax = f, the relative residual error eps = ||f - Ax|| / ||f||.
    :return: abs_residual_error, relative_residual_error
    """
    numerator: torch.Tensor = absolute_residue(x, bc_mask, f, reduction='norm', biharmonic_problem=biharmonic_problem)  # norm of size (batch_size,)

    denominator: torch.Tensor = bc_value                                         # (batch_size, image_size, image_size)

    if f is not None:
        denominator = denominator + f                                            # (batch_size, image_size, image_size)

    denominator = norm(denominator)

    return numerator, numerator / denominator                                    # (batch_size,)


"""
efficiency improvements for im2col
"""
def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    Sample x at neighbour offset (dy, dx):
        output[i, j] = x[i + dy, j + dx]   (zero outside bounds)

    Pad order for F.pad: (left, right, top, bottom).

    To look RIGHT (dx=+1): pad a zero column on the RIGHT, slice from col 1.
    To look LEFT  (dx=-1): pad a zero column on the LEFT,  slice from col 0.
    Same logic applies vertically for dy.

        pad_r = max( dx, 0)   pad right  when looking right
        pad_l = max(-dx, 0)   pad left   when looking left
        pad_b = max( dy, 0)   pad bottom when looking down
        pad_t = max(-dy, 0)   pad top    when looking up

    Slice [pad_b : pad_b+h, pad_r : pad_r+w] to recover original size.
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_l = max(-dx, 0);  pad_r = max( dx, 0)
    pad_t = max(-dy, 0);  pad_b = max( dy, 0)
    x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
    return x_pad[..., pad_b:pad_b+h, pad_r:pad_r+w]

def laplacian(x: torch.Tensor) -> torch.Tensor:
    """
    Replaces: F.conv2d(x, laplace_kernel, padding=1)
    """
    return (_shift(x,  0,  1)   # right neighbour
          + _shift(x,  0, -1)   # left neighbour
          + _shift(x,  1,  0)   # bottom neighbour
          + _shift(x, -1,  0)   # top neighbour
          - 4.0 * x)

def laplace_jacobi(x: torch.Tensor) -> torch.Tensor:
    """
      Replaces: F.conv2d(x, jacobi_kernel, padding=1)
             where jacobi_kernel already has /4 baked in.
    """
    return 0.25 * (_shift(x,  0,  1)
                 + _shift(x,  0, -1)
                 + _shift(x,  1,  0)
                 + _shift(x, -1,  0))

def biharmonic(x: torch.Tensor) -> torch.Tensor:
    """
    Biharmonic operator  ∇⁴x = ∇²(∇²x)  via two Laplacian passes.
    Replaces: F.conv2d(x, biharmonic_kernel, padding=2)
    """
    return laplacian(laplacian(x))

def biharmonic_jacobi(x: torch.Tensor) -> torch.Tensor:
    """

    Replaces: F.conv2d(x, biharmonic_jacobi_kernel, padding=2)

    Since biharmonic(x) = ∇⁴x = off_diag_part + 20·x, we have:
        off_diag_part / 20 = (biharmonic(x) - 20·x) / 20
    """
    return (biharmonic(x) - 20.0 * x) / 20.0