import math
import typing

from bicubic_pytorch import core
from utils import svf
from utils import functional

import cv2

import torch
from torch import nn
from torch.nn import functional as F

def inverse_3x3(m: torch.Tensor) -> torch.Tensor:
    '''
    Hard-coded matrix inversion for 3x3 matrices.

    Args:
        m (torch.Tensor): (3, 3) transformation matrix.

    Return:
        torch.Tensor: m^{-1}, which is calculated from cofactors.
    '''
    n = m.cpu().numpy()
    cofactor_00 = n[1, 1] * n[2, 2] - n[1, 2] * n[2, 1]
    cofactor_01 = n[1, 2] * n[2, 0] - n[1, 0] * n[2, 2]
    cofactor_02 = n[1, 0] * n[2, 1] - n[1, 1] * n[2, 0]
    cofactor_10 = n[0, 2] * n[2, 1] - n[0, 1] * n[2, 2]
    cofactor_11 = n[0, 0] * n[2, 2] - n[0, 2] * n[2, 0]
    cofactor_12 = n[0, 1] * n[2, 0] - n[0, 0] * n[2, 1]
    cofactor_20 = n[0, 1] * n[1, 2] - n[0, 2] * n[1, 1]
    cofactor_21 = n[0, 2] * n[1, 0] - n[0, 0] * n[1, 2]
    cofactor_22 = n[0, 0] * n[1, 1] - n[0, 1] * n[1, 0]
    # determinant
    d = n[0, 0] * cofactor_00 + n[0, 1] * cofactor_01 + n[0, 2] * cofactor_02

    if abs(d) < 1e-12:
        raise ValueError('Inverse matrix does not exist!')

    inv = torch.Tensor([
        [cofactor_00, cofactor_10, cofactor_20],
        [cofactor_01, cofactor_11, cofactor_21],
        [cofactor_02, cofactor_12, cofactor_22],
    ])
    inv = inv.to(dtype=m.dtype, device=m.device)
    inv /= d
    return inv

def safe_region(
        pos_src: torch.Tensor,
        h: int,
        w: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:

    '''
    Check if the sampling point is valid from the given source coordinate.

    Args:

    Return:

    '''
    with torch.no_grad():
        pos_bound = pos_src.new_tensor([w - 0.5, h - 0.5])
        pos_bound.unsqueeze_(-1)
        pos_in = torch.logical_and(pos_src >= -0.5, pos_src < pos_bound)
        pos_in = pos_in.all(0)
        pos_src = pos_src[..., pos_in]
        yi, = pos_in.nonzero(as_tuple=True)

    return pos_src, yi

def calc_area(
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor) -> torch.Tensor:

    # To avoid multiplication of large numbers
    with torch.no_grad():
        shift = (x1 + x2 + x3 + x4) / 4
        x1 = x1 - shift
        x2 = x2 - shift
        x3 = x3 - shift
        x4 = x4 - shift
        cross1 = x1[0] * x2[1] - x2[0] * x1[1]
        cross2 = x2[0] * x3[1] - x3[0] * x2[1]
        cross3 = x3[0] * x4[1] - x4[0] * x3[1]
        cross4 = x4[0] * x1[1] - x1[0] * x4[1]
        area = (cross1 + cross2 + cross3 + cross4) / 2
        area = area.abs()

    return area

def calc_dsda_projective(
        sizes: typing.Tuple[int, int],
        m: typing.Union[torch.Tensor, typing.Callable],
        eps: float=0.1) -> torch.Tensor:

    
    if isinstance(m, torch.Tensor):
        grid_funcion = svf.projective_grid
    else:
        grid_funcion = functional.functional_grid

    x1 = grid_funcion(sizes, m, eps_y=-eps, eps_x=-eps)
    x2 = grid_funcion(sizes, m, eps_y=eps, eps_x=-eps)
    x3 = grid_funcion(sizes, m, eps_y=eps, eps_x=eps)
    x4 = grid_funcion(sizes, m, eps_y=-eps, eps_x=eps)

    ds = 4 * eps * eps
    da = calc_area(x1, x2, x3, x4)
    dsda = ds / (da + 1e-8)
    return dsda

def nearest_contribution(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    range_around_0 = torch.logical_and(x.gt(-eps), x.le(1 + eps))
    cont = range_around_0.to(dtype=x.dtype)
    return cont

def contribution_2d(
        x: torch.Tensor,
        kernel: typing.Union[str, nn.Module]='bicubic') -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor): (N, 2, k), where x[0] is the x-coordinate.
        kernel (str):

    Return
        torch.Tensor: (N, k^2)
    '''
    #print(x.size())
    if isinstance(kernel, nn.Module):
        weight = kernel(x)
    elif isinstance(kernel, str):
        if kernel == 'nearest':
            weight = nearest_contribution(x)
        elif kernel == 'bilinear':
            weight = core.linear_contribution(x)
        elif kernel == 'bicubic':
            weight = core.cubic_contribution(x)

        weight = torch.einsum('bi,bj->bij', (weight[:, 1], weight[:, 0]))
        weight = weight.view(weight.size(0), -1)

    weight = weight / weight.sum(-1, keepdim=True)
    return weight

def warp_general(
        x: torch.Tensor,
        m: typing.Optional[torch.Tensor],
        sizes: typing.Optional[typing.Tuple[int, int]],
        grid: typing.Optional[torch.Tensor]=None,
        yi: typing.Optional[torch.Tensor]=None,
        kernel: typing.Optional[str]='bicubic',
        padding_type: str='reflect',
        fill_value: float=0,
        warp_inverse: bool=False,
        is_half: bool=False) -> torch.Tensor:

    '''
    An actual algorithm for general warping.
    Do not access to this function directly.
    '''
    if isinstance(kernel, str):
        kernels = {'nearest': 1, 'bilinear': 2, 'bicubic': 4}
        if kernel in kernels:
            k = kernels[kernel]
        else:
            raise ValueError('kernel: {} is not supported!'.format(kernel))
    else:
        k = int(math.sqrt(kernel.size(-1)))

    pad = k // 2

    if grid is None:
        if not warp_inverse:
            m = inverse_3x3(m)

        grid = svf.projective_grid(sizes, m)
        grid, yi = safe_region(grid, x.size(-2), x.size(-1))

    # Discretize
    # (2, N)
    pos = grid + 1 - (k % 2) / 2
    pos_discrete = pos.floor()
    pos_frac = pos - pos_discrete
    pos_discrete = pos_discrete.long()

    if isinstance(kernel, str):
        # (N, 2, 1)
        pos_frac = pos_frac.t().view(-1, 2, 1)

        # (1, 2, k)
        pos_w = torch.linspace(
            pad - k + 1, pad, k, device=x.device, requires_grad=False,
        )
        pos_w = pos_w.view(1, 1, -1).repeat(1, 2, 1)
        # (N, 2, k)
        pos_w = pos_frac - pos_w
        weight = contribution_2d(pos_w, kernel=kernel)
    else:
        weight = kernel

    # (B, k^2, HW)
    x = core.padding(x, -2, pad, pad, padding_type=padding_type)
    x = core.padding(x, -1, pad, pad, padding_type=padding_type)
    x = x.contiguous()
    weight = weight.contiguous()
    # Calculate the exact sampling point
    xi = pos_discrete[0] + x.size(-1) * pos_discrete[1]
    # Spatially varying filtering
    y = svf.svf_forward(
        x,
        weight,
        sizes,
        k,
        xi,
        yi,
        fill_value,
        is_half,
    )
    return y

def warp(
        x: torch.Tensor,
        m: typing.Optional[torch.Tensor]=None,
        sizes: typing.Optional[typing.Tuple[int, int]]=None,
        grid: typing.Optional[torch.Tensor]=None,
        yi: typing.Optional[torch.Tensor]=None,
        kernel: typing.Union[str, torch.Tensor]='bicubic',
        padding_type: str='reflect',
        fill_value: float=0,
        warp_inverse: bool=False,
        is_half: bool=False) -> torch.Tensor:

    '''
    A wrapping function for general warping.
    '''
    #x, b, c, h, w = core.reshape_input(x)
    x, dtype = core.cast_input(x)

    if grid is None:
        if m is None:
            raise ValueError('Transformation matrix m should be specified!')
        else:
            m = m.to(device=x.device)

        if sizes is None:
            #sizes = (h, w)
            sizes = (x.size(-2), x.size(-1))
        elif not isinstance(sizes, tuple):
            raise ValueError('sizes:', sizes, 'is not supported!')
    else:
        if m is not None:
            raise ValueError('m and grid cannot be specified at the same time!')

        if yi is None:
            raise ValueError('yi should be specified with grid!')

        grid = grid.cuda()
        yi = yi.cuda()

    x = warp_general(
        x,
        m=m,
        sizes=sizes,
        grid=grid,
        yi=yi,
        kernel=kernel,
        padding_type=padding_type,
        fill_value=fill_value,
        warp_inverse=warp_inverse,
        is_half=is_half,
    )

    #x = core.reshape_output(x, b, c)
    x = core.cast_output(x, dtype)
    return x
