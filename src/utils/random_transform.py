import math
import random
import typing

from utils import warp

import torch
from torch.nn import functional as F

_I = typing.Tuple[int, int]

def generate_transform(
        x: torch.Tensor,
        scale_min: float=2,
        scale_max: float=4,
        sheer_max: float=0.25,
        rotation_max: float=15,
        vp_min: float=-0.5,
        vp_max: float=0.5) -> torch.Tensor:

    m = torch.eye(3).double()

    # Sheer matrix
    sheer_x = random.uniform(-sheer_max, sheer_max)
    sheer_y = random.uniform(-sheer_max, sheer_max)

    matrix_sheer = torch.eye(3).double()
    matrix_sheer[0, 1] = sheer_x
    matrix_sheer[1, 0] = sheer_y
    m = torch.matmul(m, matrix_sheer)

    # Rotation matrix
    rotation_max = max(rotation_max, 45)
    theta = random.gauss(0, rotation_max / 3)
    theta = max(min(theta, rotation_max), -rotation_max)
    theta = 2 * math.pi * (theta / 360)

    matrix_rotation = torch.eye(3).double()
    matrix_rotation[0, 0] = math.cos(theta)
    matrix_rotation[0, 1] = math.sin(theta)
    matrix_rotation[1, 0] = -math.sin(theta)
    matrix_rotation[1, 1] = math.cos(theta)
    m = torch.matmul(m, matrix_rotation)

    # Scaling matrix
    scale_x = random.uniform(scale_min, scale_max)
    scale_y = random.uniform(scale_min, scale_max)
    matrix_scale = torch.eye(3).double()
    matrix_scale[0, 0] = scale_x
    matrix_scale[1, 1] = scale_y
    m = torch.matmul(m, matrix_scale)

    # Projection matrix
    # Appropriate parameters may vary depend on input resolution
    _, _, h, w = x.size()
    vp_x = random.uniform(vp_min / w, vp_max / w)
    vp_y = random.uniform(vp_min / h, vp_max / h)
    shift_x = random.uniform(-0.75 * w, 0.125 * w)
    shift_y = random.uniform(-0.75 * h, 0.125 * h)
    matrix_projection = torch.eye(3).double()
    matrix_projection[0, 2] = shift_x
    matrix_projection[1, 2] = shift_y
    matrix_projection[2, 0] = vp_x
    matrix_projection[2, 1] = vp_y
    m = torch.matmul(m, matrix_projection)

    return m

def scaling_transform(
        sx: float,
        sy: typing.Optional[float]=None) -> torch.Tensor:

    if sy is None:
        sy = sx

    shift_x = 0.5 * (sx - 1)
    shift_y = 0.5 * (sy - 1)
    m = torch.DoubleTensor([
        [sx, 0, shift_x],
        [0, sy, shift_y],
        [0, 0, 1],
    ])
    return m

def transform_corners(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.size()
    # For higher accuracy
    m = m.double()
    corners = m.new_tensor([
        [-0.5, -0.5, w - 0.5, w - 0.5],
        [-0.5, h - 0.5, -0.5, h - 0.5],
        [1, 1, 1, 1],
    ])
    corners = torch.matmul(m, corners)
    corners = corners / corners[-1, :]
    return corners

def get_box(corners: torch.Tensor) -> typing.Tuple[float]:
    y_min = corners[1].min().item()
    x_min = corners[0].min().item()
    h_new = corners[1].max().item() - y_min
    w_new = corners[0].max().item() - x_min
    h_new = math.ceil(math.trunc(1000 * h_new) / 1000)
    w_new = math.ceil(math.trunc(1000 * w_new) / 1000)
    y_min += 0.5
    x_min += 0.5
    return y_min, x_min, h_new, w_new

def compensate_integer(
        x: torch.Tensor,
        m: torch.Tensor) -> typing.Tuple[torch.Tensor, _I, _I]:

    corners = transform_corners(x, m)
    y_min, x_min, h_new, w_new = get_box(corners)
    '''
    print('{:.8f} {:.8f} {:.8f} {:.8f}'.format(
        corners[1].min().item(),
        corners[0].min().item(),
        corners[1].max().item(),
        corners[0].max().item(),
    ))
    '''
    y_shift = max(math.floor(y_min), 0)
    x_shift = max(math.floor(x_min), 0)
    m_comp = m.new_tensor([[1, 0, -x_shift], [0, 1, -y_shift], [0, 0, 1]])
    m = torch.matmul(m_comp, m)
    return m, (h_new + 1, w_new + 1), (y_shift, x_shift)

def compensate(
        x: torch.Tensor,
        m: torch.Tensor,
        orientation: bool=False) -> typing.Tuple[torch.Tensor, _I]:

    '''
    Args:
        x (torch.Tensor): (B, C, H, W)
        m (torch.Tensor): (3, 3)
        force_integer (bool, optional): Force displacements to be integers.
        orientation (bool, optional): Also fix orientation.
    '''
    corners = transform_corners(x, m)
    if orientation:
        matrix_translation = torch.eye(3)
        matrix_translation[0, 2] = corners[0, 0].item()
        matrix_translation[1, 2] = corners[1, 0].item()
        m = torch.matmul(matrix_translation, m)

        theta = -math.atan2(
            corners[0, 2].item() - corners[0, 0].item(),
            corners[1, 2].item() - corners[1, 0].item(),
        )
        theta += math.pi / 2
        matrix_rotation = torch.eye(3)
        matrix_rotation[0, 0] = math.cos(theta)
        matrix_rotation[0, 1] = math.sin(theta)
        matrix_rotation[1, 0] = -math.sin(theta)
        matrix_rotation[1, 1] = math.cos(theta)
        m = torch.matmul(matrix_rotation, m)
        return compensate(x, m, orientation=False)
    else:
        y_min, x_min, h_new, w_new = get_box(corners)
        m_comp = m.new_tensor([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        m = torch.matmul(m_comp, m)
        return m, (h_new, w_new)

def get_transform(
        x: torch.Tensor,
        scale_limit: typing.Optional[float]=None,
        size_limit: typing.Optional[int]=2000,
        orientation: bool=False,
        **kwargs) -> typing.Tuple[torch.Tensor, _I]:

    '''
    We want to avoid "too large" images...
    '''
    _, _, h, w = x.size()
    if scale_limit is not None:
        size_h_limit = h * scale_limit
        size_w_limit = w * scale_limit
    else:
        size_h_limit = size_limit
        size_w_limit = size_limit

    while True:
        m = generate_transform(x, **kwargs)
        m, sizes = compensate(x, m, orientation=orientation)
        if sizes[0] <= size_h_limit and sizes[1] < size_w_limit:
            break

    return m, sizes

def crop(
        x: torch.Tensor,
        ignore: torch.Tensor,
        patch_size: int) -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor):       (B, C, H, W) input.
        ignore (torch.Tensor):  (1, 1, H, W) float mask.
                                1 denotes a pixel to be ignored.
        patch_size (int):       The patch size.
    '''
    offset_x = random.randrange(0, patch_size)
    offset_y = random.randrange(0, patch_size)
    x_sub = x[..., offset_y:, offset_x:]
    ignore_sub = ignore[..., offset_y:, offset_x:]
    ignore_sub = F.max_pool2d(ignore_sub, patch_size)
    ignore_sub.squeeze_()

    samples = (ignore_sub == 0)
    samples = samples.nonzero(as_tuple=False)
    iy, ix = random.choice(samples)
    iy *= patch_size
    ix *= patch_size
    patch = x_sub[..., iy:(iy + patch_size), ix:(ix + patch_size)]
    return patch, offset_y + iy, offset_x + ix


def crop_largest(
        x: torch.Tensor,
        ignore: torch.Tensor,
        pool_size: int=16,
        patch_max: typing.Optional[int]=None,
        margin: int=0,
        stochastic: bool=True) -> torch.Tensor:

    if x.dim() != 4:
        raise ValueError(
            'Only accept 4D batched input! Got m.dim() = {}'.format(x.dim())
        )

    if patch_max is not None:
        if patch_max % pool_size != 0:
            raise ValueError(
                'patch_max ({}) should be divided by pool_size ({})!'.format(
                    patch_max, pool_size,
                )
            )

    with torch.no_grad():
        ignore_sub = F.max_pool2d(ignore, pool_size)
        ignore_sub.squeeze_()

        t = 1 - ignore_sub.long()
        max_size = 0
        max_pos_x = None
        max_pos_y = None

        for i in range(ignore_sub.size(0)):
            for j in range(ignore_sub.size(1)):
                if i > 0 and j > 0 and ignore_sub[i, j] == 0:
                    t[i, j] = min(t[i, j - 1], t[i - 1, j], t[i - 1, j - 1]) + 1

                if max_size < t[i, j].item():
                    max_size = t[i, j].item()
                    max_pos_y = i - max_size + 1
                    max_pos_x = j - max_size + 1

        torch.set_printoptions(linewidth=200)

    patch_size = pool_size * max_size
    iy = pool_size * max_pos_y
    ix = pool_size * max_pos_x
    if patch_max is not None:
        if patch_size > patch_max:
            if stochastic:
                iy += random.randrange(0, patch_size - patch_max + 1)
                ix += random.randrange(0, patch_size - patch_max + 1)
            else:
                di = (patch_size - patch_max) // 2
                iy += di
                ix += di

            patch_size = patch_max

    if margin > 0:
        iy += margin
        ix += margin
        patch_size -= 2 * margin

    patch = x[..., iy:(iy + patch_size), ix:(ix + patch_size)]
    return patch, iy, ix

def compensate_offset(m: torch.Tensor, iy: int, ix: int) -> torch.Tensor:
    m_translation = m.new_tensor([[1, 0, ix], [0, 1, iy], [0, 0, 1]])
    m = torch.matmul(m, m_translation)
    return m

