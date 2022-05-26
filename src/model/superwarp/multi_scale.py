import math
import types
import typing

import numpy as np

from utils import svf
from utils import warp
from utils import random_transform
from misc.gpu_utils import parallel_forward

import torch
from torch import nn
from torch.nn import functional as F


def default_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int=1,
        padding: int=None,
        bias: bool=True) -> nn.Conv2d:

    if padding is None:
        padding = (kernel_size - 1) // 2

    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    return conv


class ResBlock(nn.Sequential):
    '''
    Make a residual block which consists of Conv-(Norm)-Act-Conv-(Norm).

    Args:
        n_feats (int): Conv in/out_channels.
        kernel_size (int): Conv kernel_size.
        norm (<None> or 'batch' or 'layer'): Norm function.
        act (<'relu'> or 'lrelu' or 'prelu'): Activation function.
        res_scale (float, optional): Residual scaling.
        conv (funcion, optional): A function for making a conv layer.

    Note:
        Residual scaling:
        From Szegedy et al.,
        "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
        See https://arxiv.org/pdf/1602.07261.pdf for more detail.

        To modify stride, change the conv function.
    '''

    def __init__(self, n_feats: int, kernel_size: int) -> nn.Module:
        m = []
        for i in range(2):
            m.append(default_conv(n_feats, n_feats, kernel_size))
            if i == 0:
                m.append(nn.ReLU(inplace=True))

        super().__init__(*m)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + super().forward(x)
        return x


class Upsampler(nn.Sequential):
    '''
    Make an upsampling block using sub-pixel convolution
    
    Args:

    Note:
        From Shi et al.,
        "Real-Time Single Image and Video Super-Resolution
        Using an Efficient Sub-pixel Convolutional Neural Network"
        See https://arxiv.org/pdf/1609.05158.pdf for more detail
    '''

    def __init__(
            self,
            scale: int,
            n_feats: int,
            bias: bool=True) -> nn.Module:

        m = []
        log_scale = math.log(scale, 2)
        # check if the scale is power of 2
        if int(log_scale) == log_scale:
            for _ in range(int(log_scale)):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
        else:
            raise NotImplementedError

        super().__init__(*m)
        return


class MSEDSR(nn.Module):
    '''
    Multi-scale EDSR model
    '''

    def __init__(
            self,
            max_scale: int=4,
            depth: int=16,
            n_colors: int=3,
            n_feats: int=64) -> None:

        super().__init__()
        self.n_colors = n_colors

        self.conv_in = default_conv(n_colors, n_feats, 3)
        m = [ResBlock(n_feats, 3) for _ in range(depth)]
        m.append(default_conv(n_feats, n_feats, 3))
        self.resblocks = nn.Sequential(*m)
        self.up_img = Upsampler(2, n_colors, bias=False)
        self.up_feat = Upsampler(2, n_feats)
        self.conv_out = default_conv(n_feats, n_colors, 3)
        self.n_scales = int(math.log2(max_scale))
        return

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        output_list = [x]
        f = self.conv_in(x)
        for _ in range(self.n_scales):
            f = f + self.resblocks(f)
            f = self.up_feat(f)
            x = self.conv_out(f) + self.up_img(x)
            output_list.append(x)

        return output_list


class MultiScaleSampler(nn.Module):

    def __init__(
            self,
            n_pyramids: int=3,
            n_feats: int=32,
            log_scale: bool=False) -> None:

        super().__init__()
        '''
        Five input features:
            x - floor(x)
            y - floor(y)
            floor(x) + 1 - x
            floor(y) + 1 - y
            dsda
        '''
        m = [
            nn.Linear(5, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_pyramids),
        ]
        self.sampler = nn.Sequential(*m)
        self.log_scale = log_scale
        return

    def forward(
            self,
            grid: torch.Tensor,
            yi: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int],
            n: int) -> torch.Tensor:
        '''
        Args:
            grid (torch.Tensor): (2, N)
            yi (torch.Tensor):
            m (torch.Tensor): (3, 3)
            sizes (tuple):
            n (int):

        Return:
            torch.Tensor: (1, 1, sizes[0], sizes[1], n)
        '''
        with torch.no_grad():
            grid = grid.t()
            grid_floor = grid.floor()
            grid_tl = grid - grid_floor
            grid_br = 1 - grid_tl

            dsda = warp.calc_dsda_projective(sizes, m)
            dsda = dsda[yi]
            # (N, 1)
            dsda = dsda.unsqueeze(1)
            if self.log_scale:
                dsda = (dsda + 1e-8).log()

            # Put logarithm here?
            # (N, 5)
            x = torch.cat([grid_tl, grid_br, dsda], dim=1)

        logits = self.sampler(x)
        w = F.softmax(logits, dim=1)

        w_spatial = w.new_zeros(sizes[0] * sizes[1], n)
        w_spatial[yi] = w
        w_spatial = w_spatial.view(1, 1, sizes[0], sizes[1], n)
        return w_spatial


class SuperWarpMS(MSEDSR):

    def __init__(
            self,
            max_scale: int=4,
            multi_scale: bool=False,
            multi_scale_sampling: bool=False,
            log_scale: bool=False) -> None:

        super().__init__(max_scale=max_scale)

        self.multi_scale = multi_scale
        self.multi_scale_sampling = multi_scale_sampling
        if multi_scale_sampling:
            n_pyramids = int(math.log2(max_scale)) + 1
            self.mss = MultiScaleSampler(
                n_pyramids=n_pyramids, log_scale=log_scale,
            )

        self.kernel = 'bicubic'
        self.fill_value = -255
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = {
            'max_scale': cfg.max_scale,
            'multi_scale': cfg.ms,
            'multi_scale_sampling': cfg.mss,
            'log_scale': cfg.log_scale,
        }
        return kwargs

    def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int],
            get_scalemap: bool=False) -> torch.Tensor:

        pyramids = super().forward(x)
        n = len(pyramids)
        if not self.multi_scale:
            pyramids = (pyramids[-1],)

        ys = []
        for pi, p in enumerate(pyramids):
            s = x.size(-1) / p.size(-1)
            # Scale-specific matrix
            with torch.no_grad():
                ms = random_transform.scaling_transform(s)
                ms = torch.matmul(m, ms)
                ms_inverse = warp.inverse_3x3(ms).cuda()

            grid = svf.projective_grid(sizes, ms_inverse)
            grid, yi = warp.safe_region(grid, p.size(-2), p.size(-1))
            y = warp.warp(
                p,
                sizes=sizes,
                grid=grid,
                yi=yi,
                kernel=self.kernel,
                fill_value=self.fill_value,
            )
            ys.append(y)

            if pi == 0:
                # To compensate floating point error from averaging
                mask = (y != self.fill_value).float()
                if self.multi_scale_sampling:
                    # Calculate the sampling weight
                    w = self.mss(grid, yi, ms_inverse, sizes, n)

        if self.multi_scale_sampling:
            ys = [w[..., yi] * y for yi, y in enumerate(ys)]
            out = sum(ys)
        else:
            out = sum(ys) / n

        out = mask * out + (1 - mask) * self.fill_value
        # Just for debugging
        if get_scalemap:
            return out, w
        else:
            return out


REPRESENTATIVE = SuperWarpMS

