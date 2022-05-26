import math
import types
import typing

import numpy as np

from model.sr import edsr
from utils import svf
from utils import warp
from utils import random_transform
from misc.gpu_utils import parallel_forward

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleSampler(nn.Module):

    def __init__(
            self,
            n_pyramids: int=3,
            n_feats: int=32,
            m_feats: int=0,
            no_position: bool=False,
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
        if no_position:
            n_inputs = 1
            n_feats //= 2
        else:
            n_inputs = 5

        m = [
            nn.Linear(n_inputs + m_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_pyramids),
        ]
        self.n_pyramids = n_pyramids
        self.sampler = nn.Sequential(*m)
        self.no_position = no_position
        self.log_scale = log_scale
        return

    def forward(
            self,
            grid: torch.Tensor,
            yi: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int],
            feats: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        '''
        Args:
            grid (torch.Tensor): (2, N)
            yi (torch.Tensor):
            m (torch.Tensor): (3, 3)
            sizes (tuple):

        Return:
            torch.Tensor: (1, 1, sizes[0], sizes[1], n)
        '''
        with torch.no_grad():
            dsda = warp.calc_dsda_projective(sizes, m)
            dsda = dsda[yi]
            # (N, 1)
            dsda = dsda.unsqueeze(1)
            if self.log_scale:
                dsda = (dsda + 1e-8).log()

            if self.no_position:
                x = dsda
            else:
                grid = grid.t()
                grid_floor = grid.floor()
                grid_tl = grid - grid_floor
                grid_br = 1 - grid_tl
                # (N, 5)
                x = torch.cat([grid_tl, grid_br, dsda], dim=1)

        # Additional feature concatenation
        if feats is not None:
            x = torch.cat([x, feats], dim=1)

        logits = self.sampler(x)
        w = F.softmax(logits, dim=1)

        w_spatial = w.new_zeros(sizes[0] * sizes[1], self.n_pyramids)
        w_spatial[yi] = w
        w_spatial = w_spatial.view(1, 1, sizes[0], sizes[1], self.n_pyramids)
        return w_spatial


class SuperWarpMS(nn.Module):

    def __init__(
            self,
            max_scale: int=4,
            multi_scale: bool=False,
            multi_scale_sampling: bool=False,
            no_position: bool=False,
            log_scale: bool=False) -> None:

        super().__init__()
        self.x2 = edsr.EDSR(scale=2)
        self.x4 = edsr.EDSR(scale=4)
        self.multi_scale = multi_scale
        self.multi_scale_sampling = multi_scale_sampling
        if multi_scale_sampling:
            n_pyramids = int(math.log2(max_scale)) + 1
            self.mss = MultiScaleSampler(
                n_pyramids=n_pyramids,
                no_position=no_position,
                log_scale=log_scale,
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
            'no_position': cfg.no_position,
            'log_scale': cfg.log_scale,
        }
        return kwargs

    def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int],
            get_scalemap: bool=False) -> torch.Tensor:

        pyramids = [x, self.x2(x), self.x4(x)]
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
                    w = self.mss(grid, yi, ms_inverse, sizes)

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

