import math
import types
import typing

import numpy as np

from model.sr import edsr
from model.superwarp import dual
from model.superwarp import masked_conv
from utils import svf
from utils import warp
from utils import random_transform
from misc.gpu_utils import parallel_forward

import torch
from torch import nn
from torch.nn import functional as F


class MaskedSequential(nn.Module):

    def __init__(
            self,
            n_inputs: int=3,
            n_feats: int=64,
            n_outputs: int=3,
            depth: int=4) -> None:

        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2

        self.ms = nn.ModuleList()
        self.ms.append(masked_conv.MaskedConv(
            n_inputs,
            n_feats,
            kernel_size,
            padding=padding,
        ))
        self.ms.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            self.ms.append(masked_conv.MaskedConv(
                n_feats,
                n_feats,
                kernel_size,
                padding=padding,
            ))
            self.ms.append(nn.ReLU(inplace=True))

        self.ms.append(masked_conv.MaskedConv(
            n_feats,
            n_outputs,
            kernel_size,
            padding=padding,
        ))
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = mask * x
        for m in self.ms:
            if isinstance(m, nn.Conv2d):
                x = m(x, mask)
            else:
                x = m(x)

        return x


class MaskedRecon(nn.Module):

    def __init__(
            self,
            n_inputs: int=3,
            n_feats: int=64,
            n_outputs: int=3,
            depth: int=4) -> None:

        super().__init__()
        self.recon = MaskedSequential(
            n_inputs=n_inputs,
            n_feats=n_feats,
            n_outputs=n_outputs,
            depth=depth,
        )
        self.residual = (n_inputs == n_outputs)
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Zero-padding for outside regions
        if self.residual:
            x = x + self.recon(x, mask)
        else:
            x = self.recon(x, mask)

        return x


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
            self.mss = dual.MultiScaleSampler(
                n_pyramids=n_pyramids,
                no_position=no_position,
                log_scale=log_scale,
            )

        self.recon = MaskedRecon()
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
                mask = mask[:1, :1]
                if self.multi_scale_sampling:
                    # Calculate the sampling weight
                    w = self.mss(grid, yi, ms_inverse, sizes)

        if self.multi_scale_sampling:
            ys = [w[..., yi] * y for yi, y in enumerate(ys)]
            out = sum(ys)
        else:
            out = sum(ys) / n

        out = self.recon(out, mask)
        out = mask * out + (1 - mask) * self.fill_value
        # Just for debugging
        if get_scalemap:
            return out, w
        else:
            return out


REPRESENTATIVE = SuperWarpMS

