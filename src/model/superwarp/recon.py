import math
import types
import typing

from model.superwarp import multi_scale
from utils import warp
from utils import svf
from utils import random_transform
from misc.gpu_utils import parallel_forward

import torch
from torch import nn


class SuperWarpMS(multi_scale.SuperWarpMS):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int]) -> torch.Tensor:

        ys = []
        pyramid = super().forward(x)
        if not self.multi_scale:
            pyramid = (pyramid[-1],)

        for p in pyramid:
            s = x.size(-1) / p.size(-1)
            # Scale-specific matrix
            ms = torch.Tensor([
                [s, 0, 0.5 * (s - 1)],
                [0, s, 0.5 * (s - 1)],
                [0, 0, 1],
            ])
            ms.requires_grad = False
            ms = torch.matmul(m, ms)

            grid = svf.projective_grid(sizes, m)
            grid, yi = warp.safe_region(grid, x.size(-2), x.size(-1))
            y = warp.warp(
                p,
                ms,
                grid=grid,
                yi=yi,
                kernel=self.kernel,
                fill_value=self.fill_value,
            )
            ys.append(y)

        y = sum(ys) / len(pyramid)
        return y


REPRESENTATIVE = SuperWarpMS

