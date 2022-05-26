import typing
import numpy as np

from utils import svf
from utils import warp

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

def cubic_contribution(x: torch.Tensor, a: float=-0.5) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    return cont

def cubic_contribution2d(
        x: torch.Tensor,
        y: torch.Tensor,
        a: float=-0.5) -> torch.Tensor:

    #cont_x = cubic_contribution(x)
    #cont_y = cubic_contribution(y)
    cont = cubic_contribution(x) * cubic_contribution(y)
    return cont


class KernelEstimator(nn.Module):

    def __init__(self, kernel_size: int=7, regularize: bool=True) -> None:
        super().__init__()
        self.set_kernel_size(kernel_size)
        n_feats = 48
        self.net = nn.Sequential(
            nn.Linear(2 * kernel_size**2, n_feats),
            #nn.BatchNorm1d(n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            #nn.BatchNorm1d(n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            #nn.BatchNorm1d(n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, kernel_size**2),
        )
        self.regularize = regularize
        return

    def set_kernel_size(self, kernel_size: int) -> None:
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        return

    def forward(
            self,
            m_inverse: torch.Tensor,
            sizes: typing.Tuple[int, int],
            grid: torch.Tensor,
            yi: torch.Tensor,
            net: bool=False,
            raw: bool=False) -> torch.Tensor:

        with torch.no_grad():
            offset = (self.kernel_size % 2) / 2
            pos = grid + 1 - offset
            pos_discrete = pos.floor()
            pos_frac = pos - pos_discrete
            pos_discrete = pos_discrete.long()

            pl = svf.projective_grid(sizes, m_inverse, eps_x=-0.5)
            pr = svf.projective_grid(sizes, m_inverse, eps_x=0.5)
            pt = svf.projective_grid(sizes, m_inverse, eps_y=-0.5)
            pb = svf.projective_grid(sizes, m_inverse, eps_y=0.5)

            du = pb[:, yi] - pt[:, yi]
            dv = pr[:, yi] - pl[:, yi]

            # (2, N, 1, 1)
            du = du.view(2, -1, 1, 1)
            dv = dv.view(2, -1, 1, 1)

            len_du = torch.sqrt(du[0].pow(2) + du[1].pow(2))
            if self.regularize:
                min_du = 1
                mask_du = len_du < min_du
                du[:, mask_du] *= (min_du / len_du[mask_du])
                len_du[mask_du] = min_du

            len_dv = torch.sqrt(dv[0].pow(2) + dv[1].pow(2))
            if self.regularize:
                min_dv = 1
                mask_dv = len_dv < min_dv
                dv[:, mask_dv] *= (min_dv / len_dv[mask_dv])
                len_dv[mask_dv] = min_dv

            det = du[0] * dv[1] - du[1] * dv[0]
            len_du *= det
            len_dv *= det

            pos_frac.unsqueeze_(-1)
            pos_w = torch.linspace(
                self.padding - self.kernel_size + 1,
                self.padding,
                self.kernel_size,
                device=grid.device,
                requires_grad=False,
            )
            # (2, 1, k)
            pos_w += offset
            pos_w = pos_w.view(1, 1, -1)
            pos_w = pos_w.repeat(2, 1, 1)
            # (2, N, k)
            pos_w = pos_frac - pos_w

            # (N, 1, k)
            pos_wx = pos_w[0].view(-1, 1, self.kernel_size)
            # (N, k, 1)
            pos_wy = pos_w[1].view(-1, self.kernel_size, 1)
            # (x', y') coordinate
            pos_wxp = (dv[1] * pos_wx - du[1] * pos_wy) / len_du
            pos_wyp = (du[0] * pos_wy - dv[0] * pos_wx) / len_dv

            x = du[0] * pos_wxp + du[1] * pos_wyp
            y = dv[0] * pos_wxp + dv[1] * pos_wyp

        if net:
            with torch.no_grad():
                weight = torch.stack((x, y), dim=1)
                weight = weight.view(weight.size(0), -1)
                #weight = weight.abs()

            weight = self.net(weight)
            #weight = torch.softmax(weight, dim=-1)
            weight = weight / weight.sum(-1, keepdim=True)
        else:
            with torch.no_grad():
                weight = cubic_contribution2d(x, y)
                weight = weight.view(weight.size(0), -1)
                weight = weight / weight.sum(-1, keepdim=True)

        if raw:
            return weight, torch.stack((x, y), dim=1)
        else:
            return weight
