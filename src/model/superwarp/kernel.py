import typing
import numpy as np

from utils import svf
from utils import warp
from utils import functional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

_TTT = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_TTTT = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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


class Bilinear(nn.Module):

    def __init__(self, kernel_size: int, n_feats: int) -> None:
        super().__init__()
        self.f_pos = nn.Sequential(
            nn.Linear(2 * kernel_size * kernel_size, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
        )
        self.f_m = nn.Sequential(
            nn.Linear(4, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
        )
        self.pos_m = nn.Bilinear(n_feats, n_feats, n_feats)
        self.regressor = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, 2 * kernel_size * kernel_size),
        )
        return

    def forward(self, pos: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        f_pos = self.f_pos(pos)
        f_m = self.f_m(m)
        pos_m = self.pos_m(f_pos, f_m)
        estimated_kernel = self.regressor(pos_m)
        return estimated_kernel


class KernelEstimator(nn.Module):

    def __init__(
            self,
            dw: int=1,
            kernel_size: int=4,
            n_feats: int=48,
            net_regular: bool=False,
            regularize: bool=True,
            bilinear: bool=False,
            no_mul: bool=False,
            legacy: bool=True) -> None:

        super().__init__()
        self.dw = dw
        self.legacy = legacy
        self.set_kernel_size(kernel_size)
        if bilinear:
            self.net = Bilinear(kernel_size, n_feats)
        else:
            if dw == 1:
                if no_mul:
                    last_layer = nn.Linear(n_feats, kernel_size**2)
                else:
                    last_layer = nn.Linear(n_feats, 2 * kernel_size**2)

                self.net = nn.Sequential(
                    nn.Linear(2 * kernel_size**2, n_feats),
                    nn.ReLU(inplace=True),
                    nn.Linear(n_feats, n_feats),
                    nn.ReLU(inplace=True),
                    last_layer,
                )
            else:
                if no_mul:
                    last_layer = nn.Linear(2 * n_feats, dw * kernel_size**2)
                else:
                    last_layer = nn.Linear(2 * n_feats, 2 * dw * kernel_size**2)

                self.net = nn.Sequential(
                    nn.Linear(2 * kernel_size**2, n_feats),
                    nn.ReLU(inplace=True),
                    nn.Linear(n_feats, 2 * n_feats),
                    nn.ReLU(inplace=True),
                    last_layer,
                )

        self.regularize = regularize
        self.net_regular = net_regular
        self.bilinear = bilinear
        self.no_mul = no_mul
        return

    def set_kernel_size(self, kernel_size: int) -> None:
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        return

    @torch.no_grad()
    def get_transform(
            self,
            m: typing.Union[torch.Tensor, typing.Callable],
            sizes: typing.Tuple[int, int],
            yi: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(m, torch.Tensor):
            grid_function = svf.projective_grid
        else:
            grid_function = functional.functional_grid

        pl = grid_function(sizes, m, eps_x=-0.5)
        pr = grid_function(sizes, m, eps_x=0.5)
        pt = grid_function(sizes, m, eps_y=-0.5)
        pb = grid_function(sizes, m, eps_y=0.5)

        dx = pr[:, yi] - pl[:, yi]
        dy = pb[:, yi] - pt[:, yi]

        dx = dx.view(2, -1, 1, 1)
        dy = dy.view(2, -1, 1, 1)
        return dx, dy

    def get_modulator(
            self,
            dx: torch.Tensor,
            dy: torch.Tensor) -> typing.Union[_TTT, _TTTT]:

        # (N, 1, 1)
        # dudx: a, dvdx: b, dudy: c, dvdy: d
        det = dx[0] * dy[1] - dx[1] * dy[0]
        det.pow_(2)

        aa = dx[1].pow(2) + dy[1].pow(2)
        bb = dx[0].pow(2) + dy[0].pow(2)
        cc = 2 * (dx[0] * dx[1] + dy[0] * dy[1])
        aa.div_(det)
        bb.div_(det)
        cc.div_(det)
        theta = 0.5 * torch.atan2(cc, bb - aa)
        cos = theta.cos()
        sin = theta.sin()
        cos2 = cos.pow(2)
        num = 2 * cos2 - 1
        den_share = (aa + bb) * cos2
        p = num / (den_share - bb)
        q = num / (den_share - aa)
        if self.regularize:
            p.clamp_(min=1)
            q.clamp_(min=1)
        else:
            p.clamp_(min=0.1)
            q.clamp_(min=0.1)

        p.rsqrt_()
        q.rsqrt_()
        if self.legacy:
            diff = p - q
            calb_share = diff * cos2
            calb_x = q + calb_share
            calb_y = p - calb_share
            calb_diag = cos * sin * diff
            return calb_x, calb_y, calb_diag
        else:
            calb_xx = p * cos
            calb_xy = p * sin
            calb_yx = -q * sin
            calb_yy = q * cos
            return calb_xx, calb_xy, calb_yx, calb_yy

    def forward(
            self,
            m_inverse: typing.Union[torch.Tensor, typing.Callable],
            sizes: typing.Tuple[int, int],
            grid: torch.Tensor,
            yi: torch.Tensor,
            net: bool=False,
            debug: bool=False) -> torch.Tensor:

        with torch.no_grad():
            offset = (self.kernel_size % 2) / 2
            pos = grid + 1 - offset
            pos_discrete = pos.floor()
            pos_frac = pos - pos_discrete

            dx, dy = self.get_transform(m_inverse, sizes, yi)
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
            pos_frac.unsqueeze_(-1)
            # (2, N, k)
            pos_w = pos_w - pos_frac

            # (N, 1, k)
            pos_wx = pos_w[0].view(-1, 1, self.kernel_size)
            # (N, k, 1)
            pos_wy = pos_w[1].view(-1, self.kernel_size, 1)
            '''
            if self.bilinear:
                m = torch.cat((dx, dy), dim=0)
                m.squeeze_()
                m.t_()
                x = pos_wx.repeat(1, self.kernel_size, 1)
                y = pos_wy.repeat(1, 1, self.kernel_size)
            '''
            if self.net_regular:
                x = pos_wx.repeat(1, self.kernel_size, 1)
                y = pos_wy.repeat(1, 1, self.kernel_size)
            else:
                if self.legacy:
                    calb_x, calb_y, calb_diag = self.get_modulator(dx, dy)
                    x = calb_x * pos_wx + calb_diag * pos_wy
                    y = calb_y * pos_wy + calb_diag * pos_wx
                else:
                    calb_xx, calb_xy, calb_yx, calb_yy = self.get_modulator(dx, dy)
                    x = calb_xx * pos_wx + calb_xy * pos_wy
                    y = calb_yx * pos_wx + calb_yy * pos_wy

        if net:
            with torch.no_grad():
                # (N, 2, k^2)
                weight = torch.stack((x, y), dim=1)
                # (N, 2 * k^2)
                weight = weight.view(weight.size(0), -1)
                '''
                if self.dw == 1:
                    weight = weight.view(weight.size(0), -1)
                else:
                    weight = weight.view(
                        weight.size(0), 2, self.kernel_size, self.kernel_size,
                    )
                '''

            if self.bilinear:
                weight_xy = self.net(weight, m)
            else:
                weight_xy = self.net(weight)

            if self.no_mul:
                weight = weight_xy
            else:
                weight_x, weight_y = weight_xy.chunk(2, dim=1)
                weight = weight_x * weight_y

            if self.dw > 1:
                weight = weight.view(weight.size(0), self.dw, -1)

            #weight = torch.softmax(weight, dim=-1)
            #weight = weight / weight.sum(-1, keepdim=True)
        else:
            with torch.no_grad():
                weight = cubic_contribution2d(x, y)
                weight = weight.view(weight.size(0), -1)
                weight = weight / weight.sum(-1, keepdim=True)

        if debug:
            #return weight, torch.stack((x, y), dim=1)
            return weight, dx, dy
        else:
            return weight
