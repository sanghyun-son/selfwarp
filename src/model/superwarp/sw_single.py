import math
import types
import typing

import numpy as np

from model import common
from model.superwarp import kernel
from model.sr import edsr_f
from model.sr import mdsr_f
from model.sr import rrdb_f
from model.sr import mrdb_f
from model.sr import mrdb_fps
from model.sr import rdn_f
from utils import svf
from utils import warp
from utils import random_transform

import torch
from torch import cuda
from torch import nn
from torch.nn import functional as F


class MaskedConv(nn.Conv2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, bias=False, **kwargs)
        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.mask_conv.weight.data.fill_(1 / np.prod(self.kernel_size))
        self.mask_conv.requires_grad_(False)
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): (B, C, H, W)
            mask (torch.Tensor): (1, 1, H, W)

        Return:
        '''
        with torch.no_grad():
            w = self.mask_conv(mask)
            w.clamp_(min=1e-8)
            w.reciprocal_()
            w *= mask

        x = super().forward(x)
        if self.training:
            x = w * x
        else:
            x *= w

        return x


class MaskedResBlock(nn.Module):

    def __init__(self, n_feats: int=64, bn: bool=False) -> None:
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.conv1 = MaskedConv(
            n_feats,
            n_feats,
            kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv(
            n_feats,
            n_feats,
            kernel_size,
            padding=padding,
        )
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = mask * x
        else:
            x *= mask

        res = self.conv1(x, mask)
        res = self.relu(res)
        res = self.conv2(res, mask)
        x = x + res
        return x


class MaskedResSeq(nn.Module):

    def __init__(
            self,
            n_inputs: int=64,
            n_feats: int=64,
            n_outputs: int=3,
            depth: int=4) -> None:

        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2

        self.ms = nn.ModuleList()
        if n_inputs == n_feats:
            self.ms.append(MaskedResBlock(n_feats))
            depth -= 2
        else:
            self.ms.append(MaskedConv(
                n_inputs, n_feats, kernel_size, padding=padding,
            ))
            depth -= 1

        while depth > 0:
            self.ms.append(MaskedResBlock(n_feats))
            depth -= 2

        if n_outputs != n_feats:
            self.ms.append(MaskedConv(
                n_feats, n_outputs, kernel_size, padding=padding,
            ))

        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = mask * x
        else:
            x *= mask

        for m in self.ms:
            x = m(x, mask)

        return x


class MaskedResRecon(nn.Module):

    def __init__(
            self,
            n_inputs: int=64,
            n_feats: int=64,
            n_outputs: int=3,
            depth: int=4) -> None:

        super().__init__()
        self.body = MaskedResSeq(
            n_inputs=n_feats,
            n_feats=n_feats,
            n_outputs=n_feats,
            depth=depth,
        )
        self.recon = MaskedConv(n_feats, n_outputs, 3, padding=1)
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.body(x, mask)
        x = self.recon(x, mask)
        return x


class SuperWarpF(nn.Module):

    def __init__(
            self,
            n_colors: int=3,
            max_scale: int=4,
            backbone: str='edsr',
            residual: bool=False,
            normal_upsample: str='bicubic',
            elliptical_upsample: bool=False,
            kernel_net: bool=False,
            kernel_net_multi: bool=False,
            kernel_net_size: int=4,
            kernel_regularize: bool=True,
            kernel_depthwise: bool=False,
            kernel_bilinear: bool=False,
            kernel_no_mul: bool=False,
            ms_blend: str='net',
            **kwargs) -> None:

        super().__init__()
        # x1 network construction
        n_feats = 64
        # Use pre-trained models for x2 and higher
        kwargs = {'n_feats': n_feats, 'multi_scale': False}
        if backbone == 'mdsr':
            self.backbone_multi = mdsr_f.MDSRF(**kwargs)
        elif backbone == 'mrdb-ps':
            self.backbone_multi = mrdb_fps.MRDBFPS(**kwargs)

        self.recon = MaskedResRecon(
            n_inputs=n_feats,
            n_feats=n_feats,
            n_outputs=n_colors,
            depth=10,
        )

        self.residual = residual
        self.normal_upsample = normal_upsample
        self.elliptical_upsample = elliptical_upsample
        self.kernel_net = kernel_net
        self.ms_blend = ms_blend
        if elliptical_upsample or kernel_net:
            if kernel_depthwise:
                depthwise = n_feats
            else:
                depthwise = 1
                
            kwargs = {
                'kernel_size': kernel_net_size,
                'regularize': kernel_regularize,
                'dw': depthwise,
                'bilinear': kernel_bilinear,
                'no_mul': kernel_no_mul,
            }
            self.kernel_estimator = kernel.KernelEstimator(**kwargs)
        else:
            self.kernel_estimator = None

        self.fill_value = -255
        self.pre_built = None
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = {
            'n_colors': cfg.n_colors,
            'backbone': cfg.backbone,
            'residual': cfg.residual,
            'normal_upsample': cfg.normal_upsample,
            'elliptical_upsample': cfg.elliptical_upsample,
            'kernel_net': cfg.kernel_net,
            'kernel_net_multi': cfg.kernel_net_multi,
            'kernel_net_size': cfg.kernel_net_size,
            'kernel_regularize': not cfg.kernel_noreg,
            'kernel_depthwise': cfg.kernel_depthwise,
            'kernel_bilinear': cfg.kernel_bilinear,
            'kernel_no_mul': cfg.kernel_no_mul,
            'ms_blend': cfg.ms_blend,
        }
        return kwargs

    @torch.no_grad()
    def build(self, x: torch.Tensor) -> None:
        self.pre_built = self.backbone_multi(x)
        cuda.empty_cache()
        return

    def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Optional[typing.Tuple[int, int]]=None,
            get_scalemap: bool=False,
            empty_cache: bool=False) -> torch.Tensor:

        m = m.cpu()
        if sizes is None:
            sizes = (256, 256)

        # (B, C, h, w) only
        pyramids = [self.backbone_multi(x)]

        ys = []
        ref = {}
        for pi, lambda_p in enumerate(pyramids):
            if self.pre_built is not None:
                p = self.pre_built[pi]
            else:
                if isinstance(lambda_p, nn.Module):
                    p = lambda_p(x)
                else:
                    p = lambda_p

            s = float(x.size(-1)) / float(p.size(-1))
            # Scale-specific matrix
            with torch.no_grad():
                ms = random_transform.scaling_transform(s)
                ms = torch.matmul(m, ms)
                ms, sizes_c, shift = random_transform.compensate_integer(p, ms)
                ms_inverse = warp.inverse_3x3(ms).cuda()
                # Backup these values to avoid numerical instability
                if pi == 0:
                    ref['sizes'] = sizes_c
                    ref['shift'] = shift

            grid = svf.projective_grid(ref['sizes'], ms_inverse)
            grid, yi = warp.safe_region(grid, p.size(-2), p.size(-1))
            if self.kernel_estimator is not None:
                net = self.kernel_estimator
                k = net(
                    ms_inverse,
                    ref['sizes'],
                    grid,
                    yi,
                    net=(not self.elliptical_upsample),
                )
            else:
                k = self.normal_upsample

            y = warp.warp(
                p,
                sizes=ref['sizes'],
                grid=grid,
                yi=yi,
                kernel=k,
                fill_value=self.fill_value,
            )
            ys.append(y)

            if pi == 0:
                mask = (y[:1, :1] != self.fill_value).float()
                ref['grid'] = grid
                ref['yi'] = yi
                ref['m'] = ms_inverse
                if self.residual:
                    with torch.no_grad():
                        ref['res'] = warp.warp(
                            x,
                            sizes=ref['sizes'],
                            grid=grid,
                            yi=yi,
                            kernel='bicubic',
                            fill_value=self.fill_value,
                        )

            if not self.training and empty_cache:
                cuda.empty_cache()

        out = self.recon(ys[0], mask)
        if 'res' in ref:
            out = out + ref['res']

        out = mask * out + (1 - mask) * self.fill_value
        out_full = out.new_full(
            (out.size(0), out.size(1), sizes[0], sizes[1]), self.fill_value,
        )
        ref_h = ref['sizes'][0]
        ref_w = ref['sizes'][1]
        dy = ref['shift'][0]
        dx = ref['shift'][1]
        ref_h = min(ref_h, sizes[0] - dy)
        ref_w = min(ref_w, sizes[1] - dx)
        slice_h = slice(dy, dy + ref_h)
        slice_w = slice(dx, dx + ref_w)
        out_full[..., slice_h, slice_w] = out[..., :ref_h, :ref_w]
        return out_full


REPRESENTATIVE = SuperWarpF

