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
from model.sr import mrdn_f
from utils import svf
from utils import warp
from utils import random_transform

import torch
from torch import cuda
from torch import nn
from torch.nn import functional as F


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
            **kwargs) -> None:

        super().__init__()
        # x1 network construction
        n_feats = 64
        # Use pre-trained models for x2 and higher
        self.backbone_multi = None
        if backbone == 'edsr':
            self.x2 = edsr_f.EDSRF(scale=2, n_feats=n_feats)
            self.x4 = edsr_f.EDSRF(scale=4, n_feats=n_feats)
        elif backbone == 'mdsr':
            self.backbone_multi = mdsr_f.MDSRF(n_feats=n_feats)
        elif backbone == 'rrdb':
            self.x2 = rrdb_f.RRDBF(scale=2, n_feats=n_feats)
            self.x4 = rrdb_f.RRDBF(scale=4, n_feats=n_feats)
        elif backbone == 'mrdb':
            self.backbone_multi = mrdb_f.MRDBF(n_feats=n_feats)
        elif backbone == 'mrdb-ps':
            self.backbone_multi = mrdb_fps.MRDBFPS(n_feats=n_feats)
        elif backbone == 'rdn':
            self.x2 = rdn_f.RDNF(scale=2)
            self.x4 = rdn_f.RDNF(scale=4)
        elif backbone == 'mrdn':
            self.backbone_multi = mrdn_f.MRDNF()

        self.recon = nn.Conv2d(n_feats, 3, 3, padding=1)

        self.residual = residual
        self.normal_upsample = normal_upsample
        self.elliptical_upsample = elliptical_upsample
        self.kernel_net = kernel_net
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
            #'max_scale': cfg.max_scale,
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
        }
        return kwargs

    def x1_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x + self.x1(x)
        return x

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

        # (B, C, h, w), (B, C, 2 x h, 2 x w), (B, C, 4 x h, 4 x w)
        pyramids = self.backbone_multi(x)
        pyramids = [self.recon(pyramids[0])]
        if not self.training and empty_cache:
            cuda.empty_cache()

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
                if isinstance(self.kernel_estimator, nn.ModuleList):
                    net = self.kernel_estimator[pi]
                else:
                    net = self.kernel_estimator

                k = net(
                    ms_inverse,
                    ref['sizes'],
                    grid,
                    yi,
                    net=(not self.elliptical_upsample),
                )
                if not self.training and empty_cache:
                    cuda.empty_cache()
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

        out = ys[0]
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

