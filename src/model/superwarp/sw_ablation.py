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


class MSResFeature(nn.Module):

    def __init__(
            self,
            n_pyramids: int=3,
            n_feats: int=64,
            depth: int=6) -> None:

        super().__init__()
        n_outputs = 16
        self.scale_specific = nn.ModuleList()
        for _ in range(n_pyramids):
            self.scale_specific.append(MaskedResSeq(
                n_inputs=n_feats,
                n_feats=n_feats,
                n_outputs=n_outputs,
                depth=depth // 2,
            ))

        n_feats_ex = n_pyramids * n_outputs
        self.feature = MaskedResSeq(
            n_inputs=n_feats_ex,
            n_feats=n_feats_ex,
            n_outputs=n_feats_ex,
            depth=(depth - depth // 2),
        )
        self.n_pyramids = n_pyramids
        self.n_outputs = n_outputs
        return

    def forward(
            self,
            xs: typing.List[torch.Tensor],
            mask: torch.Tensor) -> torch.Tensor:

        '''
        Args:
            xs (torch.Tensor): A List of (B, C, sH, sW)
            mask (torch.Tensor): (1, 1, H, W), where mask.sum() == N

        Return:
            torch.Tensor: (B, C, H, W)
        '''
        # List of (B, C, H, W)
        b, _, h, w = xs[0].size()
        f = xs[0].new_zeros(b, self.n_outputs * self.n_pyramids, h, w)
        for i, (x, ss) in enumerate(zip(xs, self.scale_specific)):
            ci = i * self.n_outputs
            f[:, ci:(ci + self.n_outputs)] = ss(x, mask)

        #fs = [self.shared(x, mask) for x in xs]
        #f = torch.cat(fs, dim=1)
        f = f + self.feature(f, mask)
        return f


class AdaptiveMSSampler(nn.Module):
    '''
    no_position, log_scale by default
    '''

    def __init__(
            self,
            n_pyramids: int=3,
            n_feats: int=16) -> None:

        super().__init__()
        self.ms_feature = MSResFeature(
            n_pyramids=n_pyramids,
            n_feats=64,
            depth=6,
        )
        n_feats_ex = n_pyramids * n_feats
        m = [
            nn.Conv1d(1 + n_feats_ex, n_feats_ex, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feats_ex, n_feats_ex, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feats_ex, n_pyramids, 1),
        ]
        self.sampler = nn.Sequential(*m)
        self.n_pyramids = n_pyramids
        return

    def get_dsda(
            self,
            b: int,
            yi: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int]) -> torch.Tensor:

        with torch.no_grad():
            dsda = warp.calc_dsda_projective(sizes, m)
            dsda = dsda[yi]
            dsda = (dsda + 1e-8).log()
            # (1, 1, N)
            dsda = dsda.view(1, 1, -1)
            # (B, 1, N)
            dsda = dsda.repeat(b, 1, 1)

        return dsda

    def forward(
            self,
            xs: typing.List[torch.Tensor],
            mask: torch.Tensor,
            grid: torch.Tensor,
            yi: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int],
            **kwargs) -> torch.Tensor:
        '''
        Args:
            xs (torch.Tensor): A list of (B, C, H, W)
            grid (torch.Tensor): (2, N)
            yi (torch.Tensor):
            m (torch.Tensor): (3, 3)
            sizes (tuple):

        Return:
            torch.Tensor: (1, 1, sizes[0], sizes[1], n)
        '''
        f = self.ms_feature(xs, mask)
        # (B, C, N)
        f = f.view(f.size(0), f.size(1), -1)
        f = f[..., yi]
        dsda = self.get_dsda(f.size(0), yi, m, sizes)

        # (B, C + 1, N)
        f = torch.cat((f, dsda), dim=1)
        p = self.sampler(f)

        # (B, 3, H * W)
        ws = p.new_zeros(f.size(0), self.n_pyramids, sizes[0] * sizes[1])
        ws[..., yi] = p
        # (B, 3, H, W)
        ws = ws.view(f.size(0), self.n_pyramids, sizes[0], sizes[1])
        out = sum(w * x for w, x in zip(ws.split(1, dim=1), xs))
        return out


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
            kernel_net_regular: bool=False,
            kernel_regularize: bool=True,
            kernel_depthwise: bool=False,
            kernel_bilinear: bool=False,
            kernel_no_mul: bool=False,
            ms_blend: str='net',
            abl_awl: bool=False,
            abl_multi: bool=False,
            abl_recon: bool=False,
            **kwargs) -> None:

        super().__init__()
        # x1 network construction
        n_feats = 64
        # Use pre-trained models for x2 and higher
        self.backbone_multi = None
        if backbone == 'mdsr':
            self.backbone_multi = mdsr_f.MDSRF(
                n_feats=n_feats,
                multi_scale=abl_multi,
            )
        elif backbone == 'mrdb-ps':
            self.backbone_multi = mrdb_fps.MRDBFPS(
                n_feats=n_feats,
                multi_scale=abl_multi,
            )

        n_pyramids = int(math.log2(max_scale)) + 1
        if abl_multi:
            self.amss = AdaptiveMSSampler(n_pyramids=n_pyramids)
        else:
            self.amss = None

        if abl_recon:
            self.recon = MaskedResRecon(
                n_inputs=n_feats,
                n_feats=n_feats,
                n_outputs=n_colors,
                depth=10,
            )
        else:
            self.recon = MaskedConv(n_feats, 3, 3, padding=1)

        self.residual = residual
        self.normal_upsample = normal_upsample
        self.elliptical_upsample = elliptical_upsample
        self.kernel_net = kernel_net
        self.ms_blend = ms_blend
        if abl_awl:
            depthwise = n_feats
            kwargs = {
                'kernel_size': kernel_net_size,
                'net_regular': kernel_net_regular,
                'regularize': kernel_regularize,
                'dw': depthwise,
                'bilinear': kernel_bilinear,
                'no_mul': kernel_no_mul,
            }
            self.kernel_estimator = nn.ModuleList()
            for _ in range(n_pyramids):
                self.kernel_estimator.append(
                    kernel.KernelEstimator(**kwargs)
                )
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
            'kernel_net_regular': cfg.kernel_net_regular,
            'kernel_regularize': not cfg.kernel_noreg,
            'kernel_depthwise': cfg.kernel_depthwise,
            'kernel_bilinear': cfg.kernel_bilinear,
            'kernel_no_mul': cfg.kernel_no_mul,
            'ms_blend': cfg.ms_blend,
            'abl_awl': cfg.abl_awl,
            'abl_multi': cfg.abl_multi,
            'abl_recon': cfg.abl_recon,
        }
        return kwargs

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
        if self.amss is None:
            pyramids = [pyramids]

        if not self.training and empty_cache:
            cuda.empty_cache()

        ys = []
        ref = {}
        for pi, lambda_p in enumerate(pyramids):
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
                net = self.kernel_estimator[pi]
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

        if self.amss is None:
            out = ys[0]
        else:
            out = self.amss(ys, mask, **ref)

        out = mask * out
        out = self.recon(out, mask)
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

