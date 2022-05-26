import math
import types
import typing

import numpy as np

from model import common
from model.sr import edsr_f
from model.superwarp import dual
from model.superwarp import dual_recon
from model.superwarp import masked_conv
from model.superwarp import kernel
from utils import svf
from utils import warp
from utils import random_transform
from misc.gpu_utils import parallel_forward

import torch
from torch import cuda
from torch import nn
from torch.nn import functional as F


class MaskedResBlock(nn.Module):

    def __init__(self, n_feats: int=64) -> None:
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.conv1 = masked_conv.MaskedConv(
            n_feats,
            n_feats,
            kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = masked_conv.MaskedConv(
            n_feats,
            n_feats,
            kernel_size,
            padding=padding,
        )
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = mask * x
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
            self.ms.append(masked_conv.MaskedConv(
                n_inputs, n_feats, kernel_size, padding=padding,
            ))
            depth -= 1

        while depth > 0:
            self.ms.append(MaskedResBlock(n_feats))
            depth -= 2

        if n_outputs != n_feats:
            self.ms.append(masked_conv.MaskedConv(
                n_feats, n_outputs, kernel_size, padding=padding,
            ))

        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = mask * x
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
        self.shared = MaskedResSeq(
            n_inputs=n_feats,
            n_feats=n_feats,
            n_outputs=n_outputs,
            depth=depth // 2,
        )
        n_feats_ex = n_pyramids * n_outputs
        self.feature = MaskedResSeq(
            n_inputs=n_feats_ex,
            n_feats=n_feats_ex,
            n_outputs=n_feats_ex,
            depth=(depth - depth // 2),
        )
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
        fs = [self.shared(x, mask) for x in xs]
        f = torch.cat(fs, dim=1)
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
            nn.BatchNorm1d(n_feats_ex),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feats_ex, n_feats_ex, 1),
            nn.BatchNorm1d(n_feats_ex),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feats_ex, n_feats_ex, 1),
            nn.BatchNorm1d(n_feats_ex),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feats_ex, n_pyramids, 1),
        ]
        self.sampler = nn.Sequential(*m)
        self.n_pyramids = n_pyramids
        return

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
        with torch.no_grad():
            dsda = warp.calc_dsda_projective(sizes, m)
            dsda = dsda[yi]
            dsda = (dsda + 1e-8).log()
            # (1, 1, N)
            dsda = dsda.view(1, 1, -1)
            # (B, 1, N)
            dsda = dsda.repeat(f.size(0), 1, 1)

        # (B, C + 1, N)
        f = torch.cat((f, dsda), dim=1)
        logits = self.sampler(f)
        # (B, 3, N)
        p = F.softmax(logits, dim=1)

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
        kernel_size = 3
        self.recon = MaskedResSeq(
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


class SuperWarpF(nn.Module):

    def __init__(
            self,
            n_colors: int=3,
            max_scale: int=4,
            kernel_size: int=4,
            **kwargs) -> None:

        super().__init__()
        # x1 network construction
        n_feats = 64
        self.conv = common.default_conv(n_colors, n_feats, 3)
        m = [common.ResBlock(n_feats, 3) for _ in range(3)]
        m.append(common.default_conv(n_feats, n_feats, 3))
        self.x1 = nn.Sequential(*m)

        # Use pre-trained models for x2 and higher
        self.x2 = edsr_f.EDSRF(scale=2, n_feats=n_feats)
        self.x4 = edsr_f.EDSRF(scale=4, n_feats=n_feats)

        n_pyramids = int(math.log2(max_scale)) + 1
        self.amss = AdaptiveMSSampler(n_pyramids=n_pyramids)
        self.recon = MaskedResRecon(
            n_inputs=n_feats,
            n_feats=n_feats,
            n_outputs=n_colors,
        )
        self.kernel = kernel.KernelEstimator(kernel_size=kernel_size)
        self.fill_value = -255
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = {
            'n_colors': cfg.n_colors,
            'max_scale': cfg.max_scale,
            'kernel_size': cfg.kernel_size,
        }
        return kwargs

    def x1_forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x1 = x1 + self.x1(x1)
        return x1

    def forward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Optional[typing.Tuple[int, int]]=None,
            get_scalemap: bool=False) -> torch.Tensor:

        m = m.cpu()
        if sizes is None:
            sizes = (256, 256)

        # We do not build scale pyramid first to save memory
        pyramids = [self.x1_forward, self.x2, self.x4]
        ys = []
        ref = {}
        for pi, lambda_p in enumerate(pyramids):
            p = lambda_p(x)
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
            # Estimate spatially-varying kernel weights
            k = self.kernel(
                ms.cuda(), (p.size(-2), p.size(-1)), ref['sizes'], grid, yi,
            )
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

        if not self.training:
            cuda.empty_cache()

        out = self.amss(ys, mask, **ref)
        out = self.recon(out, mask)
        out = mask * out + (1 - mask) * self.fill_value
        out_full = out.new_full(
            (out.size(0), out.size(1), sizes[0], sizes[1]), self.fill_value,
        )
        ref_h, ref_w = ref['sizes']
        dy, dx = ref['shift']
        ref_h = min(ref_h, sizes[0] - dy)
        ref_w = min(ref_w, sizes[1] - dx)
        slice_h = slice(dy, dy + ref_h)
        slice_w = slice(dx, dx + ref_w)
        out_full[..., slice_h, slice_w] = out[..., :ref_h, :ref_w]
        return out_full


REPRESENTATIVE = SuperWarpF

