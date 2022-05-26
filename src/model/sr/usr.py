import types
import typing

from config import get_config
from model import common

import torch
from torch import nn


class USR(nn.Module):

    def __init__(
            self,
            scale: int=2,
            depth: int=4,
            n_colors: int=3,
            n_feats: int=64,
            conv: nn.Module=common.default_conv) -> None:

        super().__init__()
        # Learnable upsampling layer
        self.conv = conv(n_colors, n_feats, 3)
        self.resblocks = nn.Sequential(
            common.ResBlock(n_feats, 3, conv=conv),
            common.ResBlock(n_feats, 3, conv=conv),
        )
        self.upsample = nn.Sequential(
            common.Upsampler(scale, n_feats, conv=conv),
            conv(n_feats, n_colors, 3),
        )
        # VDSR-like SR module
        m = [
            conv(n_colors, n_feats, 3),
        ]
        for _ in range(depth):
            m.append(common.ResBlock(n_feats, 3, conv=conv))

        m.append(conv(n_feats, n_colors, 3))
        self.vdsr = nn.Sequential(*m)

    @staticmethod
    def get_kwargs(
            cfg: types.SimpleNamespace,
            conv: nn.Module=common.default_conv) -> dict:

        parse_list = [
            'scale',
            #'depth',
            'n_colors',
            'n_feats',
        ]
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['conv'] = conv
        return kwargs

    def forward(
            self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv(x)
        x = x + self.resblocks(x)
        x = self.upsample(x)
        y = x + self.vdsr(x)
        return x, y
