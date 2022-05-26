import math

from model import common
import torch
from torch import nn


class DownSelf(nn.Module):

    def __init__(
            self, scale=4, n_feats=64, depth=4,
            conv=common.default_conv, stochastic=False):

        super().__init__()
        self.stochastic = stochastic

        m = []
        in_channels = 2
        log_scale = int(math.log(scale, 2))
        for _ in range(log_scale):
            m.append(common.PixelSort())
            m.append(conv(4 * in_channels, n_feats, 5))
            in_channels = n_feats

        self.downsampler = nn.Sequential(*m)

        resblock = lambda: common.ResBlock(n_feats, 5, act='relu')
        m = [resblock() for _ in range(depth)]
        self.resblocks = nn.Sequential(*m)
        self.recon = conv(n_feats, 1, 5)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'scale': cfg.scale,
            'n_feats': cfg.n_feats,
            'depth': cfg.depth,
            'conv': conv,
            'stochastic': cfg.stochastic,
        }
        return kwargs

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * c, 1, h, w)
        if self.stochastic:
            n = torch.rand_like(x)
        else:
            n = 0.5 * torch.ones_like(x)

        x = torch.cat((x, n), dim=1)
        x = self.downsampler(x)
        res = self.resblocks(x)
        x = self.recon(x + res)
        hh = x.size(-2)
        ww = x.size(-1)
        x = x.view(b, c, hh, ww)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for k in own_state.keys():
            if k in state_dict:
                own_state[k] = state_dict[k]

        return super().load_state_dict(own_state, strict=strict)

