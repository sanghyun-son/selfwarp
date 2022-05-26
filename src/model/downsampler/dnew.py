from torch import nn
from model import common


class DownNew(nn.Module):

    def __init__(self, scale=2, n_feats=64, depth=3, conv=common.default_conv):
        super().__init__()
        self.input = conv(1, n_feats, 3)

        m = []
        for _ in range(depth):
            m.append(common.ResBlock(n_feats, 3, act='relu'))

        self.body = nn.Sequential(*m)

        m = [
            conv(n_feats, n_feats, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            conv(n_feats, 1, 3),
        ]
        self.down = nn.Sequential(*m)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'scale': cfg.scale,
            'n_feats': cfg.n_feats,
            'depth': cfg.depth,
        }
        return kwargs

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(-1, 1, h, w)
        x = self.input(x)
        res = self.body(x)
        x = self.down(x + res)
        _, _, hh, ww = x.size()
        x = x.view(b, c, hh, ww)
        return x