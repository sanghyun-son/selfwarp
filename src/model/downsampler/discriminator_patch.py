from model import common
from torch import nn


class PatchDiscriminator(nn.Sequential):

    def __init__(self, depth=7, n_colors=3, width=64):

        kwargs = {'norm': 'batch', 'act': 'lrelu'}
        m = [
            common.BasicBlock(n_colors, width, 3, **kwargs),
        ]
        for _ in range(4):
            m.append(common.BasicBlock(width, width, 3, **kwargs))

        for _ in range(depth - 4):
            m.append(common.BasicBlock(width, width, 1, **kwargs))

        m.append(nn.Conv2d(width, 1, 1, padding=0))
        super().__init__(*m)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'depth': cfg.depth_sub,
            'n_colors': cfg.n_colors,
            'width': cfg.width_sub,
        }
        return kwargs

