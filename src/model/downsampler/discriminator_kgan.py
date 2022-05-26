from model import common
from torch import nn


class PatchDiscriminator(nn.Sequential):

    def __init__(self, n_colors=3, width=64, normalization: str='batch'):
        kwargs = {'norm': normalization, 'act': 'lrelu', 'bias': False}
        m = [nn.Conv2d(n_colors, width, 7)]
        for _ in range(5):
            m.append(common.BasicBlock(width, width, 1, **kwargs))

        m.append(nn.Conv2d(width, 1, 1))
        super().__init__(*m)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'n_colors': cfg.n_colors,
            'width': cfg.width_sub,
            'normalization': cfg.normalization,
        }
        return kwargs

