from model import common
from torch import nn


class PatchDiscriminator(nn.Sequential):

    def __init__(self, n_colors=3, width=64):

        kwargs = {'norm': 'batch', 'act': 'lrelu', 'bias': True}
        m = [nn.Conv2d(n_colors, width, 7, padding=3)]
        for _ in range(5):
            m.append(common.BasicBlock(width, width, 3, **kwargs))

        m.append(nn.Conv2d(width, 1, 1))
        super().__init__(*m)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'n_colors': cfg.n_colors,
            'width': cfg.width_sub,
        }
        return kwargs

