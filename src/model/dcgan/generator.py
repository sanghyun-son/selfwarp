from model import common
from torch import nn


class Generator(nn.Sequential):

    def __init__(self, n_z=100, n_feats=64, n_colors=3):
        def basic_block(in_channels, out_channels, stride=2, padding=1):
            args = [in_channels, out_channels, 4]
            kwargs = {'stride': stride, 'padding': padding}
            c = nn.ConvTranspose2d(*args, **kwargs)
            b = nn.BatchNorm2d(out_channels)
            r = nn.ReLU(True)
            return c, b, r

        m = []
        m.extend(basic_block(n_z, 8 * n_feats, stride=1, padding=0))
        m.extend(basic_block(8 * n_feats, 4 * n_feats))
        m.extend(basic_block(4 * n_feats, 2 * n_feats))
        m.extend(basic_block(2 * n_feats, n_feats))
        m.append(nn.ConvTranspose2d(n_feats, n_colors, 4, stride=2, padding=1))
        m.append(nn.Tanh())

        super().__init__(*m)
        common.init_gans(self)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        return {
            'n_z': cfg.n_z,
            'n_feats': cfg.n_feats,
            'n_colors': cfg.n_colors,
        }

