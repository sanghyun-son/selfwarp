from model import common
from torch import nn


class Discriminator(nn.Sequential):
    '''
    From A. Radford, L. Metz, and S. Chintala,
    "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
    See https://arxiv.org/pdf/1511.06434.pdf for more detail.
    '''

    def __init__(self, n_colors=3, n_feats=64, norm='batch'):
        def layers(in_channels, out_channels, stride=2, padding=1):
            if norm == 'batch':
                norm_module = nn.BatchNorm2d(out_channels)
            elif norm == 'layer':
                norm_module = nn.GroupNorm(1, out_channels)

            args = [in_channels, out_channels, 4]
            kwargs = {'stride': stride, 'padding': padding}
            c = nn.Conv2d(*args, **kwargs)
            n = norm_module
            l = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            return c, n, l

        m = []
        m.append(nn.Conv2d(n_colors, n_feats, 4, stride=2, padding=1))
        m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        m.extend(layers(n_feats, 2 * n_feats))
        m.extend(layers(2 * n_feats, 4 * n_feats))
        m.extend(layers(4 * n_feats, 8 * n_feats))
        m.append(nn.Conv2d(8 * n_feats, 1, 4, stride=1, padding=0))

        super().__init__(*m)
        common.init_gans(self)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        if 'wgan' in cfg.loss and 'gp' in cfg.loss:
            norm = 'layer'
        else:
            norm = 'batch'

        return {
            'n_feats': cfg.n_feats,
            'norm': norm,
        }

    def forward(self, x):
        x = super().forward(x)
        x = x.view(-1, 1)
        return x

