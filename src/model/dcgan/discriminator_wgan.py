from model.utils import common
from torch import nn
from torch.nn import init

def model_class(cfg=None, conv=common.default_conv, make=False):
    if make:
        kwargs = DiscriminatorWGANGP.get_kwargs(cfg=cfg, conv=conv)
        return DiscriminatorWGANGP(**kwargs)
    else:
        return DiscriminatorWGANGP


class DiscriminatorWGANGP(nn.Module):

    def __init__(self, depth=8, n_colors=3, rgb=255, n_feats=64, patch=96):
        super(DiscriminatorWGANGP, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.GroupNorm(1, out_dim),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(64, 64 * 2),
            conv_ln_lrelu(64 * 2, 64 * 4),
            conv_ln_lrelu(64 * 4, 64 * 8),
            nn.Conv2d(64 * 8, 1, 4)
        )

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        return {
            'n_feats': cfg.n_feats,
            'patch': cfg.dpatch,
        }

    def forward(self, x):
        x = self.ls(x)
        x = x.view(-1, 1)
        return x

