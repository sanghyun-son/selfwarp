from model import common

from torch import nn


class Discriminator(nn.Module):

    def __init__(self, n_feats=64):
        super().__init__()
        self.n_feats = n_feats
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_feats, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(n_feats, 2 * n_feats, 4, stride=2, padding=1),
            nn.BatchNorm2d(2 * n_feats),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        in_channels = 2 * n_feats * 7 * 7
        self.linear = nn.Sequential(
            nn.Linear(in_channels, 16 * n_feats),
            nn.BatchNorm1d(16 * n_feats),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(16 * n_feats, 1),
        )
        common.init_gans(self)

    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'n_feats': cfg.n_feats,
        }
        return kwargs

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 2 * self.n_feats * 7 * 7)
        x = self.linear(x)
        return x

