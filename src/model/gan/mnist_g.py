from model import common

from torch import nn


class Generator(nn.Module):

    def __init__(self, n_z=100, n_feats=64):
        super().__init__()
        self.n_feats = n_feats
        out_channels = 2 * n_feats * 7 * 7
        self.linear = nn.Sequential(
            nn.Linear(n_z, 16 * n_feats),
            nn.BatchNorm1d(16 * n_feats),
            nn.ReLU(True),
            nn.Linear(16 * n_feats, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feats, n_feats, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feats, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        common.init_gans(self)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'n_z': cfg.n_z,
            'n_feats': cfg.n_feats,
        }
        return kwargs

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 2 * self.n_feats, 7, 7)
        x = self.deconv(x)
        return x

