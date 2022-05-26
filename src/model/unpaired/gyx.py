from model import common

import torch
from torch import nn


class GYX(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        n_feats = 64
        negative_slope = 0.2

        self.head1 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            common.ResBlock(n_feats, 5, norm='batch', act='lrelu'),
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            common.ResBlock(n_feats, 5, norm='batch', act='lrelu'),
        )
        m = [
            nn.Conv2d(2 * n_feats, n_feats, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        ]
        for _ in range(6):
            m.append(common.ResBlock(n_feats, 5, norm='batch', act='lrelu'))

        for _ in range(2):
            m.append(common.BasicBlock(
                n_feats, n_feats, 1, norm='batch', act='lrelu',
            ))

        m.append(nn.Conv2d(n_feats, 3, 1, padding=0, bias=True))
        self.body = nn.Sequential(*m)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
        n.normal_()

        h1 = self.head1(x)
        h2 = self.head2(n)
        hcat = torch.cat((h1, h2), dim=1)
        x = self.body(hcat)
        return x
