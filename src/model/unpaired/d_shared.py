from matplotlib.pyplot import new_figure_manager
from model import common

import torch
from torch import nn


class DShared(nn.Sequential):

    def __init__(self, depth: int=5, scale: int=1) -> None:
        n_feats = 64
        m = []
        if scale > 1:
            stride1 = 2
        else:
            stride1 = 1

        if scale > 2:
            stride2 = 2
        else:
            stride2 = 1

        m = [
            common.BasicBlock(
                3, n_feats, 3, stride=stride1, norm='batch', act='lrelu',
            ),
            common.BasicBlock(
                n_feats, n_feats, 3, stride=stride2, norm='batch', act='lrelu',
            ),
        ]
        for _ in range(depth - 3):
            m.append(common.BasicBlock(
                n_feats, n_feats, 3, 
            ))

        m.append(nn.Conv2d(n_feats, 3, 3, padding=1, bias=True))
        super().__init__(*m)
        return