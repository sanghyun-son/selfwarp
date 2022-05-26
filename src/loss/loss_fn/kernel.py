import torch
from torch import nn
from torch.nn import functional as F

from ops import filters


class DownsampleLoss(nn.Module):
    '''

    '''

    def __init__(self, name: str) -> None:
        super().__init__()

    def forward(self, x, y, k):
        with torch.no_grad():
            x_hat = filters.downsampling(y, k, scale=2)
            x_hat = 127.5 * (x_hat + 1)
            x_hat.round()
            x_hat = x_hat / 127.5 - 1

        loss = F.l1_loss(x, x_hat)
        return loss
