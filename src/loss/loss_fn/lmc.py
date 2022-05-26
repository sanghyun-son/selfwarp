import random

import torch
from torch import nn
from torch.nn import functional as F


class LocalMonteCarlo(nn.Module):

    def __init__(self, cfg: str=None):
        super().__init__()
        self.cfg = cfg

        n = [s for s in cfg if s.isdigit()]
        n = ''.join(n)
        self.window_size = int(n)
        self.l1 = '-l' in cfg
        self.prob = 0.75

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xh = x.size(-2)
        yh = y.size(-2)

        if xh < yh:
            return self.forward(y, x)

        if xh > yh:
            scale = xh // yh
            x_pool = self.mask_pool(x, scale * self.window_size)
            y_pool = self.mask_pool(y, self.window_size)

        if self.l1:
            loss = F.l1_loss(x_pool, y_pool)
        else:
            loss = F.mse_loss(x_pool, y_pool)

        return loss

    def mask_pool(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        _, _, h, w = x.size()
        side = min(h, w)
        side = window_size * (side // window_size)
        x = x[..., :side, :side]

        # To avoid numerical instability
        eps = 1e-8
        with torch.no_grad():
            mask = torch.rand_like(x)
            mask = (mask <= self.prob).float()
            mask_coeff = F.avg_pool2d(mask, window_size)
            mask_coeff = (mask_coeff + eps).reciprocal()
            mask_coeff = F.interpolate(
                mask_coeff, scale_factor=window_size, mode='nearest',
            )
            mask = mask_coeff * mask

        x = mask * x
        pool = F.avg_pool2d(x, window_size)
        return pool

if __name__ == '__main__':
    import imageio
    import numpy as np

    def to_tensor(x):
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        x = x / 127.5 - 1
        x.unsqueeze_(0)
        return x

    hr = imageio.imread('~/dataset/benchmark/set5/HR/butterfly.png')
    lr = imageio.imread('~/dataset/benchmark/set5/LR_bicubic/X2/butterfly.png')
    hr = to_tensor(hr)
    lr = to_tensor(lr)

    l = LocalMonteCarlo('l4')
    p = l.mask_pool(hr, 32)
    p_lr = l.mask_pool(lr, 16)
    print(p[0, 0])
    print(p_lr[0, 0])

    print(F.l1_loss(p, p_lr))
    print('')

    naive_p = F.avg_pool2d(hr, 32)
    naive_p_lr = F.avg_pool2d(lr, 16)
    print(naive_p[0, 0])
    print(naive_p_lr[0, 0])
    print(F.l1_loss(naive_p, naive_p_lr))
