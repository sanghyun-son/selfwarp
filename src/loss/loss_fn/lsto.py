import random

import torch
from torch import nn
from torch.nn import functional as F


class LocalRandomMean(nn.Module):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

        n = [s for s in name if s.isdigit()]
        n = ''.join(n)
        self.window_size = int(n)
        self.l1 = '-l' in name
        self.prob = 0.5

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xh = x.size(-2)
        yh = y.size(-2)

        if xh < yh:
            return self.forward(y, x)

        if xh > yh:
            scale = xh // yh
            if self.dense:
                pass
            else:
                x_pool = self.mask_pool(x, scale * self.window_size)
                y_pool = self.mask_pool(y, self.window_size)

        if self.l1:
            loss = F.l1_loss(x_pool, y_pool)
        else:
            loss = F.mse_loss(x_pool, y_pool)

        return loss

    def mask_pool(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        _, _, h, w = x.size()
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mult = 1 / self.prob
            for batch in mask:
                for channel in batch:
                    for ih in range(0, h, window_size):
                        for iw in range(0, w, window_size):
                            window = channel[
                                ih:(ih + window_size),
                                iw:(iw + window_size),
                            ]
                            n = window.nelement()
                            idx = [i for i in range(n)]
                            random.shuffle(idx)
                            cut = int(self.prob * n)
                            mask_value = (window.new_tensor(idx) < cut)
                            mask_value = mask_value.view(window.size()).bool()
                            window[mask_value] = mult

        y = mask * x
        pool = F.avg_pool2d(y, window_size)
        return pool


if __name__ == '__main__':
    pass