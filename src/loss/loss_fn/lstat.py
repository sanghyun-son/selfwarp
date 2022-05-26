import torch
from torch import nn
from torch.nn import functional as F


class LocalStat(nn.Module):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

        n = [s for s in name if s.isdigit()]
        n = ''.join(n)
        self.window_size = int(n)
        self.dense = 'd' in name

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()

        if self.dense:
            sx = 1
            sy = 1
        else:
            sx = None
            sy = None

        if xh > yh:
            scale = xh // yh
            sx *= scale
            # B x C x h x w
            x_mean = F.avg_pool2d(x, scale * self.window_size, stride=sx)
            x2_mean = F.avg_pool2d(x.pow(2), scale * self.window_size, stride=sx)
            y_mean = F.avg_pool2d(y, self.window_size, stride=sy)
            y2_mean = F.avg_pool2d(y.pow(2),self.window_size, stride=sy)
        else:
            scale = yh // xh
            sy *= scale
            x_mean = F.avg_pool2d(x, self.window_size, stride=sx)
            x2_mean = F.avg_pool2d(x.pow(2), self.window_size, stride=sx)
            y_mean = F.avg_pool2d(y, scale * self.window_size, stride=sy)
            y2_mean = F.avg_pool2d(y.pow(2), scale * self.window_size, stride=sx)

        eps = 1e-4
        x_sigma = torch.sqrt(x2_mean - x_mean.pow(2)) + eps
        y_sigma = torch.sqrt(y2_mean - y_mean.pow(2)) + eps

        loss = y_sigma.pow(2) + (x_mean - y_mean).pow(2)
        loss = loss / (2 * x_sigma.pow(2))
        loss = loss + (x_sigma / y_sigma).log() - 0.5
        loss = loss.mean()
        return loss

