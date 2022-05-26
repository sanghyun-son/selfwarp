import torch
from torch import nn
from torch.nn import functional


class LFL2(nn.Module):
    '''
    L2 loss between images after low-pass filtering.
    '''
    def __init__(self, kernel_size=21, sigma=3):
        super(LFL2, self).__init__()
        self.conv = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        ax = torch.arange(kernel_size).view(1, kernel_size).float()
        ax = ax.repeat(kernel_size, 1) - kernel_size // 2
        ay = ax.t()
        distmap = ax**2 + ay**2
        distmap = -distmap / (2 * sigma)
        gaussian_kernel = distmap.exp()
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        self.conv.weight.data.copy_(gaussian_kernel.view(
            1, 1, kernel_size, kernel_size
        ))

    def forward(self, x, y):
        b, c, h, w = x.size()
        x = x.view(b * c, 1, h, w)
        y = y.view(b * c, 1, h, w)
        x = self.conv(x).view(b, c, h, w)
        y = self.conv(y).view(b, c, h, w)
        loss = functional.mse_loss(x, y)
        return loss


if __name__ == '__main__':
    pass