import numpy as np

import torch
from torch import nn


class MaskedConv(nn.Conv2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.mask_conv.weight.data.fill_(1 / np.prod(self.kernel_size))
        self.mask_conv.requires_grad_(False)
        return

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): (B, C, H, W)
            mask (torch.Tensor): (1, 1, H, W)

        Return:
        '''
        with torch.no_grad():
            w = self.mask_conv(mask)
            w.clamp_(min=1e-8)
            w.reciprocal_()
            w *= mask

        x = super().forward(x)
        # To save memory
        if self.training:
            x = w * x
        else:
            x *= w

        return x

