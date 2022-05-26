from bicubic_pytorch import core

import torch
from torch import nn
from torch.nn import functional as F


class Bicubic(nn.Module):

    def __init__(self, name: str) -> None:
        super().__init__()
        return

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        wx = x.size(-1)
        wy = y.size(-1)
        if wx <= wy:
            return self.forward(y, x)

        x = core.imresize(x, scale=(wy / wx), kernel='cubic', antialiasing=True)
        loss = F.l1_loss(x, y)
        return loss
