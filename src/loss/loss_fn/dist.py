import torch
from torch import nn
from torch.nn import functional as F

from ops import filters


class Distance(nn.Module):
    '''
    A generalized L1 distance function

    Args:
        name (str):
    '''

    def __init__(self, name: str) -> None:
        super().__init__()
        parse = name.split('-')
        if len(parse) == 1:
            self.kernel = None
        else:
            sigma = int(parse[1]) / 10
            print(sigma)
            self.kernel = filters.gaussian_kernel(sigma=sigma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kernel is not None:
            x = filters.filtering(x, self.kernel)
            y = filters.filtering(y, self.kernel)

        loss = F.l1_loss(x, y)
        return loss
