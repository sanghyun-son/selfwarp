import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F


class GradientCentralize(autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        grad_mean = grad_output.mean(dim=-1, keepdim=True)
        grad_mean = grad_mean.mean(dim=-2, keepdim=True)
        grad_output = grad_output - grad_mean
        return grad_output


centralize = GradientCentralize.apply


class GradientCentralizeModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return centralize(x)
