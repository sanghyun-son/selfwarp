import torch
from torch import nn


class GradientPenalty(nn.Module):

    def __init__(self, ref_node: str) -> None:
        super().__init__()
        self.requires_ref = True

    def __str__(self) -> str:
        return 'GP'

    def forward(self, root, ref_node: str) -> float:
        target = root[ref_node].f
        if hasattr(target, 'loss_gradient'):
            loss_gp = target.loss_gradient
        else:
            loss_gp = 0

        # To prevent backpropagation
        if isinstance(loss_gp, torch.Tensor):
            loss_gp = loss_gp.item()

        return loss_gp

