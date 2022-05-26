from torch import nn
from torch.nn import functional


class KernelSimilarity(nn.Module):

    def __init__(self, name=None):
        super().__init__()

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        dist = functional.cosine_similarity(x, y, dim=0)
        return dist
