from torch import nn
from torch.nn import functional

def make_measure(cfg, name=None, logger=None):
    return Avg()


class Avg(nn.Module):

    def __init__(self):
        super(Avg, self).__init__()

    def forward(self, pred, target):
        mask = (pred == 0) + (pred == 1)
        avg = pred[mask].float().mean()
        return avg

