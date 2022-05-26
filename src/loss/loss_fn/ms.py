from torch import nn


class MeanStd(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.eps = 1e-4

    def forward(self, x, y):
        x = x.view(x.size(0) * x.size(1), -1)
        y = y.view(y.size(0) * y.size(1), -1)
        x_mean = x.mean(dim=-1)
        y_mean = y.mean(dim=-1)
        x_std = x.std(dim=-1) + self.eps
        y_std = y.std(dim=-1) + self.eps

        loss = ((y_mean - x_mean)**2 + y_std**2) / (2 * x_std**2)
        loss = loss + (x_std.log() - y_std.log() - 0.5)
        loss = loss.mean()

        return loss

