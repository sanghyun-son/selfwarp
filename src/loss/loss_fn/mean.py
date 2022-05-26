from torch import nn


class Mean(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x, y):
        x = x.view(x.size(0) * x.size(1), -1)
        y = y.view(y.size(0) * y.size(1), -1)
        x_mean = x.mean(dim=-1)
        y_mean = y.mean(dim=-1)

        loss = (y_mean - x_mean)**2
        loss = loss.mean()

        return loss

