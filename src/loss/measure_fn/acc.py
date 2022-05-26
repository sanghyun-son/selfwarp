from numpy import corrcoef
from torch import nn
from torch.nn import functional


class Acc(nn.Module):

    def __init__(self, name='acc', ignore_index=-100) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        get_topk = name.split('-')
        if len(get_topk) > 1:
            self.topk = int(get_topk[1])
        else:
            self.topk = 1

        return

    def forward(self, pred, target):
        n = (target != self.ignore_index).long().sum()
        pred = functional.softmax(pred, dim=1)
        '''
        _, pred_idx = pred.max(dim=1)
        correct = (pred_idx == target).long().sum()
        acc = 100 * correct.item() / max(1, n.item())
        '''
        _, pred_idx = pred.topk(self.topk, 1, largest=True, sorted=True)
        pred_idx.squeeze_(0)
        correct = (pred_idx == target.view(1, -1)).long()
        acc = 100 * correct.sum() / max(1, n.item())
        return acc


def main():
    import torch
    a = torch.randn(1, 2, 8, 8)
    print(a)
    _, idx = functional.softmax(a, dim=1).max(dim=1)
    print(idx)
    b = (torch.randn(1, 8, 8) > 0).long()
    b[..., 2, 3] = -100
    b[..., 5, 2] = -100
    print(b)

    acc = Acc()
    print(acc(a, b))
    print(100 * (idx == b).long().sum().item() / 62)

if __name__ == '__main__':
    main()