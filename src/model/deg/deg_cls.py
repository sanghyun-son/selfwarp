import types
import typing

import torch
from torch import nn
from torchvision import models
from torchvision.models import resnet


class DegClassfier(models.ResNet):

    def __init__(
            self,
            n_degs: int=3,
            no_srfeat: bool=False,
            regression: bool=False) -> None:

        block = resnet.Bottleneck
        # ResNet-50 configuration
        super().__init__(block, [3, 4, 6, 3])
        if no_srfeat:
            n_feats = 3
        else:
            n_feats = 15

        self.conv1 = nn.Conv2d(
            n_feats, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        
        if regression:
            n_out = 1
        else:
            n_out = 5

        self.fcs = nn.ModuleList(
            [nn.Linear(512 * block.expansion, n_out) for _ in range(n_degs)]
        )
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = {
            'no_srfeat': cfg.no_srfeat,
            'regression': cfg.regression,
        }
        return kwargs

    def _forward_impl(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = [fc(x) for fc in self.fcs]
        return x


if __name__ == '__main__':
    net = DegClassfier()
    net.cuda()
    print(net)
    x = torch.randn(1, 6, 224, 224)
    x = x.cuda()
    y = net(x)
    print(len(y))
    print(y[0].size())
