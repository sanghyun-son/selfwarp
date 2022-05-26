from model import common
from misc.gpu_utils import parallel_forward as pforward

import torch
from torch import nn
from torch.nn import functional
from torchvision import models


class ResNet(nn.Module):

    def __init__(self, name):
        super().__init__()
        resnet_available = ['18', '34', '50', '101', '152']
        resnet_model = models.resnet50
        for depth in resnet_available:
            if depth in name:
                resnet_model = getattr(models, 'resnet{}'.format(depth))
                break

        self.sub_mean = common.MeanShift()
        self.resnet = resnet_model(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Identity()

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        if 'l1' in name:
            self.dist = functional.l1_loss
        else:
            self.dist = functional.mse_loss

    def __str__(self):
        return 'loss.resnet.ResNet()'

    def forward(self, output, target):
        def get_response(x):
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)

            x = self.sub_mean(x)
            x = pforward(self.resnet, x)
            return x

        res_output = get_response(output)
        with torch.no_grad():
            res_target = get_response(target)

        loss = self.dist(res_output, res_target)
        return loss

    def state_dict(self, *args, **kwargs):
        return {}
