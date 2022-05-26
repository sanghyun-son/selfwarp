from model import common
from misc.gpu_utils import parallel_forward as pforward

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class VGG(nn.Module):
    
    def __init__(self, name):
        super(VGG, self).__init__()
        self.vgg = common.extract_vgg(name)
        self.name = name
        self.l1 = 'l' in name
        self.scale = 's' in name
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return self.name.upper()

    def forward(self, sr, hr, **kwargs):
        '''
        Args:
            sr (torch.Tensor):
            hr (torch.Tensor):
            kwargs (dict): A placeholder for dummy keyworkd arguments.
        '''
        def get_features(x):
            if self.scale:
                x = F.interpolate(
                    x, size=(224, 224), align_corners=False, mode='bicubic'
                )

            return pforward(self.vgg, x)

        vgg_sr = get_features(sr)
        with torch.no_grad():
            vgg_hr = get_features(hr)

        if self.l1:
            loss = F.l1_loss(vgg_sr, vgg_hr)
        else:
            loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

    def state_dict(self, *args, **kwargs):
        '''
        We do not have to save the VGG model in our checkpoint
        '''
        return {}

