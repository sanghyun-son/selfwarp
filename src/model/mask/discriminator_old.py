from model import common
from model.custom import masking

import torch
from torch import nn
from torch.nn import utils

def model_class(*args, **kwargs):
    return common.model_class(Discriminator, *args, **kwargs)


class Discriminator(nn.Module):
    '''
    Multi-class discriminator with semantic masks.
    Spectral normalization is applied by default.

    Args:
        depth (int, optional):
        n_colors (int, optional):
        width (int, optional):
        n_classes (int, optional): The number of classes (including background).
        norm ('batch' or 'layer' or 'instance' or 'group', optional):
            Type of the normalization layers.

    Note:
        The basic structures is from Ledig et al.,
        "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
        See https://arxiv.org/pdf/1609.04802.pdf for more detail.
    '''

    def __init__(
            self, depth=8, n_colors=3, width=64,
            n_classes=8, ignore_bg=False, mask_scale=16, norm='batch'):

        super(Discriminator, self).__init__()

        # Basic block configurations
        in_channels = n_colors
        out_channels = width
        scale = 1
        stride = 1
        # Shared feature extractor
        m = []
        # We do not consider background pixels in this case
        self.ignore_bg = ignore_bg
        if ignore_bg:
            n_classes -= 1
        # Class-specific discriminator
        cls = [[] for _ in range(n_classes)]
        for i in range(depth):
            args = [in_channels, out_channels, 3]
            kwargs = {'stride': stride, 'norm': norm, 'act': 'lrelu'}
            if scale <= mask_scale:
                m.append(common.BasicBlock(*args, **kwargs))
            else:
                for c in cls:
                    c.append(common.BasicBlock(*args, **kwargs))
        
            # Reduce resolution every even iteration
            stride = 2 - (i % 2)
            if stride == 2:
                scale *= 2
            in_channels = out_channels
            # We do not double output channels at the last layer
            if i % 2 == 1 and i < depth - 1:
                out_channels *= 2

        self.features = nn.Sequential(*m)
        '''
        PatchGAN style

        Note:
            From Isola et al.,
            "Image-to-Image Translation with Conditional Adversarial Networks"
            (pix2pix)
            See https://arxiv.org/pdf/1611.07004.pdf for more detail.
        '''
        self.multi_cls = nn.ModuleList()
        for c in cls:
            c.append(nn.Conv2d(out_channels, 1, 1, padding=0))
            self.multi_cls.append(nn.Sequential(*c))

        # Normal distribution
        # Any improvements?
        common.init_gans(self)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        return {
            'depth': cfg.depth_sub,
            'width': cfg.width_sub,
            'n_classes': cfg.n_classes,
            'ignore_bg': cfg.ignore_bg,
            'mask_scale': cfg.mask_scale,
        }

    def forward(self, x, mask):
        '''
        Args:
            x (torch.Tensor):           B x 3 x H x W in [-1, 1]
            mask (torch.BoolTensor):    B x n_classes x (H / s) x (W / s) in {0, 1}

        Return:
            (torch.Tensor): M real values
        '''
        features = self.features(x)             # B x C x (H / s) x (W / s)
        mask_split = mask.split(1, dim=1)       # B x C x (H / s) x (W / s)
        # Remove background masks
        if self.ignore_bg:
            mask_split = mask_split[1:]

        label = [masking.masking(features, m) for m in mask_split]
        label = [cls(l) for cls, l in zip(self.multi_cls, label)]
        #features = [c(x) for c in self.multi_cls]
        #features = [c[m] for c, m in zip(features, mask_split)]
        # Class-wise label results.
        # B x n_classes x (H / S) x (W / S)
        label = torch.cat(label, dim=1)
        return label

