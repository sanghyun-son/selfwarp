import os

from model import common
from model.custom import masking
from segmentation.unet_feature import unet

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
            n_classes=8, ignore_bg=False, mask_scale=1, norm='batch',
            # Jaerin
            seg_model=None, seg_n_feat=32):

        super(Discriminator, self).__init__()

        # We do not consider background pixels in this case
        self.n_classes = n_classes
        self.mask_scale = mask_scale
        self.ignore_bg = ignore_bg
        if ignore_bg:
            n_classes -= 1

        # Load segmentation network as shared feature extractor
        self.segmentation = unet(
            n_feat=seg_n_feat,
            n_classes=self.n_classes,
            as_feature_ext=True,
        )
        if seg_model is not None:
            if not os.path.splitext(seg_model)[1]:
                seg_model += '.pt'
            weights = torch.load(seg_model)
            self.segmentation.load_state_dict(weights)

        # Freeze segmentation network
        for param in self.segmentation.parameters():
            param.requires_grad = False

        # Class-specific discriminator
        cls = [[] for _ in range(n_classes)]

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
            # Make classifiers deep
            if 'deep' in template:
                c.append(nn.Conv2d(seg_n_feat, seg_n_feat, 1, padding=0))
                c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            c.append(nn.Conv2d(seg_n_feat, 1, 1, padding=0))
            self.multi_cls.append(nn.Sequential(*c))

        # Initialize parameters with normal distribution
        # NOTE: Segmentation networks should not be affected
        # TODO Any improvements?
        model_to_init = self.multi_cls
        common.init_gans(model_to_init)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        return {
            'depth': cfg.depth_sub,
            'width': cfg.width_sub,
            'n_classes': cfg.n_classes,
            'ignore_bg': cfg.ignore_bg,
            'mask_scale': cfg.mask_scale,
            # Jaerin
            'seg_model': cfg.dis_seg_model,
            'seg_n_feat': cfg.dis_seg_n_feat,
        }

    def forward(self, x, mask):
        '''
        Args:
            x (torch.Tensor):           B x 3 x H x W in [-1, 1]
            mask (torch.BoolTensor):    B x n_classes x (H / s) x (W / s) in {0, 1}

        Return:
            (torch.Tensor): M real values
        '''
        features = self.segmentation(x)         # B x C x (H / s) x (W / s)
        # Masks are assumed to have the same size as images
        mask_split = mask.split(1, dim=1)       # B x C x (H / s) x (W / s)
        # Remove background masks
        if self.ignore_bg:
            mask_split = mask_split[1:]

        label = [cls(features) for cls in self.multi_cls]
        z = zip(label, mask_split)
        label = [masking.masking_c(l, m, self.n_classes) for l, m in z]
        #features = [c(x) for c in self.multi_cls]
        #features = [c[m] for c, m in zip(features, mask_split)]
        # Class-wise label results.
        # B x n_classes x (H / S) x (W / S)
        label = torch.cat(label, dim=1)
        return label

