from data import common as dcommon
from model import common
from model.custom import masking

import torch
from torch import nn
from torch.nn import utils


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
            self, depth=8, n_colors=3, width=64, norm='batch',
            n_classes=8, ignore_bg=False, mask_scale=16, template='plain',
            early_fork=0):

        super().__init__()

        # Basic block configurations
        self.n_classes = n_classes
        self.mask_scale = mask_scale
        self.early_fork = early_fork

        # Concatenate semantic masks to input
        self.cat = 'cat' in template
        self.cat_multi = self.cat and ('multi' in template)

        # Shared feature extractor
        self.features = nn.ModuleList()
        m_seq = []
        # We do not consider background pixels in this case
        self.ignore_bg = ignore_bg
        if ignore_bg:
            n_classes -= 1

        in_channels = n_colors
        out_channels = width
        scale = 1
        stride = 1
        # Class-specific discriminator
        cls = [[] for _ in range(n_classes)]
        for i in range(depth):
            # cat and the first layer or
            # cat_multi
            if stride == 1 and ((self.cat and i == 0) or self.cat_multi):
                args = [in_channels + n_classes, out_channels, 3]
            else:
                args = [in_channels, out_channels, 3]

            kwargs = {'stride': stride, 'norm': norm, 'act': 'lrelu'}
            m_seq.append(common.BasicBlock(*args, **kwargs))
            if stride == 2:
                self.features.append(nn.Sequential(*m_seq))
                m_seq = []
            # Temporarily disabled
            '''
            if early_fork == 0 and scale <= mask_scale:
                m_seq.append(common.BasicBlock(*args, **kwargs))
            elif early_fork > 0 and i < depth - early_fork:
                m_seq.append(common.BasicBlock(*args, **kwargs))
            else:
                for c in cls:
                    c.append(common.BasicBlock(*args, **kwargs))
            '''
            # Calculate the next stride
            # This reduces feature resolution for every even iterations
            stride = 2 - (i % 2)
            if stride == 2:
                scale *= 2
            in_channels = out_channels
            if i % 2 == 1:
                out_channels *= 2

        #self.features = nn.Sequential(*m)
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
                c.append(nn.Conv2d(in_channels, in_channels, 1, padding=0))
                c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            c.append(nn.Conv2d(in_channels, 1, 1, padding=0))
            self.multi_cls.append(nn.Sequential(*c))

        # Normal distribution
        # Any improvements?
        common.init_gans(self)

        # Assign random masks
        self.random = 'random' in template

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'depth': cfg.depth_sub,
            'n_colors': cfg.n_colors,
            'width': cfg.width_sub,
            'n_classes': cfg.n_classes,
            'ignore_bg': cfg.ignore_bg,
            'mask_scale': cfg.mask_scale,
            'template': cfg.template,
            'early_fork': cfg.dis_early_fork,
        }
        return kwargs

    def forward(self, x, mask):
        '''
        Args:
            x (torch.Tensor):           B x 3 x H x W in [-1, 1]
            mask (torch.BoolTensor):    B x n_classes x H x W in {0, 1}

        Return:
            (torch.Tensor): M real values
        '''
        # Remove background masks
        if self.ignore_bg:
            mask = mask[1:]

        if self.random:
            # Check wheter a certain pixel has class information
            is_class = mask.any(dim=1, keepdim=True)
            # B x 1 x H x W
            shuffle = torch.randint_like(
                is_class,
                low=0,
                high=self.n_classes,
                dtype=torch.uint8,
            )
            mask_shuffled = [(shuffle == i) for i in range(self.n_classes)]
            # B x n_classes x H x W
            mask_shuffled = torch.cat(mask_shuffled, dim=1)
            mask = (mask_shuffled & is_class)

        # Output: B x C x (H / s) x (W / s)
        for i, m in enumerate(self.features):
            if (self.cat and i == 0) or self.cat_multi:
                scale = 2**i
                with torch.no_grad():
                    if scale > 1:
                        mask_resize = dcommon.resize_mask(mask, scale)
                    else:
                        mask_resize = mask

                    mask_real = 2 * mask_resize.float() - 1

                x = torch.cat((x, mask_real), dim=1)

            x = m(x)

        with torch.no_grad():
            mask = dcommon.resize_mask(mask, self.mask_scale)
            mask_split = mask.float().split(1, dim=1)
            divider = sum(mask_split).clamp(min=1)

        label = [cls(x) for cls in self.multi_cls]
        label = sum(l * m for l, m in zip(label, mask_split)) / divider
        return label

