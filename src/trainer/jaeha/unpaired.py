import os
from os import path

from trainer.gan import dcgan
from model.utils import forward_utils as futils
from ops import shave
from ops import filters
from ops import transformation
from misc import gpu_utils

from PIL import Image
from scipy import io
import torch
from torch.nn import functional as F
from torchvision import utils as vutils
from torchvision.transforms import functional as TF

_parent_class = dcgan.GANTrainer


class JaehaTrainer(_parent_class):

    def __init__(
            self, *args,
            scale=2, x8=False, test_period=1, test_hr=None, kernel_gt=None,
            warmup=None, adl=False, adl_interval=None, lasso=False,
            **kwargs):

        super().__init__(*args, **kwargs)
        self.scale = scale
        self.x8 = x8
        self.test_period = test_period
        self.test_hr = test_hr
        self.kernel = None
        self.kernel_adl = None
        self.kernel_gt = kernel_gt
        self.warmup = warmup
        self.adl = adl
        self.adl_interval = adl_interval
        self.lasso = lasso

        # Whether to use the default data loss
        self.flag = True

    @staticmethod
    def get_kwargs(cfg):
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['scale'] = cfg.scale
        kwargs['x8'] = cfg.x8
        kwargs['test_period'] = cfg.test_period
        kwargs['test_hr'] = '../lab/texture.png'
        if cfg.kernel_gt is not None:
            kernel_gt = io.loadmat(cfg.kernel_gt)
            kernel_gt = torch.Tensor(kernel_gt['kernel'])
            kwargs['kernel_gt'] = gpu_utils.obj2device(kernel_gt)
        else:
            kwargs['kernel_gt'] = None

        kwargs['warmup'] = cfg.warmup
        kwargs['adl'] = cfg.adl
        kwargs['adl_interval'] = cfg.adl_interval
        kwargs['lasso'] = cfg.lasso
        return kwargs

    def forward(self, **samples):
        if self.training:
            samples = self.split_batch(**samples)
            lr_d = samples['d']['lr']
            lr_g = samples['g']['lr']
            hr_d = samples['d']['hr']
            hr = samples['g']['hr']

            down = self.pforward(hr)
            if self.get_epoch() < self.warmup:
                hr = None

            loss = self.loss(
                g=self.model,                   # Generator
                hr_d=hr_d,                      # Input HR for D
                lr_d=lr_d,                      # Real LR for D
                hr=hr,                          # Input HR for G
                down=down,                      # Generated LR
                lr_g=lr_g,                      # Real LR for G
                flag=self.flag,
                kernel=self.kernel_adl,
                kernel_gt=None,
            )
            if self.get_epoch() < self.warmup:
                loss = 0
        else:
            hr = samples['hr']
            if self.test_period > 0:
                down = self.pforward(hr)
                loss = self.loss(
                    g=None,                         # Generator
                    hr_d=None,                      # Input HR for D
                    lr_d=None,                      # Real LR for D
                    hr=hr,                          # Input HR for G
                    down=down,                      # Fake LR for G
                    lr_g=samples['lr'],             # Real LR for G
                    flag=True,
                    kernel=self.kernel_adl,
                    kernel_gt=self.kernel_gt,
                )
            else:
                if self.x8:
                    down = futils.x8_forward(self.model, hr)
                else:
                    down = self.pforward(hr)

                loss = 0

        epoch = self.get_epoch()
        save_dict = {
            '{:0>2}'.format(epoch): down,
        }
        return loss, save_dict

    def at_epoch_end(self):
        super().at_epoch_end()
        with torch.no_grad():
            hr = Image.open(self.test_hr)
            hr = TF.to_tensor(hr).unsqueeze(0)
            hr = gpu_utils.obj2device(hr)
            hr = 2 * hr - 1
            lr = self.pforward(hr)
            self.kernel = filters.find_kernel(
                hr, lr, scale=self.scale, k=20, max_patches=2**18, lasso=self.lasso,
            )
            rgb = filters.visualize_kernel(self.kernel)


        epoch = self.get_epoch()
        save_to = self.logger.get_path('kernel')
        os.makedirs(save_to, exist_ok=True)
        rgb.save(path.join(save_to, '{:0>2}.png'.format(epoch)))
        if self.kernel_gt is not None:
            rgb = filters.visualize_kernel(self.kernel_gt)
            rgb.save(path.join(save_to, 'kernel_gt.png'))

        if self.get_epoch() % self.adl_interval == 0:
            # Disable the default data loss
            self.flag = False
            # Update the ADL kernel
            self.kernel_adl = self.kernel
            self.logger('ADL kernel is updated')
