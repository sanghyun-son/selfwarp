import types
import typing

from trainer.gan import dcgan
from model import loader as mloader
from model.utils import forward_utils as futils
from misc.gpu_utils import parallel_forward as pforward

import torch
from torch import nn
_parent_class = dcgan.GANTrainer


class SRTrainer(_parent_class):

    def __init__(
            self, *args,
            x8: bool=False,
            quads: bool=False,
            ref: nn.Module=None,
            **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.x8 = x8
        self.quads = quads

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['x8'] = cfg.x8
        kwargs['quads'] = cfg.quads
        return kwargs

    def forward(
            self, **samples: dict) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        if self.training:
            samples = self.split_batch(**samples)
            lr_d = samples['d']['lr']
            lr_g = samples['g']['lr']
            hr_d = samples['d']['hr']
            hr = samples['g']['hr']

            psr, sr = pforward(self.model, lr_g)
            loss = self.loss(
                lr=lr_g,
                g=self.model,
                lr_d=lr_d,
                hr_d=hr_d,
                psr=psr,
                sr=sr,
                hr=hr,
            )
        else:
            lr = samples['lr']
            psr, sr = pforward(self.model, lr)

            loss = self.loss(
                lr=lr,
                g=None,
                lr_d=None,
                hr_d=None,
                psr=psr,
                sr=sr,
                hr=samples['hr'],
            )

        epoch = self.get_epoch()
        if epoch == 1 or epoch % 10 == 0:
            img = {'{:0>3}'.format(epoch): sr, 'p{:0>3}'.format(epoch): psr}
        else:
            img = sr

        return loss, img
