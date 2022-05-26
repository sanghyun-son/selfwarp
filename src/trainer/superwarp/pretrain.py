import types

import typing

from utils import warp
from utils import random_transform
from ops import filters
from trainer import base_trainer
from misc.gpu_utils import parallel_forward as pforward

import torch
from torch import cuda

_parent_class = base_trainer.BaseTrainer


class SuperWarpTrainer(_parent_class):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        return kwargs

    def forward(self, **samples) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        hr = samples['hr']
        lr = samples['lr']
        s = hr.size(-1) // lr.size(-1)
        m = random_transform.scaling_transform(s)

        if self.training:
            m = m.repeat(cuda.device_count(), 1)

        sr = pforward(self.model, lr, m, (hr.size(-2), hr.size(-1)))
        loss = self.loss(sr=sr, hr=hr)
        return loss, sr

