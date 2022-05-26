import random
import types
import typing

from bicubic_pytorch import core
from utils import warp
from utils import random_transform
from ops import filters
from trainer import base_trainer
from misc.gpu_utils import parallel_forward as pforward

import torch
from torch import cuda

_parent_class = base_trainer.BaseTrainer


class SuperWarpTrainer(_parent_class):

    def __init__(
            self,
            *args,
            min_scale: int=1,
            max_scale: int=4,
            kernel_reset: bool=False,
            **kwargs) -> None:

        self.kernel_reset = kernel_reset
        super().__init__(*args, **kwargs)
        step = 0.1
        if min_scale == max_scale:
            self.scales = [min_scale]
        else:
            if min_scale == 1.1:
                self.scales = [
                    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                ]
            else:
                self.scales = [
                    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                ]

        #self.scales = torch.arange(min_scale, max_scale + step, step)
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['min_scale'] = cfg.min_scale
        kwargs['max_scale'] = cfg.max_scale
        kwargs['kernel_reset'] = cfg.kernel_reset
        return kwargs

    def preprocess_state(self, state: dict) -> dict:
        pop_list = []
        if self.kernel_reset:
            for k in state.keys():
                if 'kernel_estimator' in k:
                    pop_list.append(k)

        for p in pop_list:
            state.pop(p)

        return state

    def forward(self, **samples) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        hr = samples['img']
        if self.training:
            s = random.choice(self.scales)
            if isinstance(s, torch.Tensor):
                s = s.item()

            self.rt_log_postfix = '(x{:.1f})'.format(s)
        else:
            s = 2.5

        with torch.no_grad():
            lr = 127.5 * (hr + 1)
            lr = core.imresize(
                lr,
                scale=(1 / s),
                kernel='cubic',
                range_8bit=True,
            )
            lr = lr / 127.5 - 1

        '''
        sy = hr.size(-2) / lr.size(-2)
        sx = hr.size(-1) / lr.size(-1)
        m = random_transform.scaling_transform(sx, sy=sy)
        '''
        m = random_transform.scaling_transform(s)
        m, sizes = random_transform.compensate(lr, m)
        if self.training:
            m = m.repeat(cuda.device_count(), 1)

        sr = pforward(self.model, lr, m, sizes)
        min_h = min(sizes[0], hr.size(-2))
        min_w = min(sizes[1], hr.size(-1))
        sr = sr[..., :min_h, :min_w]
        hr = hr[..., :min_h, :min_w]
        loss = self.loss(sr=sr, hr=hr)
        return loss, sr

