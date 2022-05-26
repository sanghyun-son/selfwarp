import math
import types
import typing

from bicubic_pytorch import core

from trainer.sr import base
from misc.gpu_utils import parallel_forward as pforward
from model.utils import forward_utils as futils

import torch

_parent_class = base.SRTrainer


class MultiScaleSRTrainer(_parent_class):

    def __init__(self, *args, max_scale: int=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_scale = max_scale
        self.n_scales = int(math.log2(max_scale))

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> typing.Mapping[str, object]:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['max_scale'] = cfg.max_scale
        return kwargs

    def forward(self, **samples):
        samples = self.split_batch(**samples)
        hr = samples['g']['hr']
        with torch.no_grad():
            lrs = [hr]
            for i in range(self.n_scales):
                child = core.imresize(
                    lrs[-1],
                    0.5,
                    kernel='cubic',
                    antialiasing=True,
                )
                lrs.append(child)

            pyramid_dict = {}
            for i in range(len(lrs)):
                scale = 2 ** (self.n_scales - i)
                if self.training:
                    lrs[i] = lrs[i][..., scale:-scale, scale:-scale]

                if i < len(lrs) - 1:
                    pyramid_dict['hr_x{}'.format(scale)] = lrs[i]

        if self.training:
            srs = pforward(self.model, lrs[-1])
        else:
            if self.quads:
                srs = futils.quad_forward(self.model, lrs[-1])
            elif self.x8:
                srs = futils.x8_forward(self.model, lrs[-1])
            else:
                srs = pforward(self.model, lrs[-1])

        for i, sr in enumerate(srs):
            pyramid_dict['sr_x{}'.format(2 ** (i + 1))] = sr

        # For the adversarial training purpose
        # Just put dummy values for easy debugging
        pyramid_dict['g'] = -987654321
        pyramid_dict['z_d'] = -987654321
        pyramid_dict['real_d'] = -987654321
        loss = self.loss(**pyramid_dict)

        result_dict = {
            key: val for key, val in pyramid_dict.items() if 'sr' in key
        }
        return loss, result_dict
