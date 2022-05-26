import time
import random
import types
import typing

from trainer import base_trainer
from misc import image_utils
from misc import gpu_utils

import torch
from torch import cuda
from torch import nn

from srwarp import transform
from srwarp import crop
from srwarp import warp
from srwarp import wtypes

_parent_class = base_trainer.BaseTrainer


class SRWarpTrainer(_parent_class):

    def __init__(
            self,
            *args,
            patch_max: int=64,
            resize_only: typing.Optional[str]=None,
            no_adaptive_down: bool=False,
            adversarial: bool=False,
            reset_kernel: bool=False,
            reset_sampler: bool=False,
            **kwargs) -> None:

        self.reset_kernel = reset_kernel
        self.reset_sampler = reset_sampler
        super().__init__(*args, **kwargs)

        self.patch_max = patch_max
        self.resize_only = resize_only
        self.no_adaptive_down = no_adaptive_down
        self.adversarial = adversarial
        self.time_acc = 0
        self.count = 0

        self.pretrained_srwarp = None
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['patch_max'] = cfg.patch_max
        kwargs['resize_only'] = cfg.resize_only
        kwargs['no_adaptive_down'] = cfg.no_adaptive_down
        kwargs['adversarial'] = cfg.adversarial
        kwargs['reset_kernel'] = cfg.reset_kernel
        kwargs['reset_sampler'] = cfg.reset_sampler
        return kwargs

    def preprocess_state(self, state: dict) -> dict:
        pop_list = []
        for k in state.keys():
            name = k.split('.')[0]
            if self.reset_kernel and name == 'k':
                pop_list.append(k)

        for p in pop_list:
            state.pop(p)

        return state

    @torch.no_grad()
    def get_input(
            self,
            hr: torch.Tensor,
            m_inv: torch.Tensor) -> wtypes._TT:

        # Operations on a small matrix is very slow on the GPU
        # The transformation matrix is shared in the same batch
        m_inv, sizes, _ = transform.compensate_matrix(hr, m_inv)
        m = transform.inverse_3x3(m_inv)

        # Generate the input LR image
        ignore_value = self.model.fill

        if self.pretrained_srwarp is None or not self.training:
            lr = warp.warp_by_function(
                hr,
                m,
                sizes=sizes,
                kernel_type='bicubic',
                adaptive_grid=(not self.no_adaptive_down),
                fill=ignore_value,
            )
        else:
            m_inv = transform.replicate_matrix(m_inv, do_replicate=True)
            lr, _ = gpu_utils.parallel_forward(
                self.pretrained_srwarp,
                hr,
                m_inv,
                sizes=sizes,
            )
            '''
            lr_test = warp.warp_by_function(
                hr,
                m,
                sizes=sizes,
                kernel_type='bicubic',
                adaptive_grid=(not self.no_adaptive_down),
                fill=ignore_value,
            )
            '''

        # 100x faster than the crop_largest
        if self.training:
            patch_max = self.patch_max
        else:
            patch_max = 1024

        lr_crop, iy, ix = crop.valid_crop(
            lr,
            self.model.fill,
            patch_max=patch_max,
            stochastic=self.training,
        )
        # Quantization is important
        lr_crop = image_utils.quantize(lr_crop)
        # For backward compatibility...
        m = transform.compensate_offset(m, ix, iy)
        return lr_crop, m

    def forward(self, **samples) -> wtypes._TT:
        # Target image
        hr = samples['img']
        if self.resize_only is not None and not self.training:
            # Scale: 1 ~ 3
            sx = 1 / (1.2 + 2 * random.random())
            if self.resize_only == 'iso':
                m_inv = transform.scaling(sx, sy=sx)
            elif self.resize_only == 'aniso':
                sy = 1 / (1.2 + 2 * random.random())
                m_inv = transform.scaling(sx, sy=sy)
        else:
            m_inv = random.choice(samples['m'])
            m_inv = m_inv.cpu()

        #lr_crop, m, lr_test = self.get_input(hr, m_inv)
        lr_crop, m = self.get_input(hr, m_inv)
        m = transform.replicate_matrix(m, do_replicate=self.training)
        sizes = (hr.size(-2), hr.size(-1))

        if self.debug:
            self.pause(count_max=10, lr_crop=lr_crop, hr=hr)

        #t_begin = time.time()
        #sr, mask = self.pforward(lr_crop, m, sizes=sizes)
        sr, mask = gpu_utils.parallel_forward(
            self.model,
            lr_crop,
            m,
            sizes=sizes,
        )
        '''
        self.time_acc += (time.time() - t_begin)
        self.count += 1
        if self.count <= 2:
            self.time_acc = 0
        else:
            print(self.time_acc / (self.count - 2))
        '''
        if not self.training:
            sr = image_utils.quantize(sr)
            sr = mask * sr + (1 - mask) * self.model.fill

        # For adversarial loss
        if self.training and self.adversarial:
            sr_crop, iy, ix = crop.valid_crop(
                sr,
                self.model.fill,
                patch_max=96,
                stochastic=self.training,
            )
            _, _, h, w = sr_crop.size()
            hr_crop = hr[..., iy:(iy + h), ix:(ix + w)]
        else:
            sr_crop = None
            hr_crop = None

        loss = self.loss(
            sr=sr,
            hr=hr,
            mask=mask,
            sr_crop=sr_crop,
            hr_crop=hr_crop,
            dummy_1=0,
            dummy_2=0,
            dummy_3=0,
        )

        return loss, sr

    def at_epoch_end(self) -> None:
        super().at_epoch_end()
        cuda.empty_cache()
        return

