import math
import random
import types
import typing

from trainer import base_trainer
from misc import image_utils

import torch
from torch import cuda

from srwarp import transform
from srwarp import crop
from srwarp import resize
from srwarp import wtypes

_parent_class = base_trainer.BaseTrainer


class SuperWarpTrainer(_parent_class):

    def __init__(
            self,
            *args,
            patch_max: int=96,
            scale: float=4,
            scale_min: float=1.1,
            scale_max: float=4,
            shuffle_updown: bool=False,
            w_up: float=1,
            w_down: float=1,
            reset_kernel: bool=False,
            reset_sampler: bool=False,
            **kwargs) -> None:

        self.reset_kernel = reset_kernel
        self.reset_sampler = reset_sampler
        super().__init__(*args, **kwargs)
        self.patch_max = patch_max
        self.margin = 4

        self.scale = scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shuffle_updown = shuffle_updown
        self.w_up = w_up
        self.w_down = w_down
        if scale_min == 1.1:
            self.scales = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5,
            ]
            '''
            self.scales = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
            ]
            '''
            #3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
        else:
            if scale_min == scale_max:
                self.scales = [scale_min]

        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['patch_max'] = cfg.patch_max
        kwargs['scale'] = cfg.scale
        kwargs['scale_min'] = cfg.scale_min
        kwargs['scale_max'] = cfg.scale_max
        kwargs['shuffle_updown'] = cfg.shuffle_updown
        kwargs['w_up'] = cfg.w_up
        kwargs['w_down'] = cfg.w_down
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

    def _crop_with_margin(
            self,
            x: torch.Tensor,
            sizes: typing.Tuple[int, int]) -> typing.Tuple[torch.Tensor, int, int]:

        iy = self.margin
        h_margin = sizes[0] - 2 * self.margin - self.patch_max
        if h_margin > 0:
            py = random.randrange(0, h_margin + 1)
            iy += py
            slice_y = slice(py, py + self.patch_max)
        else:
            slice_y = slice(self.margin, -self.margin)

        ix = self.margin
        w_margin = sizes[1] - 2 * self.margin - self.patch_max
        if w_margin > 0:
            px = random.randrange(0, w_margin + 1)
            ix += px
            slice_x = slice(px, px + self.patch_max)
        else:
            slice_x = slice(self.margin, -self.margin)

        x_crop = x[..., slice_y, slice_x]
        return x_crop, iy, ix

    def forward(self, **samples) -> wtypes._TT:
        # Determine resizing factor
        if 'img' in samples:
            if self.training:
                x = samples['img']
                gt = None
                '''
                sx = random.choice(self.scales)
                sy = random.choice(self.scales)

                if random.random() < 0.5:
                    sx = 1 / sx

                if random.random() < 0.5:
                    sy = 1 / sy

                '''

                sx = 1 / random.choice(self.scales)
                sy = 1 / random.choice(self.scales)
                self.rt_log_postfix = f'(x{sx:.2f}, {sy:.2f})'
            else:
                gt = samples['img']
                x = resize.imresize(gt, 1 / self.scale)
                sx = self.scale
                sy = self.scale

            m = transform.scaling(sx, sy=sy)
            # m would not be changed.
            # Only for calculating the output dimension.
            m, sizes, _ = transform.compensate_matrix(x, m)
        else:
            x = samples['lr']
            gt = samples['hr']
            m = transform.scaling(self.scale)
            _, _, h, w = x.size()
            sizes = (math.ceil(h * self.scale), math.ceil(w * self.scale))

        m = transform.replicate_matrix(m, do_replicate=self.training)
        y, mask = self.pforward(x, m, sizes=sizes)

        if self.training:
            y_crop, iy, ix = self._crop_with_margin(y, sizes)
            m_inv = transform.inverse_3x3(m[:3])
            m_inv = transform.compensate_offset(m_inv, ix, iy)
            m_inv = transform.replicate_matrix(m_inv, do_replicate=self.training)
            sizes_recon = (x.size(-2), x.size(-1))

            x_recon, mask_recon = self.pforward(
                y_crop, m_inv, sizes=sizes_recon,
            )
            loss = self.loss(
                recon=x_recon,
                lr=x,
                sr=y,
                sr_crop=y_crop,
                mask_sr=mask,
                mask_recon=mask_recon,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
        else:
            y = image_utils.quantize(y)
            y = mask * y + (1 - mask) * self.model.fill

            min_h = min(sizes[0], gt.size(-2))
            min_w = min(sizes[1], gt.size(-1))
            y = y[..., :min_h, :min_w]
            gt = gt[..., :min_h, :min_w]
            mask = mask[..., :min_h, :min_w]

            loss = self.loss(
                recon=y,
                lr=gt,
                sr=y,
                sr_crop=None,
                mask_sr=mask,
                mask_recon=mask,
            )

        # To make it compatible with the Meta-SR evaluation
        if 'img' not in samples:
            sizes = (int(h * self.scale), int(w * self.scale))

        #self.pause(count_max=10, x=x, y=y_crop, x_recon=x_recon)
        return loss, {'lr': x, 'sr': y}
