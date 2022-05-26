import sys
from os import path
import time
import types
import typing

from trainer.srwarp import warptrainer
from misc import image_utils

import numpy as np
import torch
import cv2

from srwarp import grid
from srwarp import transform
from srwarp import warp
from srwarp import wtypes

from liif import models
from liif.test import batched_predict
sys.path.append('liif')

_parent_class = warptrainer.SRWarpTrainer


class LIIFPredictor(_parent_class):

    def __init__(
            self,
            *args,
            scale: int=4,
            naive: bool=False,
            interpolation: str='bicubic',
            is_demo: bool=False,
            **kwargs) -> None:

        super().__init__(*args, **kwargs)
        if interpolation == 'rdn':
            liif_pt = path.join('liif', 'ckpt', 'rdn-liif.pth')
        else:
            liif_pt = path.join('liif', 'ckpt', 'edsr-baseline-liif.pth')

        self.model = models.make(torch.load(liif_pt)['model'], load_sd=True)
        self.model.cuda()

        self.fill = -255
        self.is_demo = is_demo
        self.time_acc = 0
        self.count = 0
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['scale'] = cfg.scale
        kwargs['naive'] = cfg.cv2_naive
        kwargs['interpolation'] = cfg.cv2_interpolation
        kwargs['is_demo'] = ('srwarp.demo' in cfg.dtest)
        return kwargs

    def forward(self, **samples) -> typing.Tuple[torch.Tensor, float]:
        if self.is_demo:
            hr = samples['gt']
            lr_crop = samples['img']
            m = samples['m'][0].cpu()
        else:
            hr = samples['img']
            m_inv = samples['m'][0].cpu()
            lr_crop, m = self.get_input(hr, m_inv)

        m = transform.inverse_3x3(m)
        sizes = (hr.size(-2), hr.size(-1))
        grid_raw, yi = grid.get_safe_projective_grid(
            m,
            sizes,
            (lr_crop.size(-2), lr_crop.size(-1)),
        )
        grid_liif = grid.convert_coord(
            grid_raw,
            (lr_crop.size(-2), lr_crop.size(-1)),
        )

        cell = torch.ones_like(grid_liif)
        cell[:, 0] *= 2 / sizes[1]
        cell[:, 1] *= 2 / sizes[0]
        pred = batched_predict(
            self.model,
            lr_crop,
            grid_liif.unsqueeze(0),
            cell.unsqueeze(0),
            bsize=30000,
        )
        bg = pred.new_full((sizes[0] * sizes[1], 3), self.fill)
        bg[yi] = pred
        pred = bg.view(sizes[0], sizes[1], 3)
        sr = pred.permute(2, 0, 1)
        mask = torch.logical_and(sr[0] == self.fill, sr[1] == self.fill)
        mask = torch.logical_and(mask, sr[1] == self.fill)
        mask = 1 - mask.float()
        
        sr.unsqueeze_(0)
        mask.unsqueeze_(0)
        sr = mask * (sr + 1) - 1

        loss = self.loss(sr=sr, hr=hr, mask=mask)
        return loss, {'sr': sr}
