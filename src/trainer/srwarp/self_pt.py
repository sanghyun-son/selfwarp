import os
from os import path
import random
import types
import typing

from ops import filters
from trainer.srwarp import warptrainer
from model import loader

import torch
from torch import nn

from srwarp import transform
from srwarp import crop
from srwarp import warp
from srwarp import wtypes

_parent_class = warptrainer.SRWarpTrainer


class DualWarpTrainer(_parent_class):

    def __init__(
            self,
            *args,
            shuffle_updown: bool=False,
            w_up: float=1,
            w_down: float=1,
            keep_prob: float=0,
            reg_prob: float=0.1,
            w_self: float=1,
            w_id: float=1,
            w_dual: float=1,
            pretrained_srwarp: nn.Module,
            **kwargs) -> None:

        self.adaptive_kernel = None
        super().__init__(*args, **kwargs)
        self.shuffle_updown = shuffle_updown
        self.w_up = w_up
        self.w_down = w_down
        self.keep_prob = keep_prob
        self.reg_prob = reg_prob

        self.w_self = w_self
        self.w_id = w_id
        self.w_dual = w_dual

        self.pretrained_srwarp = pretrained_srwarp
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['shuffle_updown'] = cfg.shuffle_updown
        kwargs['w_up'] = cfg.w_up
        kwargs['w_down'] = cfg.w_down
        kwargs['keep_prob'] = cfg.keep_prob
        kwargs['reg_prob'] = cfg.reg_prob

        kwargs['w_self'] = cfg.w_self
        kwargs['w_id'] = cfg.w_id
        kwargs['w_dual'] = cfg.w_dual

        net_pt = loader.get_model(cfg)
        #print(net_pt)
        state = torch.load(cfg.pretrained_srwarp, map_location='cpu')
        model_state = state['model']
        net_pt.load_state_dict(model_state)

        kwargs['pretrained_srwarp'] = net_pt
        return kwargs

    def forward(self, **samples) -> wtypes._TT:
        #if not self.training:
        return super().forward(**samples)

        # Target image
        ref = samples['img']
        m_orig = samples['m'][0].cpu()
        if 'm_inter' in samples:
            m_inter = samples['m_inter'][0].cpu()

        is_keep = False
        is_reg = False

        r = random.random()
        if r < self.keep_prob:
            m_orig = transform.identity()
            is_keep = True
        elif r < (self.keep_prob + self.reg_prob):
            is_reg = True

        m, sizes, offsets = transform.compensate_matrix(ref, m_orig)

        # Identity transform
        if is_keep:
            warped_keep, _ = self.pforward(ref, m, sizes=sizes)
            loss = self.w_id * self.loss(
                ref=ref,
                warped_keep=warped_keep,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {}
        # Dual-path regularization
        elif is_reg:
            # First transformation matrix
            m_1, sizes_1, _ = transform.compensate_matrix(ref, m_inter)
            warped_1, _ = self.pforward(ref, m_1, sizes=sizes_1)
            '''
            # For debugging
            warped_1 = warp.warp_by_function(
                ref,
                m_1,
                f_inverse=False,
                sizes=sizes_1,
                fill=self.model.fill,
            )
            '''

            warped_1_crop, iy, ix = crop.valid_crop(
                warped_1,
                self.model.fill,
                patch_max=192,
                stochastic=self.training,
            )

            m_2 = transform.inverse_3x3(m_1)
            m_2 = transform.compensate_offset(m_2, ix, iy)
            m_2 = torch.matmul(m_orig, m_2)
            _, sizes_2, offsets_2 = transform.compensate_matrix(warped_1_crop, m_2)
            m_2 = transform.compensate_offset(
                m_2,
                offsets[1],
                offsets[0],
                offset_first=False,
            )

            offsets_new = (
                offsets_2[0] - offsets[0],
                offsets_2[1] - offsets[1],
            )
            sizes_new = (
                max(sizes[0], sizes_2[0]) + 2 * abs(offsets_new[0]),
                max(sizes[1], sizes_2[1]) + 2 * abs(offsets_new[1]),
            )
            warped_2, mask_2 = self.pforward(warped_1_crop, m_2, sizes=sizes_new)
            warped_direct, mask_direct = self.pforward(ref, m, sizes=sizes_new)

            '''
            warped_2_cv = warp.warp_by_function(
                warped_1_crop,
                m_2,
                f_inverse=False,
                sizes=sizes_new,
                fill=self.model.fill,
            )
            warped_direct_cv = warp.warp_by_function(
                ref,
                m,
                f_inverse=False,
                sizes=sizes_new,
                fill=self.model.fill,
            )
            '''
            mask_union = mask_2 * mask_direct
            warped_2 *= mask_union
            warped_direct *= mask_union
            '''
            warped_direct_crop, _, _ = crop.valid_crop(
                warped_direct,
                self.model.fill,
                patch_max=96,
                stochastic=self.training,
            )
            '''
            '''
            self.pause(
                count_max=1,
                ref=ref,
                warped_1=warped_1,
                warped_2=warped_2,
                warped_direct=warped_direct,
                warped_2_cv=warped_2_cv,
                warped_direct_cv=warped_direct_cv,
            )
            '''
            loss = self.w_dual * self.loss(
                warped=warped_2,
                warped_direct=warped_direct,
                mask=mask_union,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {}
        else:
            warped, mask_warped = self.pforward(ref, m, sizes=sizes)
            warped_crop, iy, ix = crop.valid_crop(
                warped,
                self.model.fill,
                patch_max=96,
                stochastic=self.training,
            )

            m_inv = transform.inverse_3x3(m)
            m_inv = transform.compensate_offset(m_inv, ix, iy)
            sizes_recon = (ref.size(-2), ref.size(-1))
            recon, mask_recon = self.pforward(
                warped_crop, m_inv, sizes=sizes_recon,
            )
            '''
            warped_cv = warp.warp_by_function(
                ref,
                m,
                f_inverse=False,
                sizes=sizes,
                fill=self.model.fill,
            )
            '''
            loss = self.w_self * self.loss(
                ref=ref,
                warped=warped,
                recon=recon,
                mask_warped=mask_warped,
                mask=mask_recon,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {
                'ref': ref,
                'recon': recon,
                'warped': warped,
                #'warped_cv': warped_cv,
            }

        return loss, ret_dict

    def at_epoch_begin(self) -> None:
        super().at_epoch_begin()
        epoch = self.get_epoch()
        for v in self.loader_train.dataset.datasets:
            v.epoch = epoch

        for v in self.loader_eval.values():
            v.dataset.epoch = epoch

        return

    def get_state(self) -> dict:
        state_dict = super().get_state()
        state_dict['adaptive_kernel'] = self.adaptive_kernel
        return state_dict

    def load_additional_state(self, state: dict) -> None:
        self.adaptive_kernel = state['adaptive_kernel']
        return

    def pforward(
            self,
            x: torch.Tensor,
            m: torch.Tensor,
            sizes: typing.Tuple[int, int]) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        m = transform.replicate_matrix(m, do_replicate=self.training)
        return super().pforward(x, m, sizes=sizes)
