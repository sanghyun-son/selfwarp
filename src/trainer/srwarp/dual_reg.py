import os
from os import path
import random
import types
import typing

from ops import filters
from utils import random_cut
from trainer import base_trainer
from trainer.srwarp import warptrainer
from misc import image_utils

import tqdm
import torch
from torch import cuda
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
            adl_begin: int=10,
            adl_period: int=10,
            **kwargs) -> None:

        self.adaptive_kernel = None
        super().__init__(*args, **kwargs)
        self.shuffle_updown = shuffle_updown
        self.w_up = w_up
        self.w_down = w_down
        self.keep_prob = keep_prob
        self.reg_prob = reg_prob
        self.adl_begin = adl_begin
        self.adl_period = adl_period
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['shuffle_updown'] = cfg.shuffle_updown
        kwargs['w_up'] = cfg.w_up
        kwargs['w_down'] = cfg.w_down
        kwargs['keep_prob'] = cfg.keep_prob
        kwargs['reg_prob'] = cfg.reg_prob
        #kwargs['adl_begin'] = cfg.adl_begin
        #kwargs['adl_period'] = cfg.adl_period
        return kwargs

    def forward(self, **samples) -> wtypes._TT:
        if not self.training:
            super().forward(**samples)

        # Target image
        ref = samples['img']
        ref_noisy = ref + (1 / 255) * torch.randn_like(ref)
        m = samples['m'][0].cpu()

        is_keep = False
        is_reg = False
        if self.training:
            r = random.random()
            if r < self.keep_prob:
                m = transform.identity()
                is_keep = True
            elif r < (self.keep_prob + self.reg_prob):
                scale = 2
                m_comp = transform.scaling(scale)
                m_1 = transform.scaling(1 / scale)
                m_2 = torch.matmul(m, m_comp)
                is_reg = True

        m, sizes, _ = transform.compensate_matrix(ref, m)
        m_rep = transform.replicate_matrix(m, do_replicate=self.training)

        if is_keep:
            warped_keep, _ = self.pforward(ref, m_rep, sizes=sizes)
            loss = self.loss(
                ref=ref,
                warped_keep=warped_keep,
                real=ref_noisy,
                fake=warped_keep,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {}
        elif is_reg:
            m_1, sizes_1, _ = transform.compensate_matrix(ref, m_1)
            m_1_rep = transform.replicate_matrix(
                m_1, do_replicate=self.training,
            )
            warped_1, _ = self.pforward(ref, m_1_rep, sizes=sizes_1)

            m_2, sizes_2, _ = transform.compensate_matrix(warped_1, m_2)
            m_2_rep = transform.replicate_matrix(
                m_2, do_replicate=self.training,
            )
            warped_2, mask = self.pforward(warped_1, m_2_rep, sizes=sizes_2)

            warped_direct, _ = self.pforward(ref, m_rep, sizes=sizes)
            warped_direct_crop, _, _ = crop.valid_crop(
                warped_direct,
                self.model.fill,
                patch_max=96,
                stochastic=self.training,
            )
            loss = self.loss(
                warped=warped_2,
                warped_direct=warped_direct,
                warped_down=warped_1,
                ref=ref,
                kernel=self.adaptive_kernel,
                mask=mask,
                real=ref_noisy,
                fake=warped_direct_crop,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {}
        else:
            warped, mask_warped = self.pforward(ref, m_rep, sizes=sizes)
            warped_crop, iy, ix = crop.valid_crop(
                warped,
                self.model.fill,
                patch_max=96,
                stochastic=self.training,
            )

            m_inv = transform.inverse_3x3(m)
            m_inv = transform.compensate_offset(m_inv, ix, iy)
            m_inv_rep = transform.replicate_matrix(
                m_inv, do_replicate=self.training,
            )
            sizes_recon = (ref.size(-2), ref.size(-1))
            recon, mask_recon = self.pforward(
                warped_crop, m_inv_rep, sizes=sizes_recon,
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
            loss = self.loss(
                ref=ref,
                warped=warped,
                mask_warped=mask_warped,
                recon=recon,
                mask=mask_recon,
                real=ref_noisy,
                fake=warped_crop,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {'ref': ref, 'recon': recon, 'warped': warped, 'warped_cv': warped_cv}

        return loss, ret_dict

    def at_epoch_begin(self) -> None:
        super().at_epoch_begin()
        epoch = self.get_epoch()
        for v in self.loader_train.dataset.datasets:
            v.epoch = epoch

        for v in self.loader_eval.values():
            v.dataset.epoch = epoch

        return

    @torch.no_grad()
    def at_epoch_end(self) -> None:
        super().at_epoch_end()
        if self.adl_period < 1:
            return

        epoch = self.get_epoch()
        if epoch < self.adl_begin:
            return

        if epoch % self.adl_period == 0:
            dataset = self.loader_train.dataset.datasets[0]
            n_samples = 32
            scale = 2
            indices = random.choices(range(len(dataset)), k=n_samples)
            batch = [dataset[idx]['img'] for idx in indices]
            batch = torch.stack(batch, dim=0)
            #from torchvision import utils as tutils
            #tutils.save_image((batch + 1) / 2, 'example/batch.png', padding=0)
            batch = batch.cuda()
            m = transform.scaling(1 / scale)
            m, sizes, _ = transform.compensate_matrix(batch, m)
            m = transform.replicate_matrix(m, do_replicate=True)
            lr, _ = self.pforward(batch, m, sizes=sizes)

            self.logger('Calculating adaptive kernel...')
            # Normalized ADL
            batch_mean = batch.mean(dim=(2, 3), keepdim=True)
            batch_std = batch.std(dim=(2, 3), keepdim=True)
            batch_z = (batch - batch_mean) / batch_std

            lr_mean = lr.mean(dim=(2, 3), keepdim=True)
            lr_std = lr.std(dim=(2, 3), keepdim=True)
            lr_z = (lr - lr_mean) / lr_std

            self.adaptive_kernel = filters.find_kernel(batch_z, lr_z, scale, 16)
            kernel = filters.visualize_kernel(self.adaptive_kernel)

            save_as = self.logger.get_path('kernel')
            os.makedirs(save_as, exist_ok=True)

            kernel.save(path.join(save_as, f'kernel_{epoch:0>3}.png'))

        return

    def get_state(self) -> dict:
        state_dict = super().get_state()
        state_dict['adaptive_kernel'] = self.adaptive_kernel
        return state_dict

    def load_additional_state(self, state: dict) -> None:
        self.adaptive_kernel = state['adaptive_kernel']
        return
