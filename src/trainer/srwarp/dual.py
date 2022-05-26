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

        super().__init__(*args, **kwargs)
        self.shuffle_updown = shuffle_updown
        self.w_up = w_up
        self.w_down = w_down
        self.keep_prob = keep_prob
        self.reg_prob = reg_prob
        self.adl_begin = adl_begin
        self.adl_period = adl_period
        self.adaptive_kernel = None
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['shuffle_updown'] = cfg.shuffle_updown
        kwargs['w_up'] = cfg.w_up
        kwargs['w_down'] = cfg.w_down
        kwargs['keep_prob'] = cfg.keep_prob
        kwargs['reg_prob'] = cfg.reg_prob
        kwargs['adl_begin'] = cfg.adl_begin
        kwargs['adl_period'] = cfg.adl_period
        return kwargs

    def forward(self, **samples) -> wtypes._TT:
        # Target image
        ref = samples['img']
        m = samples['m'][0].cpu()

        patch_max = 256
        w_loss = 1
        is_reg = False
        if self.training:
            r = random.random()
            if r < self.keep_prob:
                m = transform.identity()
            elif r < (self.keep_prob + self.reg_prob):
                m = transform.scaling(0.5)
                is_reg = True
            elif self.shuffle_updown and random.random() < 0.5:
                # Do up first
                # ref -> warped (SR) -> recon
                patch_size = 48
                _, _, h, w = ref.size()
                py = random.randrange(h - patch_size + 1)
                px = random.randrange(w - patch_size + 1)
                # Get a random crop to prevent memory issue.
                ref = ref[..., py:(py + patch_size), px:(px + patch_size)]
                patch_max = self.patch_max
                w_loss = self.w_down
            else:
                # Do down first
                # ref -> warped (LR) -> recon
                m = transform.inverse_3x3(m)
                w_loss = self.w_up

        m, sizes, _ = transform.compensate_matrix(ref, m)
        m_rep = transform.replicate_matrix(m, do_replicate=self.training)
        warped, mask_warped = self.pforward(ref, m_rep, sizes=sizes)

        # Get a random crop to prevent memory issue.
        warped_crop, iy, ix = crop.valid_crop(
            warped,
            self.model.fill,
            patch_max=patch_max,
            stochastic=self.training,
        )

        m_inv = transform.inverse_3x3(m)
        m_inv = transform.compensate_offset(m_inv, ix, iy)
        m_inv_rep = transform.replicate_matrix(m_inv, do_replicate=self.training)
        sizes_recon = (ref.size(-2), ref.size(-1))
        recon, mask_recon = self.pforward(warped_crop, m_inv, sizes=sizes_recon)

        if not self.training:
            recon = image_utils.quantize(recon)
            recon = mask_recon * recon + (1 - mask_recon) * self.model.fill

        ref_noisy = ref + (1 / 255) * torch.randn_like(ref)
        ref_lfl = None
        ref_adl = None
        warped_lfl = None
        warped_adl = None
        if is_reg:
            if self.adaptive_kernel is not None:
                ref_adl = ref
                warped_adl = warped
            else:
                ref_lfl = ref
                warped_lfl = warped

        loss = w_loss * self.loss(
            ref=ref,
            ref_noisy=ref_noisy,
            ref_lfl=ref_lfl,
            ref_adl=ref_adl,
            recon=recon,
            warped=warped,
            warped_crop=warped_crop,
            warped_lfl=warped_lfl,
            warped_adl=warped_adl,
            kernel_adl=self.adaptive_kernel,
            mask_warped=mask_warped,
            mask_recon=mask_recon,
            dummy_1=0,
            dummy_2=0,
            dummy_3=0,
        )
        #self.pause(count_max=10, lr=lr, sr=sr, recon=recon, reset=True)
        return loss, {'ref': ref, 'recon': recon, 'warped': warped}

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
            n_samples = 16
            scale = 2
            indices = random.choices(range(len(dataset)), k=n_samples)
            batch = [dataset[idx]['img'] for idx in indices]
            batch = torch.stack(batch, dim=0)
            #from torchvision import utils as tutils
            #tutils.save_image((batch + 1) / 2, 'example/batch.png', padding=0)
            batch = batch.cuda()
            m = transform.scaling(1 / scale)
            m, sizes, _ = transform.compensate_matrix(batch, m)
            lr, _ = self.pforward(batch, m, sizes=sizes)

            self.logger('Calculating adaptive kernel...')
            self.adaptive_kernel = filters.find_kernel(batch, lr, scale, 16)
            kernel = filters.visualize_kernel(self.adaptive_kernel)

            save_as = self.logger.get_path('kernel')
            os.makedirs(save_as, exist_ok=True)

            kernel.save(path.join(save_as, f'kernel_{epoch:0>3}.png'))

        return
