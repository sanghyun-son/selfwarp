import os
from os import path
import random
import types

from ops import filters
from trainer.srwarp import warptrainer

import torch

from srwarp import transform
from srwarp import crop
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
            normalize_adl: bool=False,
            gt_hat: bool=False,
            test_sr: bool=False,
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
        self.normalize_adl = normalize_adl
        self.gt_hat = gt_hat
        self.test_sr = test_sr
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
        kwargs['normalize_adl'] = cfg.normalize_adl
        kwargs['gt_hat'] = cfg.gt_hat
        kwargs['test_sr'] = cfg.test_sr
        return kwargs

    def forward(self, **samples) -> wtypes._TT:
        if not self.training:
            if self.test_sr:
                hr = samples['hr']
                lr = samples['lr']
                m = transform.scaling(2)

                sizes = (hr.size(-2), hr.size(-1))
                sr, _ = self.model(lr, m, sizes=sizes)

                loss = self.loss(
                    ref=hr,
                    warped_keep=sr,
                    dummy_1=0,
                    dummy_2=0,
                    dummy_3=0,
                )
                ret_dict = {'sr': sr}
                return loss, ret_dict
            else:
                return super().forward(**samples)

        ref = samples['img']
        if 'img_target' in samples:
            with torch.no_grad():
                img_target = samples['img_target']
                #img_target += (1 / 255) * torch.randn_like(img_target)
        else:
            img_target = None

        m = samples['m'][0].cpu()

        r = random.random()
        is_keep = False
        is_reg = False
        if r < self.keep_prob:
            m = transform.identity()
            is_keep = True
        elif r < (self.keep_prob + self.reg_prob):
            if self.adaptive_kernel is not None:
                m = transform.scaling(1 / 2)
                is_reg = True

        m, sizes, _ = transform.compensate_matrix(ref, m)
        m_rep = transform.replicate_matrix(m, do_replicate=self.training)

        if is_keep:
            warped_keep, _ = self.pforward(ref, m_rep, sizes=sizes)
            loss = self.loss(
                ref=ref,
                warped_keep=warped_keep,
                #real=img_target,
                #fake=warped_keep,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {}
        elif is_reg:
            m_rep = transform.replicate_matrix(m, do_replicate=True)
            sizes_lr = (ref.size(-2) // 2, ref.size(-1) // 2)
            lr, _ = self.pforward(ref, m_rep, sizes=sizes_lr)

            m_sr = transform.scaling(2)
            m_sr_rep = transform.replicate_matrix(m_sr, do_replicate=True)
            sizes_sr = (ref.size(-2), ref.size(-1))
            sr, _ = self.pforward(lr, m_sr_rep, sizes=sizes_sr)

            if self.gt_hat:
                m_id = transform.identity()
                m_id, sizes_id, _ = transform.compensate_matrix(
                    img_target, m_id,
                )
                m_id_rep = transform.replicate_matrix(
                    m_id, do_replicate=self.training,
                )
                with torch.no_grad():
                    img_target, _ = self.pforward(
                        img_target, m_id_rep, sizes=sizes_id,
                    )

            loss = self.loss(
                lr=lr,
                sr=sr,
                ref=ref,
                kernel=self.adaptive_kernel,
                real=img_target,
                fake=lr,
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
            loss = self.loss(
                ref=ref,
                warped=warped,
                mask_warped=mask_warped,
                recon=recon,
                mask=mask_recon,
                #real=ref_noisy,
                #fake=warped_crop,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
            ret_dict = {'ref': ref, 'recon': recon, 'warped': warped}

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
            if self.normalize_adl:
                batch_mean = batch.mean(dim=(2, 3), keepdim=True)
                batch_std = batch.std(dim=(2, 3), keepdim=True)
                batch_z = (batch - batch_mean) / batch_std

                lr_mean = lr.mean(dim=(2, 3), keepdim=True)
                lr_std = lr.std(dim=(2, 3), keepdim=True)
                lr_z = (lr - lr_mean) / lr_std
            else:
                batch_z = batch
                lr_z = lr

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
