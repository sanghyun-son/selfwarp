import random

import types
import typing
from numpy.random import rand

import torch
from torch.nn import functional as F
import tqdm

from model.sr import edsr
from trainer import base_trainer
from misc import downloader
from ops import filters
from diff_jpeg import DiffJPEG
from bicubic_pytorch import core

import torch

_parent_class = base_trainer.BaseTrainer


class DegPredictor(_parent_class):

    def __init__(
            self,
            *args: typing.Optional[typing.List[typing.Any]],
            no_srfeat: bool=False,
            regression: bool=False,
            test_specific: bool=False,
            **kwargs: typing.Optional[typing.Mapping[str, typing.Any]]) -> None:

        super().__init__(*args, **kwargs)
        self.scale = 2
        pretrained = downloader.download(f'edsr-baseline-x{self.scale}')
        self.net_sr = edsr.EDSR(scale=self.scale)
        self.net_sr.cuda()
        self.net_sr.load_state_dict(pretrained['model'], strict=True)

        self.blur_sigmas = [0, 0.4, 1.0, 1.6, 2.4]
        self.noise_sigmas = [0, 5, 15, 25, 35]
        self.jpeg_qs = [30, 50, 70, 95, 100]
        self.jpeg_modules = []
        for jpeg_q in self.jpeg_qs:
            if jpeg_q == 100:
                self.jpeg_modules.append(None)
                continue

            jpeg_module = DiffJPEG.DiffJPEG(
                128, 128, differentiable=False, quality=jpeg_q,
            )
            jpeg_module.cuda()
            self.jpeg_modules.append(jpeg_module)

        self.no_srfeat = no_srfeat
        self.regression = regression
        self.test_specific = test_specific
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['no_srfeat'] = cfg.no_srfeat
        kwargs['regression'] = cfg.regression
        kwargs['test_specific'] = cfg.test_specific
        return kwargs

    def forward(
            self,
            **samples: dict) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            hr = samples['img']
            if not self.training:
                if self.test_specific:
                    img_id = samples['img_id'].item()
                    deg_id = samples['deg_id'].item()
                else:
                    deg_id = samples['idx'].item()

            img = 127.5 * (hr + 1)
            # Iterate over batch
            img_pool = []
            blur_idx_pool = []
            noise_idx_pool = []
            jpeg_idx_pool = []
            for img_sub in img:
                img_sub.unsqueeze_(0)
                if self.training:
                    blur_idx = random.randrange(len(self.blur_sigmas))
                    noise_idx = random.randrange(len(self.noise_sigmas))
                    jpeg_idx = random.randrange(len(self.jpeg_qs))
                else:
                    if deg_id == 0 and self.test_specific:
                        tqdm.tqdm.write(f'Image {img_id + 1}')
                        self.loss.reset_log()

                    blur_idx = deg_id % 5
                    noise_idx = (deg_id % 25) // 5
                    jpeg_idx = (deg_id % 125) // 25

                if self.regression and self.training:
                    blur_sigma = random.uniform(0.4, 2.4)
                else:
                    blur_sigma = self.blur_sigmas[blur_idx]
                
                if blur_sigma > 0:
                    kernel = filters.gaussian_kernel(sigma=blur_sigma)
                    img_sub = filters.downsampling(img_sub, kernel, self.scale)
                else:
                    img_sub = core.imresize(img_sub, 1 / self.scale)

                if self.regression and self.training:
                    noise_sigma = random.uniform(5, 35)
                else:
                    noise_sigma = self.noise_sigmas[noise_idx]

                if noise_sigma > 0:
                    n = noise_sigma * torch.randn_like(img_sub)
                    img_sub += n

                img_sub.clamp_(min=0, max=255)
                img_sub.round_()
                img_sub /= 255

                # Margin for blurring
                img_sub = img_sub[..., 4:-4, 4:-4]

                if self.regression and self.training:
                    jpeg_quality = random.uniform(30, 95)
                    jpeg_module = DiffJPEG.DiffJPEG(
                        128, 128, differentiable=False, quality=jpeg_quality,
                    )
                    jpeg_module.cuda()
                else:
                    jpeg_module = self.jpeg_modules[jpeg_idx]

                if jpeg_module is not None:
                    img_sub = jpeg_module(img_sub)

                img_pool.append(img_sub)
                if self.regression:
                    if not self.training:
                        blur_sigma = self.blur_sigmas[blur_idx]
                        noise_sigma = self.noise_sigmas[noise_idx]
                        jpeg_quality = self.jpeg_qs[jpeg_idx]

                    # Normalize
                    blur_idx_pool.append(blur_sigma - 1.4)
                    noise_idx_pool.append((noise_sigma - 20) / 15)
                    jpeg_idx_pool.append((jpeg_quality - 62.5) / 32.5)

                else:
                    blur_idx_pool.append(blur_idx)
                    noise_idx_pool.append(noise_idx)
                    jpeg_idx_pool.append(jpeg_idx)

            img_lr = torch.cat(img_pool, dim=0)
            img_lr = 2 * img_lr - 1

            blur_label = torch.tensor(blur_idx_pool)
            blur_label = blur_label.cuda()
            noise_label = torch.tensor(noise_idx_pool)
            noise_label = noise_label.cuda()
            jpeg_label = torch.tensor(jpeg_idx_pool)
            jpeg_label = jpeg_label.cuda()

            if self.no_srfeat:
                x = img_lr
            else:
                sr = self.net_sr(img_lr)
                sr_unshuffle = F.pixel_unshuffle(sr, self.scale)
                x = torch.cat((sr_unshuffle, img_lr), dim=1)

        blur_pred, noise_pred, jpeg_pred = self.pforward(x)
        if self.regression:
            blur_pred.squeeze_(-1)
            noise_pred.squeeze_(-1)
            jpeg_pred.squeeze_(-1)
            if not self.training:
                blur_pred.clamp_(min=-1, max=1)
                noise_pred.clamp_(min=-1, max=1)
                jpeg_pred.clamp_(min=-1, max=1)

        '''
        print('Label')
        print(blur_label.item(), noise_label.item(), jpeg_label.item())
        print('Prediction')
        print(blur_pred.item(), noise_pred.item(), jpeg_pred.item())
        '''
        loss = self.loss(
            blur_pred=blur_pred,
            noise_pred=noise_pred,
            jpeg_pred=jpeg_pred,
            blur_label=blur_label,
            noise_label=noise_label,
            jpeg_label=jpeg_label,
        )
        ret_dict = {}
        if not self.training and self.test_specific:
            if deg_id == 124:
                self.loss.print_tree(self.logger)
                ret_dict = {'lr': img_lr, 'hr': hr}

        return loss, ret_dict
