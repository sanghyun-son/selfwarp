from os import path
import random
import typing

import numpy as np

from data.sr import dataclass

_parent_class = dataclass.SRData


class DIV2KUnpaired(_parent_class):

    def __init__(self, *args, batch_size: int=16, **kwargs) -> None:
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
        return

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = _parent_class.get_kwargs(cfg, train=train)
        kwargs['batch_size'] = cfg.batch_size
        return kwargs

    def __len__(self) -> int:
        return self.batch_size * 1000

    def apath(self) -> str:
        return path.join(self.dpath, 'DIV2K')

    def get_path(
            self,
            degradation: str='bicubic',
            scale: int=2) -> typing.Mapping[str, str]:

        scale = int(scale)
        split = 'train'

        path_hr = path.join(self.apath(), f'DIV2K_{split}_HR')
        path_lr = path.join(
            self.apath(), f'DIV2K_{split}_LR_{degradation}', f'X{scale}',
        )
        ret_dict = {'lr': path_lr, 'hr': path_hr}
        return ret_dict

    def get_idx(self, idx: int) -> typing.Mapping[str, int]:
        idx_dict = {
            'lr': random.randrange(len(self.data['lr'])),
            'hr': random.randrange(len(self.data['hr'])),
        }
        return idx_dict

    def get_patch(self, **kwargs) -> typing.Mapping[str, np.array]:
        img_dict = self.pre.modcrop(**kwargs)
        if self.train:
            lr = img_dict['lr']
            hr = img_dict['hr']

            lr_patch = self.pre.get_patch(lr=lr)
            lr_patch = self.pre.augment(**lr_patch)

            hr_patch = self.pre.get_patch(hr=hr)
            hr_patch = self.pre.augment(**hr_patch)

            img_dict = {'lr': lr_patch['lr'], 'hr': hr_patch['hr']}

        return img_dict