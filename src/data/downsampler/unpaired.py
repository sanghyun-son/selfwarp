from os import path
import glob
import random
from data import common

import numpy as np
import imageio

from torch import utils


class Unpaired(utils.data.Dataset):
    '''
    Unpaired super-resolution data class

    Args:
    '''

    def __init__(
            self,
            unpaired_hr: str=None, stat_hr: str=None,
            unpaired_lr: str=None, stat_lr: str=None,
            augmentation=None, preprocessing=None):

        super().__init__()
        self.list_hr = glob.glob(path.join(unpaired_hr, '*.png'))
        self.list_lr = glob.glob(path.join(unpaired_lr, '*.png'))
        if stat_hr is not None:
            self.stat_hr = np.load(stat_hr)
        else:
            self.stat_hr = None

        if stat_lr is not None:
            self.stat_lr = np.load(stat_lr)
        else:
            self.stat_lr = None

        self.augmentation = augmentation
        self.pre = preprocessing

        print('HR: {} samples'.format(len(self.list_hr)))
        print('LR: {} samples'.format(len(self.list_lr)))

    @staticmethod
    def get_kwargs(cfg, train: bool=True) -> dict:
        # Baseline arguments for self-learning
        kwargs = {}
        if cfg.unpaired_hr is None:
            kwargs['unpaired_hr'] = path.join(
                '..', 'dataset', 'DIV2K', 'DIV2K_train_HR'
            )
        else:
            kwargs['unpaired_hr'] = cfg.unpaired_hr

        if cfg.unpaired_lr is None:
            kwargs['unpaired_lr'] = kwargs['unpaired_hr']
        else:
            kwargs['unpaired_lr'] = cfg.unpaired_lr

        kwargs['stat_hr'] = cfg.stat_hr
        kwargs['stat_lr'] = cfg.stat_lr
        kwargs['augmentation'] = cfg.augmentation
        kwargs['preprocessing'] = common.make_pre(cfg)
        return kwargs

    def __getitem__(self, idx: int) -> dict:
        hr = random.choice(self.list_hr)
        hr = imageio.imread(hr)
        lr = self.list_lr[idx]
        lr = imageio.imread(lr)
        # Mixup for lr images
        if 'm' in self.augmentation:
            lr = lr.astype(np.float32)
            lr = self.pre.augment(lr=lr)['lr']
            lr_2 = random.choice(self.list_lr)
            lr_2 = imageio.imread(lr_2).astype(np.float32)
            lr_2 = self.pre.augment(lr=lr_2)['lr']
            alpha = random.random()
            #alpha = random.choice([0, 0.25, 0.5, 0.75, 1])
            lr_mixed = alpha * lr + (1 - alpha) * lr_2
            lr = lr_mixed.round().astype(np.uint8)

        hr = self.get_patch(hr, is_hr=True)
        lr = self.get_patch(lr, is_hr=False)

        img_dict = {'hr': hr, 'lr': lr}
        img_dict = self.pre.set_color(**img_dict)
        # Linear transformation to match image statistics
        '''
        if self.stat_hr is not None and self.stat_lr is not None:
            hr = img_dict['hr'].astype(np.float32)
            if self.pre.n_colors == 3:
                pass
            else:
                ym_source = self.stat_hr[3]
                ys_source = self.stat_hr[7]
                ym_target = self.stat_lr[3]
                ys_target = self.stat_lr[7]

                hr = (hr - ym_source) / ys_source
                hr = ym_target + (hr * ys_target)
            
            hr = hr.round().clip(0, 255)
            hr = hr.astype(np.uint8)
            img_dict['hr'] = hr
        '''
        img_dict = self.pre.np2Tensor(**img_dict)
        return img_dict

    def __len__(self) -> int:
        return len(self.list_lr)

    def get_patch(self, img: np.array, is_hr: bool=True) -> np.array:
        patch = self.pre.patch
        if is_hr:
            patch *= self.pre.scale

        ret_dict = self.pre.get_patch(img=img, patch=patch)
        ret_dict = self.pre.augment(**ret_dict)
        ret = ret_dict['img']
        return ret

