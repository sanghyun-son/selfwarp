from os import path
import glob
import random
import collections

from data import common
from misc import mask_utils

import imageio
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision.transforms import functional


class CelebAMask(data.Dataset):
    _PARTS_MAP = collections.OrderedDict(
        skin=0,
        neck=0,
        nose=1,
        mouth=2,
        u_lip=2,
        l_lip=2,
        hair=3,
        r_brow=4,
        l_brow=4,
        r_eye=5,
        l_eye=5,
        r_ear=6,
        l_ear=6,
        cloth=7,
        hat=7,
        eye_g=8,  # Eyeglasses
        neck_l=9,  # Necklaces
        ear_r=9,  # Earrings
    )
    _N_CLASSES = 10

    def __init__(
            self, dpath=None, scale=8, hr_size=512, train=True, **kwargs):

        n_eval = 500
        n_test = 500

        lr_size = hr_size // scale
        self.hr_size = hr_size
        self.train = train

        root = path.join(dpath, 'CelebAMask')
        path_hr = path.join(root, 'CelebA-img-{}px'.format(hr_size))
        path_lr = path.join(root, 'CelebA-img-{}px'.format(lr_size))
        self.list_hr = sorted(glob.glob(path.join(path_hr, '*.png')))
        self.list_lr = sorted(glob.glob(path.join(path_lr, '*.png')))

        mask_dirname = 'CelebA-mask-{}px-single'.format(hr_size)
        self.mask = path.join(root, mask_dirname)

        n_split = n_eval + n_test
        if train:
            self.list_hr = self.list_hr[:-n_split]
            self.list_lr = self.list_lr[:-n_split]
        else:
            self.list_hr = self.list_hr[-n_split:-n_test]
            self.list_lr = self.list_lr[-n_split:-n_test]

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = {
            'scale': cfg.scale,
            'dpath': cfg.dpath,
            'train': train
        }
        return kwargs

    def load_mask(self, name, flip=False):
        make_full = lambda: np.full((self.hr_size, self.hr_size), False)
        mask_list = [make_full() for _ in range(CelebAMask._N_CLASSES)]
        for k, v in CelebAMask._PARTS_MAP.items():
            part_name = path.join(self.mask, '{:0>5}_{}.png'.format(name, k))
            # Only if mask exists
            if path.isfile(part_name):
                mask = imageio.imread(part_name)
                mask = (mask != 0)
                mask_list[v] = (mask | mask_list[v])

        # Will become (C, H, W)
        mask = np.stack(mask_list)
        # Horizontal flip
        if flip:
            mask = np.flip(mask, axis=2)

        mask = np.ascontiguousarray(mask)
        mask = torch.from_numpy(mask)
        return mask
        
    def __getitem__(self, idx):
        flip = self.train and (random.random() < 0.5)
        def transform(x):
            x = Image.open(x)
            # Random augmentation (hflip only)
            if flip:
                x = functional.hflip(x)

            x = functional.to_tensor(x)
            # PIL Image range is 0 ~ 1
            x = 2 * x - 1
            return x

        img_lr = self.list_lr[idx]
        img_hr = self.list_hr[idx]

        name = path.splitext(path.basename(img_hr))[0]

        img_dict = {'lr': img_lr, 'hr': img_hr}
        img_dict = {k: transform(v) for k, v in img_dict.items()}

        mask = self.load_mask(name, flip=flip)
        img_dict['mask'] = mask
        img_dict['mask_coded'] = mask_utils.mask_coding(mask)
        if not self.train:
            img_dict['name'] = name

        return img_dict

    def __len__(self):
        return len(self.list_hr)


