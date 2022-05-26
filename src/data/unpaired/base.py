import os
from os import path
import glob
import random
import pickle
import typing

from data import common

import tqdm
import numpy as np
import imageio

from torch import utils


class Unpaired(utils.data.Dataset):
    '''
    Unpaired super-resolution data class
    '''

    def __init__(
            self,
            unpaired_hr: str=None,
            unpaired_lr: str=None,
            unpaired_split: bool=False,
            bin_path: str=None,
            augmentation=None,
            preprocessing=None,
            is_binary: bool=True) -> None:

        super().__init__()
        self.list_hr = sorted(glob.glob(path.join(unpaired_hr, '*.png')))
        self.list_lr = sorted(glob.glob(path.join(unpaired_lr, '*.png')))

        if unpaired_split:
            len_hr_half = len(self.list_hr) // 2
            len_lr_half = len(self.list_lr) // 2
            self.list_hr = self.list_hr[:len_hr_half]
            self.list_lr = self.list_lr[len_lr_half:]

        if is_binary:
            self.list_hr = self.make_binary(self.list_hr, unpaired_hr, bin_path)
            self.list_lr = self.make_binary(self.list_lr, unpaired_lr, bin_path)

        self.is_binary = is_binary
        self.augmentation = augmentation
        self.pre = preprocessing

        print('HR: {} samples'.format(len(self.list_hr)))
        print('LR: {} samples'.format(len(self.list_lr)))
        return

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

        kwargs['unpaired_split'] = cfg.unpaired_split
        kwargs['bin_path'] = cfg.bin_path
        kwargs['augmentation'] = cfg.augmentation
        kwargs['preprocessing'] = common.make_pre(cfg)
        return kwargs

    def make_binary(
            self,
            img_list: list,
            data_path: str,
            bin_path: typing.Optional[str]) -> list:

        if bin_path is None:
            bin_path = data_path

        b = path.join(bin_path, 'bin')
        if bin_path in data_path:
            bin_path = data_path.replace(bin_path, b)
        else:
            bin_path = b

        print(bin_path)
        bin_ext = 'pt'
        os.makedirs(bin_path, exist_ok=True)
        for img in tqdm.tqdm(img_list, ncols=80):
            name = path.basename(img)
            bin_name = name.replace('png', bin_ext)
            bin_name = path.join(bin_path, bin_name)
            if not path.isfile(bin_name):
                x = imageio.imread(img)
                with open(bin_name, 'wb') as f:
                    pickle.dump(x, f)

        list_new = sorted(glob.glob(path.join(bin_path, '*.' + bin_ext)))
        return list_new

    def __getitem__(self, idx: int) -> dict:
        name_hr = random.choice(self.list_hr)
        #lr = self.list_lr[idx]
        name_lr = random.choice(self.list_lr)
        if self.is_binary:
            with open(name_hr, 'rb') as f:
                hr = pickle.load(f)

            with open(name_lr, 'rb') as f:
                lr = pickle.load(f)
        else:
            hr = imageio.imread(name_hr)
            lr = imageio.imread(name_lr)

        hr = self.get_patch(hr, is_hr=True)
        lr = self.get_patch(lr, is_hr=False)

        img_dict = {'hr': hr, 'lr': lr}
        img_dict = self.pre.set_color(**img_dict)
        img_dict = self.pre.np2Tensor(**img_dict)
        return img_dict

    def __len__(self) -> int:
        #return len(self.list_lr)
        # temp.
        return 800

    def get_patch(self, img: np.array, is_hr: bool=True) -> np.array:
        patch = self.pre.patch
        if is_hr:
            patch = int(self.pre.scale * patch)

        ret_dict = self.pre.get_patch(img=img, patch=patch)
        ret_dict = self.pre.augment(**ret_dict)
        ret = ret_dict['img']
        return ret

