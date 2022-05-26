import os
from os import path
import glob
import random
import types
import typing
import pickle

from data import common

import numpy as np
import imageio
from srwarp import wtypes

import torch
import tqdm

class Unpaired(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path_hr: str=None,
            data_path_lr: str=None,
            unpaired_split: bool=False,
            bin_path: str=None,
            m_path: str=None,
            is_binary: bool=True,
            preprocessing: common.Preprocessing=None,
            train: bool=True) -> None:

        self.imgs_hr = sorted(glob.glob(path.join(data_path_hr, '*.png')))
        self.imgs_lr = sorted(glob.glob(path.join(data_path_lr, '*.png')))
        if unpaired_split:
            self.imgs_hr = self.imgs_hr[:len(self.imgs_hr) // 2]
            self.imgs_lr = self.imgs_lr[len(self.imgs_lr) // 2:]

        if is_binary:
            self.imgs_hr = self._make_binary(
                self.imgs_hr, data_path_hr, bin_path,
            )
            self.imgs_lr = self._make_binary(
                self.imgs_lr, data_path_lr, bin_path,
            )

        self.is_binary = is_binary
        self.preprocessing = preprocessing
        self.train = train
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace, train: bool=True) -> dict:
        kwargs = {
            'data_path_hr': cfg.unpaired_hr,
            'data_path_lr': cfg.unpaired_lr,
            'unpaired_split': cfg.unpaired_split,
            'bin_path': cfg.bin_path,
            'm_path': cfg.m_path,
            'preprocessing': common.make_pre(cfg),
            'train': train,
        }
        return kwargs

    def _make_binary(
            self,
            target: str,
            data_path: str,
            bin_path: str) -> typing.List[str]:

        if bin_path is None:
            bin_path = data_path

        b = path.join(bin_path, 'bin')
        if bin_path in data_path:
            bin_path = data_path.replace(bin_path, b)
        else:
            bin_path = b

        bin_ext = 'pt'
        os.makedirs(bin_path, exist_ok=True)
        for img in tqdm.tqdm(target, ncols=80):
            name = path.basename(img)
            bin_name = name.replace('png', bin_ext)
            bin_name = path.join(bin_path, bin_name)
            if not path.isfile(bin_name):
                x = imageio.imread(img)
                with open(bin_name, 'wb') as f:
                    pickle.dump(x, f)

        target = sorted(glob.glob(path.join(bin_path, '*.' + bin_ext)))
        return target

    def _crop(self, x: np.array, patch_size: int=96) -> np.array:
        h, w, _ = x.shape
        px = random.randrange(0, w - patch_size + 1)
        py = random.randrange(0, h - patch_size + 1)
        x = x[py:(py + patch_size), px:(px + patch_size)]
        return x

    def __getitem__(self, idx: int) -> wtypes._TT:
        name_hr = random.choice(self.imgs_hr)
        name_lr = random.choice(self.imgs_lr)

        if self.is_binary:
            with open(name_hr, 'rb') as f:
                img_hr = pickle.load(f)

            with open(name_lr, 'rb') as f:
                img_lr = pickle.load(f)
        else:
            img_hr = imageio.imread(name_hr)
            img_lr = imageio.imread(name_lr)

        img_hr = self._crop(img_hr)
        img_lr = self._crop(img_lr)
        img_dict = {'hr': img_hr, 'lr': img_lr}
        img_dict = self.preprocessing.set_color(**img_dict)
        img_dict = self.preprocessing.np2Tensor(**img_dict)

        img_dict['name_hr'] = path.splitext(path.basename(name_hr))[0]
        img_dict['name_lr'] = path.splitext(path.basename(name_lr))[0]
        return img_dict

    def __len__(self) -> int:
        return len(self.imgs_hr)
