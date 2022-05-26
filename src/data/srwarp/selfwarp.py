import os
from os import path
import glob
import random
import types
import typing
import pickle

from data import common
from data.srwarp import warpdata
from utils import random_cut

import numpy as np
import imageio
from srwarp import pyramid
from srwarp import wtypes

import torch
import tqdm


class DualWarpData(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path: str=None,
            target_path: str=None,
            unpaired_split: bool=False,
            bin_path: str=None,
            m_path: str=None,
            patch_size: int=128,
            patch_size_test: int=96,
            is_binary: bool=True,
            grow: str='none',
            preprocessing: common.Preprocessing=None,
            train: bool=True) -> None:

        self.__imgs = sorted(glob.glob(path.join(data_path, '*.png')))
        self.__imgs_target = sorted(glob.glob(path.join(target_path, '*.png')))

        if unpaired_split:
            len_imgs =  len(self.__imgs) // 2
            len_target = len(self.__imgs_target) // 2
            self.__imgs = self.__imgs[:len_imgs]
            self.__imgs_target = self.__imgs_target[len_target:]

        self.__patch_size = patch_size
        self.__patch_size_test = patch_size_test
        self.__p = pyramid.Pyramid(patch_size, patch_size)
        self.__grow = grow
        self.__epoch = -1

        if train:
            self.__ms = None
        else:
            self.__ms = torch.load(m_path)

        if is_binary:
            self.__imgs = self.make_binary(self.__imgs, bin_path, data_path)
            self.__imgs_target = self.make_binary(
                self.__imgs_target, bin_path, data_path,
            )

        self.__is_binary = True
        self.__preprocessing = preprocessing
        self.__train = train
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace, train: bool=True) -> dict:
        kwargs = {
            'target_path': cfg.unpaired_lr,
            'unpaired_split': cfg.unpaired_split,
            'bin_path': cfg.bin_path,
            'm_path': cfg.m_path,
            'patch_size': cfg.patch,
            'patch_size_test': cfg.patch_test,
            'grow': cfg.grow,
            'preprocessing': common.make_pre(cfg),
            'train': train,
        }
        if train:
            kwargs['data_path'] = cfg.data_path_train
        else:
            kwargs['data_path'] = cfg.data_path_test

        return kwargs

    def make_binary(self, target: str, bin_path: str, data_path: str) -> None:
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

        bin_list = sorted(glob.glob(path.join(bin_path, '*.' + bin_ext)))
        return bin_list

    def random_crop(self, img: np.array, is_target: bool=False) -> np.array:
        if is_target:
            patch_size = 48
        else:
            patch_size = self.__patch_size

        h, w, _ = img.shape
        if self.__train:
            px = random.randrange(0, w - patch_size + 1)
            py = random.randrange(0, h - patch_size + 1)
        else:
            px = (w - patch_size) // 2
            py = (h - patch_size) // 2

        img = img[py:(py + patch_size), px:(px + patch_size)]
        return img

    def __getitem__(self, idx: int) -> wtypes._TT:
        name = self.__imgs[idx]
        name_target = random.choice(self.__imgs_target)
        if self.__is_binary:
            with open(name, 'rb') as f:
                img = pickle.load(f)

            with open(name_target, 'rb') as f:
                img_target = pickle.load(f)
        else:
            img = imageio.imread(name)
            img_target = imageio.imread(name_target)

        img = self.random_crop(img)
        img_target = self.random_crop(img_target, is_target=True)

        img_dict_target = self.__preprocessing.augment(img_target=img_target)
        if self.__train:
            img_dict = self.__preprocessing.augment(img=img)
        else:
            img_dict = {'img': img}

        img_dict = self.__preprocessing.set_color(**img_dict)
        img_dict = self.__preprocessing.np2Tensor(**img_dict)

        img_dict_target = self.__preprocessing.set_color(**img_dict_target)
        img_dict_target = self.__preprocessing.np2Tensor(**img_dict_target)

        if self.__train:
            if self.__grow == 'none':
                z_center = 1
            elif self.__grow == 'inc':
                z_center = 1 + 1 * min(self.__epoch / 300, 1)
            elif self.__grow == 'dec':
                z_center = 1 - 0.25 * min(self.__epoch / 300, 1)

            z_std = 0.2
            z_min = z_center - z_std
            z_max = z_center + z_std
            m = self.__p.get_random_m(
                z_min=z_min,
                z_max=z_max,
                phi_min=0.1,
                phi_max=0.25,
                random_aspect=False,
            )
        else:
            '''
            if self.__epoch >= 200:
                difficulty = 'large'
            elif self.__epoch >= 100:
                difficulty = 'medium'
            else:
            '''
            difficulty = 'medium'
            m = self.__ms[self.__patch_size_test][difficulty][idx % 100]

        img_dict['m'] = m.double()
        img_dict['name'] = path.splitext(path.basename(name))[0]
        ret_dict = {**img_dict, **img_dict_target}
        return ret_dict

    def __len__(self) -> int:
        return len(self.__imgs)

    @property
    def epoch(self) -> int:
        return self.__epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.__epoch = value
        return
