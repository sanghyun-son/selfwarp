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

import imageio
from srwarp import pyramid
from srwarp import wtypes

import torch
import tqdm


class DualWarpData(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path: str=None,
            bin_path: str=None,
            m_path: str=None,
            patch_size: int=128,
            patch_size_test: int=96,
            is_binary: bool=True,
            preprocessing: common.Preprocessing=None,
            train: bool=True) -> None:

        self.__imgs = sorted(glob.glob(path.join(data_path, '*.png')))
        self.__patch_size = patch_size
        self.__patch_size_test = patch_size_test
        self.__p = pyramid.Pyramid(patch_size, patch_size)
        self.__epoch = -1

        if train:
            self.__ms = None
        else:
            self.__ms = torch.load(m_path)
            if 'input_size' in self.__ms:
                self.__patch_size = self.__ms['input_size']
            elif 'patch_size' in self.__ms:
                self.__patch_size = self.__ms['patch_size']

        if is_binary:
            if bin_path is None:
                bin_path = data_path

            b = path.join(bin_path, 'bin')
            if bin_path in data_path:
                bin_path = data_path.replace(bin_path, b)
            else:
                bin_path = b

            bin_ext = 'pt'
            os.makedirs(bin_path, exist_ok=True)
            for img in tqdm.tqdm(self.__imgs, ncols=80):
                name = path.basename(img)
                bin_name = name.replace('png', bin_ext)
                bin_name = path.join(bin_path, bin_name)
                if not path.isfile(bin_name):
                    x = imageio.imread(img)
                    with open(bin_name, 'wb') as f:
                        pickle.dump(x, f)

            self.__imgs = sorted(glob.glob(path.join(bin_path, '*.' + bin_ext)))

        self.__is_binary = True
        self.__preprocessing = preprocessing
        self.__train = train
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace, train: bool=True) -> dict:
        kwargs = {
            'bin_path': cfg.bin_path,
            'patch_size': cfg.patch,
            'patch_size_test': cfg.patch_test,
            'preprocessing': common.make_pre(cfg),
            'train': train,
        }
        if train:
            kwargs['data_path'] = cfg.data_path_train
            kwargs['m_path'] = cfg.m_path
        else:
            kwargs['data_path'] = cfg.data_path_test
            kwargs['m_path'] = cfg.m_path_eval

        return kwargs

    def __getitem__(self, idx: int) -> wtypes._TT:
        name = self.__imgs[idx]
        if self.__is_binary:
            with open(name, 'rb') as f:
                img = pickle.load(f)
        else:
            img = imageio.imread(name)

        h, w, _ = img.shape
        if self.__train:
            px = random.randrange(0, w - self.__patch_size + 1)
            py = random.randrange(0, h - self.__patch_size + 1)
        else:
            px = (w - self.__patch_size) // 2
            py = (h - self.__patch_size) // 2

        img = img[py:(py + self.__patch_size), px:(px + self.__patch_size)]

        if self.__train:
            img_dict = self.__preprocessing.augment(img=img)
        else:
            img_dict = {'img': img}

        img_dict = self.__preprocessing.set_color(**img_dict)
        img_dict = self.__preprocessing.np2Tensor(**img_dict)

        if self.__train:
            #z_center = 1 + 2 * min(self.__epoch / 400, 1)
            z_center = 1
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
            m_inter = self.__p.get_random_m(
                z_min=z_min,
                z_max=z_max,
                phi_min=0.1,
                phi_max=0.25,
                random_aspect=False,
            )
            img_dict['m_inter'] = m_inter.double()
        else:
            m = self.__ms['eval'][idx]

        img_dict['m'] = m.double()
        img_dict['name'] = path.splitext(path.basename(name))[0]
        return img_dict

    def __len__(self) -> int:
        return len(self.__imgs)

    @property
    def epoch(self) -> int:
        return self.__epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.__epoch = value
        return
