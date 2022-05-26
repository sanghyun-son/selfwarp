import os
from os import path
import glob
import random
import types
import typing
import pickle

from data import common

import imageio

import torch
import tqdm

class ImageData(torch.utils.data.Dataset):

    def __init__(
            self,
            data_path: str=None,
            preprocessing: typing.Optional[common.Preprocessing]=None,
            train: bool=True) -> None:

        self.imgs = glob.glob(path.join(data_path, '*.png'))
        self.is_binary = False
        self.preprocessing = preprocessing
        self.train = train
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace, train=True) -> dict:
        kwargs = {
            'preprocessing': common.make_pre(cfg),
            'train': train,
        }
        if train:
            kwargs['data_path'] = cfg.data_path_train
        else:
            kwargs['data_path'] = cfg.data_path_test

        return kwargs

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        name = self.imgs[idx]
        if self.is_binary:
            with open(name, 'rb') as f:
                img = pickle.load(f)
        else:
            img = imageio.imread(name)

        if self.train:
            img_dict = self.preprocessing.get_patch(img=img)
            img_dict = self.preprocessing.augment(**img_dict)
        else:
            img_dict = {'img': img}

        img_dict = self.preprocessing.set_color(**img_dict)
        img_dict = self.preprocessing.np2Tensor(**img_dict)
        img_dict['name'] = path.splitext(path.basename(name))[0]
        return img_dict

    def __len__(self) -> int:
        return len(self.imgs)
