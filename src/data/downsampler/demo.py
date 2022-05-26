from os import path
import glob

from data import common

import numpy as np
from PIL import Image

from torch import utils


class Demo(utils.data.Dataset):

    def __init__(self, dpath: str, scale: int, preprocessing=None):
        self.hr_list = sorted(glob.glob(path.join(dpath, '*.png')))
        self.scale = scale
        self.pre = preprocessing

    @staticmethod
    def get_kwargs(cfg, train: bool=False) -> dict:
        kwargs = {
            'dpath': cfg.dpath,
            'scale': cfg.scale,
            'preprocessing': common.make_pre(cfg),
        }
        return kwargs

    def __getitem__(self, idx):
        name = self.hr_list[idx]
        hr = Image.open(name)
        w, h = hr.size
        ww = w // self.scale
        hh = h // self.scale
        lr = hr.resize((ww, hh), resample=Image.BICUBIC)
        img_dict = {'hr': hr, 'lr': lr}
        img_dict = {k: np.array(v) for k, v in img_dict.items()}
        img_dict = self.pre.set_color(**img_dict)
        img_dict = self.pre.np2Tensor(**img_dict)
        img_dict['name'] = path.splitext(path.basename(name))[0]
        return img_dict

    def __len__(self) -> int:
        return len(self.hr_list)

