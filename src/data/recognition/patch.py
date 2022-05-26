from os import path
import glob
import random
from data import common

import numpy as np
import imageio

import torch
from torch import utils


class Unpaired(utils.data.Dataset):
    '''
    Unpaired super-resolution data class

    Args:
    '''

    def __init__(
            self,
            unpaired_hr: str=None,
            unpaired_lr: str=None,
            preprocessing=None,
            train: bool=True):

        super().__init__()
        list_hr = glob.glob(path.join(unpaired_hr, '*.png'))
        list_lr = glob.glob(path.join(unpaired_lr, '*.png'))
        self.pre = preprocessing

        min_len = min(len(list_hr), len(list_lr))
        cross_val = 5
        n_train = min_len * (cross_val - 1) // cross_val

        if train:
            list_hr = list_hr[:n_train]
            list_lr = list_lr[:n_train]
        else:
            list_hr = list_hr[n_train:min_len]
            list_lr = list_lr[n_train:min_len]

        label_hr = [0 for _ in list_hr]
        label_lr = [1 for _ in list_lr]

        self.list_img = [*list_hr, *list_lr]
        self.list_label = torch.LongTensor([*label_hr, *label_lr])

    @staticmethod
    def get_kwargs(cfg, train: bool=True) -> dict:
        # Baseline arguments for self-learning
        kwargs = {'train': train}
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

        kwargs['preprocessing'] = common.make_pre(cfg)
        return kwargs

    def __getitem__(self, idx: int) -> dict:
        img = self.list_img[idx]
        img = imageio.imread(img)
        img = self.get_patch(img)
        ret_dict = {'img': img}
        ret_dict = self.pre.set_color(**ret_dict)
        ret_dict = self.pre.np2Tensor(**ret_dict)
        ret_dict['label'] = self.list_label[idx]
        return ret_dict

    def __len__(self) -> int:
        return len(self.list_img)

    def get_patch(self, img: np.array) -> np.array:
        ret_dict = self.pre.get_patch(img=img)
        ret_dict = self.pre.augment(**ret_dict)
        ret = ret_dict['img']
        return ret

