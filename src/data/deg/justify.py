from os import path
import random

from data import common
from data.sr import dataclass

_parent_class = dataclass.SRData

class DIV2K(_parent_class):
    '''
    DIV2K mean:
        R: 0.4488
        G: 0.4371
        B: 0.4040
    '''

    def __init__(
            self,
            *args,
            dbegin=1,
            dend=100,
            custom_length: int=None,
            tval=False,
            specific_deg: int=-1,
            **kwargs):

        self.dbegin = dbegin
        self.dend = dend
        self.custom_length = custom_length
        self.tval = tval
        super(DIV2K, self).__init__(*args, **kwargs)

        with open('deg_info.txt', 'r') as f:
            lines = f.read().splitlines()

        self.deg_list = []
        for line in lines:
            name, kernel_sigma, noise_sigma, jpeg_q = line.split(' ')
            deg_dict = {
                'name': name,
                'kernel_sigma': float(kernel_sigma),
                'noise_sigma': float(noise_sigma),
                'jpeg_q': int(jpeg_q),
            }
            self.deg_list.append(deg_dict)

        self.specific_deg = specific_deg
        return

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = _parent_class.get_kwargs(cfg, train=train)
        if train:
            dbegin = None
            dend = None
        else:
            kwargs['tval'] = ('t' in cfg.val_range)
            val_range = cfg.val_range.replace('t', '')
            dbegin, dend = [int(x) for x in val_range.split('-')]

        kwargs['dbegin'] = dbegin
        kwargs['dend'] = dend
        kwargs['custom_length'] = cfg.custom_length

        if train:
            kwargs['specific_deg'] = cfg.specific_deg_train
        else:
            kwargs['specific_deg'] = cfg.specific_deg_eval

        return kwargs

    def __len__(self) -> int:
        if self.custom_length is not None:
            return self.custom_length

        return super().__len__()

    def __getitem__(self, idx: int) -> dict:
        ret_dict = super().__getitem__(idx)
        if self.specific_deg != -1:
            ret_dict['deg'] = self.deg_list[self.specific_deg]
        else:
            ret_dict['deg'] = random.choice(self.deg_list)

        return ret_dict        

    def scan(self, target_path):
        filelist = super(DIV2K, self).scan(target_path)
        if self.dbegin and self.dend:
            filelist = filelist[self.dbegin - 1:self.dend]

        return filelist

    def apath(self):
        return path.join(self.dpath, 'DIV2K')

    def get_path(self, degradation, scale):
        if not isinstance(scale, int) and scale.is_integer():
            scale = int(scale)

        if not (self.train or self.tval):
            split = 'valid'
        else:
            split = 'train'

        path_hr = path.join(self.apath(), 'DIV2K_{}_HR'.format(split))

        if scale == 1:
            path_lr = path_hr
        else:
            if 'jit' in degradation:
                path_lr = degradation
            else:
                path_lr = 'DIV2K_{}_LR_{}'.format(split, degradation)
                path_lr = path.join(self.apath(), path_lr, 'X{}'.format(scale))

        return {'lr': path_lr, 'hr': path_hr}

