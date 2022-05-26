import os
from os import path
import glob
from PIL import Image

from data import common

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional


class CelebA(data.Dataset):

    def __init__(self, dpath, pre, n_z=100, batch_size=128, train=True):
        self.n_z = n_z
        self.pre = pre
        self.train = train

        path_img = path.join(dpath, 'CelebA', 'img_align_celeba_png')
        if train:
            self.images = glob.glob(path.join(path_img, '*.png'))
        else:
            self.noise = torch.randn(batch_size, n_z, 1, 1)

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = {
            'dpath': cfg.dpath,
            'pre': common.make_pre(cfg),
            'n_z': cfg.n_z,
            'batch_size': cfg.batch_size_eval,
            'train': train,
        }
        return kwargs

    def __getitem__(self, idx):
        if self.train:
            img = Image.open(self.images[idx])
            compose = transforms.Compose([
                transforms.CenterCrop(min(*img.size)),
                transforms.Resize(self.pre.patch, interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                # PIL Image range is 0 ~ 1
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img = compose(img)
            z = torch.randn(self.n_z, 1, 1)
            return {'z': z, 'img': img}
        else:
            z = self.noise[idx]
            return {'z': z, 'name': 'generated'}

    def __len__(self):
        if self.train:
            return len(self.images)
        else:
            return len(self.noise)

