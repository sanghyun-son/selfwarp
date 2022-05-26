import os
from os import path
import glob
import pickle
import tqdm
import imageio

from config import get_config
from data import common
from misc import mask_utils

import numpy as np
import torch
from torch import utils
from torch.distributions import categorical


class MixedSR(utils.data.Dataset):
    '''
    Bicubic SR train pairs from various datasets.

    Args:
        dpath (str):
        scale (int):
        mask_scale (int):
        n_classes (int):
        preprocessing ():
        div2k (bool): Use the DIV2K dataset.
        imagenet (bool): Use the ImageNet validation dataset.
        ost (bool): Use the OutdoorScene dataset.
        flickr (bool): Use the Flickr2K dataset.
        no_mask (bool): Do not use the mask information.
        raw (bool):
        train (bool):
    '''
    def __init__(
            self, dpath=None, scale=4, mask_scale=16, n_classes=8,
            preprocessing=None,
            div2k=True, imagenet=False, ost=False, flickr=False, no_mask=False,
            raw=False, use_patch=False, train=True, **kwargs):

        # Evaluate with OST dataset
        if not train:
            div2k = False
            imagenet = False
            ost = True
            flickr = False

        self.scale = scale
        self.mask_scale = mask_scale
        self.n_classes = n_classes
        self.pre = preprocessing
        self.no_mask = no_mask
        self.train = train

        scan_dict = {}
        patch_dir = 'patch_importance'
        if div2k:
            if use_patch:
                scan_dict['DIV2K'] = path.join('DIV2K', patch_dir)
            else:
                scan_dict['DIV2K'] = 'DIV2K'
        if imagenet:
            scan_dict['ImageNet'] = 'ILSVRC2012'
        if ost:
            prefix = 'OutdoorSeg'
            if train:
                scan_dict['OST'] = path.join(prefix, 'OutdoorSeg')
            else:
                scan_dict['OST'] = path.join(prefix, 'OutdoorSceneTest300')
        if flickr:
            if use_patch:
                scan_dict['Flickr2K'] = path.join('Flickr2K', patch_dir)
            else:
                scan_dict['Flickr2K'] = 'Flickr2K'

        # Data will be cached in dictionary format
        self.data = {'hr': [], 'lr': [], 'mask': []}
        sample_size = []
        for k, v in scan_dict.items():
            print('Scan {}'.format(v))
            scan_dir = path.join(dpath, v)

            # Search for GT segmentation masks
            mask_dir = 'annotations'
            mask_dir_cropped = path.join(scan_dir, 'crop', mask_dir)
            if path.isdir(mask_dir_cropped):
                mask_dir = mask_dir_cropped
            else:
                mask_dir = path.join(scan_dir, mask_dir)

            if not no_mask and train and path.isdir(mask_dir):
                if k == 'OST':
                    hr_ext = 'jpg'
                else:
                    hr_ext = 'png'

                mask_list = glob.glob(path.join(mask_dir, '*.png'))
                map_key = lambda x: path.basename(x).replace('png', hr_ext)
                mask_dict = {map_key(m): m for m in mask_list}
            else:
                mask_dict = None

            with open(path.join(scan_dir, 'available.txt'), 'r') as f:
                pair_list = f.read().splitlines()
                print('\t{} pairs'.format(len(pair_list)))
                for pair in tqdm.tqdm(pair_list, ncols=80):
                    hr, lr, h, w = pair.split(' ')
                    hr_name = path.join(scan_dir, hr)
                    lr_name = path.join(scan_dir, lr)
                    # Supports bin loading
                    if k in ('DIV2K', 'Flickr2K') and not use_patch:
                        # DIV2K has a special naming rule
                        if k == 'DIV2K':
                            #lr = lr.replace('.png', 'x{}.png'.format(scale))
                            lr_name = path.join(scan_dir, lr)
                        if not raw:
                            hr = hr.replace('png', 'pt')
                            lr = lr.replace('png', 'pt')
                            load_from = scan_dir.replace(
                                path.join(dpath, v),
                                path.join(dpath, v, 'bin'),
                            )
                            hr_name = path.join(load_from, hr)
                            lr_name = path.join(load_from, lr)
                            for name in (hr_name, lr_name):
                                if not path.isfile(name):
                                    self.make_binary(name)

                    self.data['hr'].append(hr_name)
                    self.data['lr'].append(lr_name)
                    # Count the number of available patches
                    sample_size.append((int(h), int(w)))
                    # Check if mask exists
                    name = path.basename(hr)
                    name = name.replace('.pt', '.png')
                    if mask_dict is not None and name in mask_dict:
                        self.data['mask'].append(mask_dict[name])
                    else:
                        self.data['mask'].append(None)

        n_masks = sum(mask is not None for mask in self.data['mask'])
        print('Total {} pairs'.format(len(self.data['hr'])))
        print('\t{} samples with mask'.format(n_masks))
        # Larger images will have more chances to be sampled
        self.set_sampler(sample_size)

    @staticmethod
    def get_kwargs(cfg, train=True):
        parse_list = [
            'dpath', 'scale', 'n_classes', 'raw', 'use_patch',
        ]
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs_add = {
            'preprocessing': common.make_pre(cfg),
            'div2k': cfg.use_div2k,
            'ost': cfg.use_ost,
            'imagenet': cfg.use_imagenet,
            'flickr': cfg.use_flickr,
            'no_mask': cfg.no_mask,
            'train': train,
        }
        kwargs = {**kwargs, **kwargs_add}
        return kwargs

    def set_sampler(self, sample_size):
        p = self.pre.patch
        sample_size = [(h - p + 1) * (w - p + 1) for h, w in sample_size]
        self.sampler = categorical.Categorical(probs=torch.Tensor(sample_size))

    def make_binary(self, name):
        dirname = path.dirname(name)
        os.makedirs(dirname, exist_ok=True)
        img = name.replace('bin', '.')
        img = img.replace('.pt', '.png')
        img = imageio.imread(img)
        with open(name, 'wb') as f:
            pickle.dump(img, f)

    def mask_align(self, img_dict):
        '''
        Align resolutions of images and the mask to avoid range error.
        '''
        if 'mask' in img_dict:
            mask = img_dict['mask']
            h = mask.shape[0]
            w = mask.shape[1]
            hh = h // self.scale
            ww = w // self.scale
            img_dict['hr'] = img_dict['hr'][:h, :w]
            img_dict['lr'] = img_dict['lr'][:hh, :ww]

        return img_dict

    def __len__(self):
        return len(self.data['hr'])

    def __getitem__(self, idx):
        def read_file(x):
            if x is None:
                return None
            # Mask
            if 'pth' in x:
                return torch.load(x)
            # Image binary
            elif 'pt' in x:
                with open(x, 'rb') as f:
                    return pickle.load(f)
            # Image
            else:
                return imageio.imread(x)

        if self.train:
            # Random sampling for train
            idx = self.sampler.sample().item()

        img_dict = {k: v[idx] for k, v in self.data.items()}
        name = path.basename(img_dict['hr'])
        name = path.splitext(name)[0]
        img_dict = {k: read_file(v) for k, v in img_dict.items()}
        mask = img_dict['mask']
        if mask is None:
            h, w, _ = img_dict['hr'].shape
            mask = np.zeros((h, w))

        mask_binary = [mask == i for i in range(self.n_classes)]
        mask_binary = np.stack(mask_binary, axis=-1)
        img_dict['mask'] = mask_binary

        img_dict = self.mask_align(img_dict)
        if self.train:
            img_dict = self.pre.get_patch(**img_dict)
            img_dict = self.pre.augment(**img_dict)
        else:
            hh, ww, _ = img_dict['lr'].shape
            h = hh * self.scale
            w = ww * self.scale
            img_dict['hr'] = img_dict['hr'][:h, :w]

        img_dict = self.pre.set_color(**img_dict)
        img_dict = self.pre.np2Tensor(**img_dict)
        if not self.no_mask:
            img_dict['mask_coded'] = mask_utils.mask_coding(img_dict['mask'])

        img_dict['name'] = name
        return img_dict

