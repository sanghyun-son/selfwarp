from os import path
import time
import collections

#from matplotlib import pyplot as plt

import torch

""" 
Python implementation of the color map function for the PASCAL VOC data set. 
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""
import numpy as np

def color_map(n=256, ref=None, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    if ref is not None:
        cmap = ref.new_zeros((n, 3))
    else:
        cmap = torch.zeros((n, 3))

    if normalized:
        cmap = cmap.float()
    else:
        cmap = cmap.byte()

    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = cmap.new_tensor([r, g, b])

    if normalized:
        cmap /= 255

    return cmap

def color_bar(save_as, n=256, m=8):
    if m == 8:
        color_dict = collections.OrderedDict(
            background=0,
            sky=4,
            water=6,
            grass=3,
            mountain=5,
            building=1,
            plant=2,
            animal=255,
            void=7,
        )
    else:
        color_dict = {}

    cmap = color_map(n=256, normalized=True).numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    w, h = fig.get_dpi() * fig.get_size_inches()
    h_margin = h / (len(color_dict) + 1)

    for i, (k, v) in enumerate(color_dict.items()):
        color = cmap[v]
        y = h - (i + 1) * h_margin
        xi_line = 0.05 * w
        xf_line = 0.25 * w
        xi_text = 0.30 * w

        ax.text(
            xi_text,
            y,
            k,
            fontsize=(0.80 * h_margin),
            horizontalalignment='left',
            verticalalignment='center',
        )
        ax.hlines(
            y + 0.10 * h_margin,
            xi_line,
            xf_line,
            color=color,
            linewidth=(0.60 * h_margin),
        )

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.savefig(path.join(save_as, 'color_bar.pdf'))
    plt.close()

def mask_coding(mask):
    '''
    Void pixels will be coded with -1
    Args:
        x (torch.BoolTensor): C x H x W mask

    Return:
        torch.LongTensor: H x W coded mask
    '''
    void = mask.any(0)
    void = void.long() - 1
    mask = sum(i * m.long() for i, m in enumerate(mask))
    mask = mask + void
    return mask

def softmax2img(softmax):
    label = softmax.max(dim=1)[1]
    img = code2img(label)
    return img

def code2img(mask_coded, ost=False):
    '''
    Args:
        mask_coded (torch.LongTensor): B x H x W or H x W encoded mask

    Return:
        torch.Tensor: B x 3 x H x W or 3 x H x W mask image
    '''
    # Void labels will be assigned to 255
    mask_coded[mask_coded == -1] = 255
    if ost:
        bg = (mask_coded == 0)
        sky = (mask_coded == 1)
        water = (mask_coded == 2)
        grass = (mask_coded == 3)
        mountain = (mask_coded == 4)
        bd = (mask_coded == 5)
        plant = (mask_coded == 6)
        animal = (mask_coded == 7)
        void = (mask_coded == 255)

        mask_coded[bg] = 0
        mask_coded[sky] = 4
        mask_coded[water] = 6
        mask_coded[grass] = 3
        mask_coded[mountain] = 5
        mask_coded[bd] = 1
        mask_coded[plant] = 2
        mask_coded[animal] = 255
        mask_coded[void] = 7

    cmap = color_map(n=256, ref=mask_coded, normalized=True)
    img = cmap[mask_coded]

    # Batched
    if img.dim() == 4:
        img = img.permute(0, 3, 1, 2)
    # Single image
    elif img.dim() == 3:
        img = img.permute(2, 0, 1)

    return img

if __name__ == '__main__':
    cmap = color_map(n=4)
    print(cmap)
