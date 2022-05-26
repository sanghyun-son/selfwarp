import sys
sys.path.append('..')
from misc import visualization

import numpy as np
import imageio

import torch
from torchvision import io

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def main() -> None:
    jdet = torch.load('jdet.pth')
    mask = (jdet != -255).float()
    jdet = jdet * mask
    maxval = jdet.abs().max()

    hsl = visualization.to_hsl(jdet.squeeze(), allow_negative=True, q=1)

    img = 127.5 * (hsl + 1)
    img.clamp_(min=0, max=255)
    img.round_()

    img = mask * img + (1 - mask) * 255
    img = img.byte()
    img = img.cpu()
    io.write_png(img, 'hsl.png')

    _, h, w = img.size()
    bar = (h - 1) - torch.arange(h).float()
    bar = bar.view(-1, 1)
    bar = bar.repeat(1, 32)
    bar_hsl = visualization.to_hsl(bar, allow_negative=False, q=1)

    img = 127.5 * (bar_hsl + 1)
    img.clamp_(min=0, max=255)
    img.round_()
    img = img.byte()
    img = img.cpu()

    io.write_png(img, 'bar.png')

    x = Image.open('bar.png')
    d = ImageDraw.Draw(x)
    fontsize = 14
    font = ImageFont.truetype('../../lab/tnr.ttf', fontsize)

    d.text((4, 1), f'{maxval:.2f}', fill='rgb(0, 0, 0)', font=font)
    d.text((2, h - fontsize - 3), f'{-maxval:.2f}', fill='rgb(255, 255, 255)', font=font)
    x.save('bar_text.png')

    img_left = imageio.imread('hsl.png')
    img_z = np.full((h, 16, 3), 255, dtype=np.uint8)
    img_right = imageio.imread('bar_text.png')
    img_merged = np.concatenate((img_left, img_z, img_right), axis=1)
    imageio.imwrite('hsl_merged.png', img_merged)

    ref = imageio.imread('ref.png')
    hh, ww, c = ref.shape
    buffer = np.full((h, ww, c), 255, dtype=np.uint8)
    pad_up = (h - hh) // 2
    pad_down = h - hh - pad_up
    print(pad_up, pad_down)
    buffer[pad_up:-pad_down] = ref
    imageio.imwrite('ref_padded.png', buffer)
    return

if __name__ == '__main__':
    main()