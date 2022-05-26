import torch
from torch import nn
from torch.nn import functional as F


class LocalMean(nn.Module):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

        n = [s for s in name if s.isdigit()]
        n = ''.join(n)
        self.window_size = int(n)
        self.dense = 'd' in name
        self.l1 = '-l' in name

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()

        if self.dense:
            sx = 1
            sy = 1
        else:
            sx = None
            sy = None

        if xh > yh:
            scale = xh // yh
            if sx is not None:
                sx *= scale
            x_pool = F.avg_pool2d(x, scale * self.window_size, stride=sx)
            y_pool = F.avg_pool2d(y, self.window_size, stride=sy)
        else:
            scale = yh // xh
            if sy is not None:
                sy *= scale
            x_pool = F.avg_pool2d(x, self.window_size, stride=sx)
            y_pool = F.avg_pool2d(y, scale * self.window_size, stride=sy)

        if self.l1:
            loss = F.l1_loss(x_pool, y_pool)
        else:
            loss = F.mse_loss(x_pool, y_pool)

        return loss

if __name__ == '__main__':
    from os import path
    import glob
    import numpy as np
    import imageio
    import tqdm

    loss_funcs = {
        'ap1': LocalMean('1'),
        'ap4': LocalMean('4'),
        'ap8': LocalMean('8'),
        'ap16': LocalMean('16')
    }
    lr_dirs = [
        '../../../dataset/DIV2K/DIV2K_train_LR_bicubic/X2',
        '../../../dataset/DIV2K/DIV2K_train_LR_bicubic/X3',
        '../../../dataset/DIV2K/DIV2K_train_LR_bicubic/X4',
        #'../../../dataset/RealSR_V3/Nikon/Train/2',
        #'../../../dataset/RealSR_V3/Nikon/Train/3',
        #'../../../dataset/RealSR_V3/Nikon/Train/4',
        #'../../../dataset/RealSR_V3/Canon/Train/2',
        #'../../../dataset/RealSR_V3/Canon/Train/3',
        #'../../../dataset/RealSR_V3/Canon/Train/4',
    ]
    hr_dirs = [
        '../../../dataset/DIV2K/DIV2K_train_HR',
        '../../../dataset/DIV2K/DIV2K_train_HR',
        '../../../dataset/DIV2K/DIV2K_train_HR',
        #'../../../dataset/RealSR_V3/Nikon/Train/2',
        #'../../../dataset/RealSR_V3/Nikon/Train/3',
        #'../../../dataset/RealSR_V3/Nikon/Train/4',
        #'../../../dataset/RealSR_V3/Canon/Train/2',
        #'../../../dataset/RealSR_V3/Canon/Train/3',
        #'../../../dataset/RealSR_V3/Canon/Train/4',
    ]
    lr_dirs = [sorted(glob.glob(path.join(d, '*.png'))) for d in lr_dirs]
    hr_dirs = [sorted(glob.glob(path.join(d, '*.png'))) for d in hr_dirs]
    for lr_dir, hr_dir in zip(lr_dirs, hr_dirs):
        loss_dict = {}
        for lr_img, hr_img in tqdm.tqdm(zip(lr_dir, hr_dir), ncols=80, total=len(lr_dir)):
            lr = imageio.imread(lr_img)
            hr = imageio.imread(hr_img)
            def to_tensor(x):
                x = np.transpose(x, (2, 0, 1))
                x = torch.from_numpy(x).float()
                x = x / 127.5 - 1
                x.unsqueeze_(0)
                return x

            lr = to_tensor(lr)
            hr = to_tensor(hr)
            for k, v in loss_funcs.items():
                loss = v(lr, hr)
                if k in loss_dict:
                    loss_dict[k] += loss
                else:
                    loss_dict[k] = loss

        for k, v in loss_dict.items():
            print('{}: {}'.format(k, v / len(lr_dir)))
