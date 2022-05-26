'''
Provided by Seungjun Nah (seungjun.nah@gmail.com)
'''

import torch
from torch import nn
from torch.nn import functional as F

class SSIM(nn.Module):

    def __init__(self, name=None):
        super(SSIM, self).__init__()
        self.truncate = 3.5
        self.sigma = 1.5

    def ssim_weight(self, x):
        r = int(self.truncate * self.sigma + 0.5)
        win_size = 2 * r + 1
        # Number of color channels
        c = x.size(1)

        coord = [x - win_size // 2 for x in range(win_size)]
        w = x.new_tensor(coord)
        w.pow_(2)
        w /= -(2 * self.sigma**2)
        w.exp_()
        w.unsqueeze_(1)

        w = w.mm(w.t())
        w /= w.sum()
        w = w.repeat(c, 1, 1, 1)

        return w

    def forward(self, x, y):
        x = 127.5 * (x + 1)
        y = 127.5 * (y + 1)

        k1 = 0.01
        k2 = 0.03
        sigma = 1.5
        
        truncate = 3.5
        r = int(truncate * sigma + 0.5)  # radius as in ndimage
        win_size = 2 * r + 1

        c = x.size(1)

        if x.size(2) < win_size or x.size(3) < win_size:
            raise ValueError(
                "win_size exceeds image extent.  If the input is a multichannel "
                "(color) image, set multichannel=True.")

        w = self.ssim_weight(x)
        print(w.size())
        def filter_func(img):
            return F.conv2d(img, w, padding=0, groups=c)

        # compute (weighted) means
        ux = filter_func(x)
        uy = filter_func(y)

        # compute (weighted) variances and covariances
        uxx = filter_func(x * x)
        uyy = filter_func(y * y)
        uxy = filter_func(x * y)
        vx = (uxx - ux * ux)
        vy = (uyy - uy * uy)
        vxy = (uxy - ux * uy)

        c1 = (255 * k1) ** 2
        c2 = (255 * k2) ** 2

        a1 = 2 * ux * uy + c1
        a2 = 2 * vxy + c2
        b1 = ux ** 2 + uy ** 2 + c1
        b2 = vx + vy + c2

        d = b1 * b2
        s = (a1 * a2) / d

        # compute (weighted) mean of ssim
        mssim = s.mean()
        return mssim

