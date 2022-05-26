import types

import typing

from utils import svf
from utils import warp
from utils import random_transform
from ops import filters
from trainer import base_trainer
from model.superwarp import kernel

import torch
from torch import cuda

_parent_class = base_trainer.BaseTrainer


class SuperWarpTrainer(_parent_class):

    def __init__(
            self,
            *args,
            patch_max: int=128,
            elliptical: bool=False,
            kernel_size: int=4,
            **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.patch_max = patch_max
        self.elliptical = elliptical
        if elliptical:
            self.kernel_estimator = kernel.KernelEstimator(
                kernel_size=kernel_size,
            )
            if cuda.is_available():
                self.kernel_estimator.cuda()
        else:
            self.kernel_estimator = None

        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['patch_max'] = cfg.patch_max
        kwargs['elliptical'] = cfg.elliptical
        kwargs['kernel_size'] = cfg.kernel_size
        return kwargs

    def forward(self, **samples) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Target image
        y = samples['img']
        ignore_value = self.model.fill_value
        # The transformation matrix is shared in the same batch
        # Operations on a small matrix is very slow on the GPU
        m = samples['m'][0].cpu()
        m, sizes = random_transform.compensate(y, m, orientation=False)
        m_inverse = warp.inverse_3x3(m)
        with torch.no_grad():
            # Elliptical interpolation prevents aliasing
            if self.elliptical:
                grid = svf.projective_grid(sizes, m_inverse)
                grid, yi = warp.safe_region(grid, y.size(-2), y.size(-1))
                k = self.kernel_estimator(m_inverse, sizes, grid, yi)
                x = warp.warp(
                    y,
                    sizes=sizes,
                    grid=grid,
                    yi=yi,
                    kernel=k,
                    fill_value=ignore_value,
                )
            else:
                # To avoid aliasing
                g = filters.gaussian_filtering(y, sigma=1.0)
                x = warp.warp(
                    g,
                    m=m_inverse,
                    sizes=sizes,
                    fill_value=ignore_value,
                    warp_inverse=True,
                )

            # Crop the largest region
            ignore = (x[0:1, 0:1] == ignore_value).float()
            x, iy, ix = random_transform.crop_largest(
                x,
                ignore,
                patch_max=self.patch_max,
                stochastic=self.training,
            )

            # To avoid possible boundary effects
            margin = 2
            if margin > 0:
                x = x[..., margin:-margin, margin:-margin]
                iy += margin
                ix += margin

            # Construct the backward transform
            m_translate = m_inverse.new_tensor([
                [1, 0, ix],
                [0, 1, iy],
                [0, 0, 1],
            ])
            m_back = torch.matmul(m_inverse, m_translate)

        # Parallel forward is implemented in the model
        out = self.model(x, m_back, (y.size(-2), y.size(-1)))
        mask = (out != ignore_value).float()
        # For debugging
        '''
        sample_dict = {'y': y, 'g': g, 'x': x, 'out': out, 'mask': mask}
        self.pause(**sample_dict)
        '''
        loss = self.loss(out=out, y=y, mask=mask)
        return loss, out

