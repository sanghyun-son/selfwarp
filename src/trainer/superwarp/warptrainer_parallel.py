import random
import types
import typing

from utils import svf
from utils import warp
from utils import random_transform
from ops import filters
from trainer import base_trainer
from misc import image_utils
from misc.gpu_utils import parallel_forward as pforward
from model.superwarp import kernel

import torch
from torch import cuda
from torch import nn

_parent_class = base_trainer.BaseTrainer


class ParallelWorker(nn.Module):

    def __init__(
            self,
            net: nn.Module,
            patch_max: int=128,
            kernel_estimator: typing.Optional[nn.Module]=None) -> None:

        super().__init__()
        self.net = net
        self.patch_max = patch_max
        self.kernel_estimator = kernel_estimator
        return

    @torch.no_grad()
    def elliptical_warping(
            self,
            y: torch.Tensor,
            m_inverse: torch.Tensor,
            sizes: typing.Tuple[int, int]) -> torch.Tensor:

        grid = svf.projective_grid(sizes, m_inverse)
        grid, yi = warp.safe_region(grid, y.size(-2), y.size(-1))
        k = self.kernel_estimator(m_inverse, sizes, grid, yi)
        x = warp.warp(
            y, sizes=sizes,
            grid=grid,
            yi=yi,
            kernel=k,
            fill_value=self.net.fill_value,
        )

        return x

    @torch.no_grad()
    def get_input(
            self,
            y: torch.Tensor,
            m: torch.Tensor,
            training: bool=True,
            identity: bool=True) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        # Operations on a small matrix is very slow on the GPU
        if hasattr(self.net, 'fill_value'):
            ignore_value = self.net.fill_value
        else:
            ignore_value = self.net.fill

        # The transformation matrix is shared in the same batch
        m = m.float().cpu()
        m, sizes = random_transform.compensate(y, m, orientation=False)
        m_inverse = warp.inverse_3x3(m)
        if identity:
            return y, m

        if self.kernel_estimator is not None:
            x = self.elliptical_warping(y, m_inverse, sizes)
            #cuda.empty_cache()
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
            margin=2,
            stochastic=training,
        )
        x = image_utils.quantize(x)
        m_back = random_transform.compensate_offset(m_inverse, iy, ix)
        return x, m_back

    def forward(
            self,
            y: torch.Tensor,
            m: torch.Tensor,
            training: bool=True,
            identity: bool=False) -> torch.Tensor:

        x, m_back = self.get_input(y, m, training=training, identity=identity)
        out = self.net(x, m_back, (y.size(-2), y.size(-1)))
        return out


class SuperWarpTrainer(_parent_class):

    def __init__(
            self,
            *args,
            patch_max: int=128,
            identity_p: float=0,
            elliptical: bool=False,
            kernel_size: int=4,
            kernel_reset: bool=False,
            sampler_reset: bool=False,
            **kwargs) -> None:

        self.kernel_reset = kernel_reset
        self.sampler_reset = sampler_reset
        super().__init__(*args, **kwargs)
        if elliptical:
            kernel_estimator = kernel.KernelEstimator(kernel_size=kernel_size)
            kernel_estimator.cuda()
        else:
            kernel_estimator = None

        self.worker = ParallelWorker(
            self.model,
            patch_max=patch_max,
            kernel_estimator=kernel_estimator,
        )
        self.identity_p = identity_p
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['patch_max'] = cfg.patch_max
        kwargs['identity_p'] = cfg.identity_p
        kwargs['elliptical'] = cfg.elliptical
        kwargs['kernel_size'] = cfg.kernel_size
        kwargs['kernel_reset'] = cfg.kernel_reset
        kwargs['sampler_reset'] = cfg.sampler_reset
        return kwargs

    def preprocess_state(self, state: dict) -> dict:
        pop_list = []
        for k in state.keys():
            if self.kernel_reset and 'kernel_estimator' in k:
                pop_list.append(k)

            if self.sampler_reset and 'amss' in k:
                pop_list.append(k)

        for p in pop_list:
            state.pop(p)

        return state

    def forward(self, **samples) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Target image
        y = samples['img']
        m = samples['m'][0].cpu()
        identity = False
        if self.training:
            if random.random() < self.identity_p:
                # Identity transform
                _, _, h, w = y.size()
                patch_size = 64
                iy = random.randrange(0, h - patch_size + 1)
                ix = random.randrange(0, w - patch_size + 1)
                y = y[..., iy:(iy + patch_size), ix:(ix + patch_size)]
                m = torch.eye(3)
                identity = True

            m = m.repeat(cuda.device_count(), 1)

        out = pforward(self.worker, y, m, training=self.training, identity=identity)
        if isinstance(out, tuple):
            out, mask = out
        else:
            mask = (out != self.model.fill_value).float()

        loss = self.loss(out=out, y=y, mask=mask)
        return loss, out

    def at_epoch_end(self) -> None:
        super().at_epoch_end()
        cuda.empty_cache()
        return


REPRESENTATIVE = SuperWarpTrainer
