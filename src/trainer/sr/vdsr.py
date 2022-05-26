import random

from bicubic_pytorch import core
from trainer.gan import dcgan
from misc.gpu_utils import parallel_forward as pforward
from model.utils import forward_utils as futils
_parent_class = dcgan.GANTrainer


class SRTrainer(_parent_class):

    def __init__(
            self,
            *args,
            scale: float=2.0,
            x8: bool=False,
            quads: bool=False,
            **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.scale = scale
        self.x8 = x8
        self.quads = quads
        return

    @staticmethod
    def get_kwargs(cfg):
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['scale'] = cfg.scale
        kwargs['x8'] = cfg.x8
        kwargs['quads'] = cfg.quads
        return kwargs

    def forward(self, **samples):
        hr = samples['img']
        if self.training:
            scale = random.choice((2, 3, 4))
        else:
            scale = self.scale
            h = hr.size(-2)
            w = hr.size(-1)
            hh = h - h % int(scale)
            ww = w - w % int(scale)
            hr = hr[..., :hh, :ww]

        lr = core.imresize(hr, scale=(1 / scale), kernel='cubic')
        sr = pforward(self.model, lr, scale)
        loss = self.loss(sr=sr, hr=hr)
        #self.pause(lr=lr, hr=hr, sr=sr)
        return loss, sr
