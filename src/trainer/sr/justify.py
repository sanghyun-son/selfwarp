import torch

from trainer.gan import dcgan
from misc.gpu_utils import parallel_forward as pforward
from model.utils import forward_utils as futils
import synth
_parent_class = dcgan.GANTrainer


class SRTrainer(_parent_class):

    def __init__(self, *args, x8=False, quads=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.x8 = x8
        self.quads = quads
        self.scale = 4
        return

    @staticmethod
    def get_kwargs(cfg):
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['x8'] = cfg.x8
        kwargs['quads'] = cfg.quads
        return kwargs

    def forward(self, **samples):
        hr = samples['hr']
        deg = samples['deg']
        lr_list = []
        for idx, hr_sub in enumerate(hr):
            hr_sub = hr_sub.unsqueeze(0)
            lr = synth.synthesize_deg(
                hr_sub,
                deg['kernel_sigma'][idx],
                deg['noise_sigma'][idx],
                deg['jpeg_q'][idx],
                scale=self.scale,
            )
            lr_list.append(lr)

        lr = torch.cat(lr_list, dim=0)
        if self.training:
            sr = pforward(self.model, lr)
            loss = self.loss(
                sr=sr,
                hr=hr,
            )
        else:
            if self.quads:
                sr = futils.quad_forward(self.model, lr)
            elif self.x8:
                sr = futils.x8_forward(self.model, lr)
            else:
                sr = pforward(self.model, lr)

            hr_crop = hr[..., :sr.size(-2), :sr.size(-1)]
            loss = self.loss(
                sr=sr,
                hr=hr_crop,
            )

        return loss, sr
