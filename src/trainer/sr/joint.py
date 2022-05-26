from os import path
import types

from trainer.gan import dcgan
from model import loader
from model.utils import forward_utils as futils
from misc.gpu_utils import parallel_forward as pforward

import torch

_parent_class = dcgan.GANTrainer


def parse_exp(cfg):
    config_path = path.join(cfg.exp_down, 'config.txt')
    print('Make the downsampler from {}...'.format(config_path))
    with open(config_path, 'r') as f:
        lines = f.read().splitlines()

    # Remove header and command line arguments
    lines = lines[2:]
    cfg_down = types.SimpleNamespace()
    for line in lines:
        if ':' in line:
            k, v = line.split(':')
            # Remove whitespace
            v = v[1:]
            if v.isdecimal():
                v = int(v)
            if v == 'True':
                v = True
            if v == 'False':
                v = False

            setattr(cfg_down, k, v)

        if '--' in line:
            break

    downsampler = loader.get_model(cfg_down)
    print(downsampler)
    ckpt_path = path.join(cfg.exp_down, 'latest.ckpt')
    print('Load the pre-trained downsampler from {}...'.format(ckpt_path))
    ckpt = torch.load(ckpt_path)
    state = ckpt['model']
    downsampler.load_state_dict(state, strict=True)
    downsampler.eval()
    return downsampler


class SRJoint(_parent_class):

    def __init__(
            self, *args,
            downsampler=None, scale=2, noise=0, shave=4,
            x8=False, quads=False, **kwargs):

        super().__init__(*args, **kwargs)
        self.downsampler = downsampler
        self.scale = scale
        self.shave = shave
        self.x8 = x8
        self.quads = quads
        self.noise = noise

    @staticmethod
    def get_kwargs(cfg):
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['downsampler'] = parse_exp(cfg)
        kwargs['scale'] = cfg.scale
        kwargs['noise'] = cfg.noise
        kwargs['shave'] = cfg.shave_down
        kwargs['x8'] = cfg.x8
        kwargs['quads'] = cfg.quads
        return kwargs

    def forward(self, **samples):
        if self.training:
            samples = self.split_batch(**samples)
            lr_d = samples['d']['lr']
            # LR images are generated dynamically
            #lr_g = samples['g']['lr']
            hr_d = samples['d']['hr']
            hr = samples['g']['hr']

            with torch.no_grad():
                lr_syn = pforward(self.downsampler, hr)
                if self.shave > 0:
                    s = self.shave
                    ss = self.scale * s
                    lr_syn = lr_syn[..., s:-s, s:-s]
                    hr = hr[..., ss:-ss, ss:-ss]

                lr_syn = 127.5 * (lr_syn + 1)
                if self.noise > 0:
                    n = self.noise * torch.randn_like(lr_syn)
                    lr_syn += n

                lr_syn.round_()
                lr_syn = (lr_syn / 127.5) - 1

            sr = pforward(self.model, lr_syn)
            loss = self.loss(
                g=self.model,
                lr_d=lr_d,
                hr_d=hr_d,
                sr=sr,
                hr=hr,
            )
        else:
            lr = samples['lr']
            if self.quads:
                sr = futils.quad_forward(self.model, lr)
            elif self.x8:
                sr = futils.x8_forward(self.model, lr)
            else:
                sr = pforward(self.model, lr)

            loss = self.loss(
                g=None,
                lr_d=None,
                hr_d=None,
                sr=sr,
                hr=samples['hr'],
            )

        return loss, sr
