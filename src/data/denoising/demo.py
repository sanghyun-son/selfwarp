from data import common
from data.sr import demo
import numpy as np
from PIL import Image

_parent_class = demo.Demo


class Demo(_parent_class):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = _parent_class.get_kwargs(cfg, train=train)
        kwargs['scale'] = 1
        return kwargs

    def get_patch(self, **kwargs):
        s = 64
        hr = kwargs['hr']
        h = hr.shape[0]
        w = hr.shape[1]
        h_mod = s * (h // s)
        w_mod = s * (w // s)
        for k, v in kwargs.items():
            kwargs[k] = v[:h_mod, :w_mod]

        return kwargs
