from os import path

from data.sr.div2k import base

_parent_class = base.DIV2K

class DIV2KDenoising(_parent_class):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = _parent_class.get_kwargs(cfg, train=train)
        kwargs['scale'] = 1
        return kwargs

    def get_path(self, degradation, scale):
        if not (self.train or self.tval):
            split = 'valid'
        else:
            split = 'train'

        path_hr = path.join(self.apath(), 'DIV2K_{}_HR'.format(split))
        path_lr = '{}_{}'.format(path_hr, degradation)
        return {'lr': path_lr, 'hr': path_hr}

    def get_patch(self, **kwargs):
        if self.train:
            return super().get_patch(**kwargs)
        else:
            s = 64
            hr = kwargs['hr']
            h = hr.shape[0]
            w = hr.shape[1]
            h_mod = s * (h // s)
            w_mod = s * (w // s)
            for k, v in kwargs.items():
                kwargs[k] = v[:h_mod // 2, :w_mod // 2]

            return kwargs

