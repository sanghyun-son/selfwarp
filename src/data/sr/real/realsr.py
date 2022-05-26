from os import path
import glob

from data.sr import base

_parent_class = base.SRBase


class RealSR(base.SRBase):

    def __init__(self, *args, **kwargs):
        self.camera = kwargs.pop('camera')
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_kwargs(cfg, train=True):
        kwargs = _parent_class.get_kwargs(cfg, train=train)
        kwargs['camera'] = cfg.camera
        return kwargs

    def scan_dirs(self) -> dict:
        if self.camera.lower() == 'canon':
            camera = 'Canon'
        elif self.camera.lower() == 'nikon':
            camera = 'Nikon'

        if self.train:
            split = 'Train'
        else:
            split = 'Test'

        dpath = path.join(self.dpath, 'RealSR_V3')
        target = path.join(dpath, camera, split, str(self.scale))
        scan_dict = {'hr': target, 'lr': target}
        return scan_dict

    def scan_rule(self, scan_dir: str, k: str) -> list:
        if k == 'hr':
            scan_list = glob.glob(path.join(scan_dir, '*HR.png'))
        elif k == 'lr':
            scan_list = glob.glob(
                path.join(scan_dir, '*LR{}.png'.format(self.scale))
            )

        return scan_list
