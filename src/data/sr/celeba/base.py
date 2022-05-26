from os import path

from data import common
from data.sr import dataclass

_parent_class = dataclass.SRData

class CelebAHD(_parent_class):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    def scan(self, target_path):
        filelist = super().scan(target_path)
        if self.train:
            filelist = filelist[:-500]
        else:
            filelist = filelist[-500:]

        return filelist

    def apath(self):
        return path.join(self.dpath, 'CelebAMask')

    def get_path(self, degradation, scale):
        path_hr = path.join(self.apath(), 'CelebA-img-512px')
        path_lr = path.join(self.apath(), 'CelebA-img-128px')
        return {'lr': path_lr, 'hr': path_hr}

