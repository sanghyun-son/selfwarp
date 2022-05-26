from os import path

from data import common
from data.sr import dataclass

_parent_class = dataclass.SRData


class CameraSR(_parent_class):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        super().__init__(*args, **kwargs)

