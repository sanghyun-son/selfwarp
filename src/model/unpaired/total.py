import typing

from model import common
from model.unpaired import gyx
from model.unpaired import gxy
from model.unpaired import uyy

import torch
from torch import nn


class UnpairedTotal(nn.Module):

    def __init__(self, scale: int=4) -> None:
        super().__init__()
        self.scale = scale
        self.gyx = gyx.GYX()
        self.gxy = gxy.GXY()
        self.uyy = uyy.UYY()
        return

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv) -> dict:
        return {
            'scale': cfg.scale,
        }

    def forward(
            self,
            x: torch.Tensor,
            source: bool=False) -> typing.Tuple[torch.Tensor]:

        if source:
            return self.forward_source(x)
        else:
            return self.forward_target(x)

    def forward_source(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        gxy_x = self.gxy(x)
        x_up = self.uyy(gxy_x)
        return gxy_x, x_up

    def forward_target(self, y_down: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        gyx_y_down = self.gyx(y_down)
        y_o = self.gxy(gyx_y_down)
        uyy_y_o = self.uyy(y_o)
        return gyx_y_down, y_o, uyy_y_o
