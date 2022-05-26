from bicubic_pytorch import core
from trainer import base_trainer

import torch
from torch import nn
from torch.nn import functional

_parent_class = base_trainer.BaseTrainer


class UnpairedTrainer(_parent_class):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return

    def forward(self, **samples):
        x = samples['lr']
        y = samples['hr']
        if self.training:
            with torch.no_grad():
                y_down = core.imresize(y, scale=0.25)

            gxy_x, x_up = self.pforward(x, source=True)
            gyx_y_down, y_o, uyy_y_o = self.pforward(y_down, source=False)
            gxy_x_geo = self.x8_forward(x, self.model.gxy)
            loss = self.loss(
                x=x,
                y=y,
                gyx_y_down=gyx_y_down,
                gxy_x=gxy_x,
                gxy_x_geo=gxy_x_geo,
                y_down=y_down,
                uyy_y_o=uyy_y_o,
                x_up=x_up,
                y_o=y_o,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )
        else:
            _, x_up = self.pforward(x, source=True)
            loss = self.loss(
                x=None,
                y=y,
                gyx_y_down=None,
                gxy_x=None,
                gxy_x_geo=None,
                y_down=None,
                uyy_y_o=None,
                x_up=x_up,
                y_o=None,
                dummy_1=0,
                dummy_2=0,
                dummy_3=0,
            )

        return loss, x_up


    def x8_forward(self, x: torch.Tensor, net: nn.Module) -> torch.Tensor:
        n_tfs = 8
        xs = []
        for i in range(n_tfs):
            x_aug = x
            if i % 2 > 0:
                x_aug = x_aug.flip(-1)

            if i % 4 > 1:
                x_aug = x_aug.flip(-2)

            if i % 4 > 3:
                x_aug = x_aug.transpose(-1, -2)

            xs.append(x_aug)

        y_augs = [net(_x) for _x in xs]
        ys = []
        for i, y_aug in enumerate(y_augs):
            if i % 4 > 3:
                y_aug = y_aug.transpose(-1, -2)

            if i % 4 > 1:
                y_aug = y_aug.flip(-2)

            if i % 2 > 0:
                y_aug = y_aug.flip(-1)

            ys.append(y_aug)

        y = sum(ys) / n_tfs
        return y