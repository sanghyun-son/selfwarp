from torch import nn

# Should be fixed later...
from model.jaeha.common import *

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class MultiScaleDis(nn.Module):

    def __init__(self, input_dim=3, n_scale=3, n_layer=5, norm='None'):
        super().__init__()
        ch = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm))

        self.apply(gaussian_weights_init)

    @staticmethod
    def get_kwargs(cfg, conv=None):
        kwargs = {
            'input_dim': cfg.input_dim_b,
            'n_scale': cfg.dis_scale,
            'norm': cfg.dis_norm,
        }
        return kwargs

    def _make_net(self, ch, input_dim, n_layer, norm):
        model = [MyConv2d(input_dim, ch, kernel_size=7, stride=1, padding=3, norm=norm, Leaky=True)]
        tch = ch

        for _ in range(1,n_layer):
            model += [MyConv2d(tch, min(1024, tch * 2), kernel_size=5, stride=2, padding=2, norm=norm, Leaky=True)]
            tch *= 2
            tch = min(1024, tch)
    
        model += [nn.Conv2d(tch, 1, 2, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for Dis in self.Diss:
            outs.append(Dis(x))
            x = self.downsample(x)

        # Special case: When n_scale == 1
        if len(self.Diss) == 1:
            outs = outs[0]

        return outs

REPRESENTATIVE = MultiScaleDis
