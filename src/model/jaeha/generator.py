from torch import nn

# Should be fixed later...
from model.jaeha.common import *

####################################################################
#--------------------------- Generators ----------------------------
####################################################################

class Generator(nn.Module):

    def __init__(self, output_dim_a=3, output_dim_b=3, input_dim_b=3, norm=None, nl_layer=None):
        super().__init__()
        
        tch = 64
        ### make_noised part ###
        headB = [MyConv2d(input_dim_b, tch, kernel_size=3, stride=1, padding=1, norm=norm)]
        self.headB = nn.Sequential(*headB)

        bodyB1 = [
            MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=True),
            MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=True),
            MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=True),
            MyConv2d(tch, tch, kernel_size=5, stride=1, padding=2, norm=norm, Res=True),
        ]

        bodyB2 = [MyConv2d(tch, tch*2, kernel_size=2, stride=2, padding=0, norm=norm),]
    
        bodyB3 = [
            MyConv2d(tch*2, tch*2, kernel_size=3, stride=1, padding=1, norm=norm, Res=True),
            MyConv2d(tch*2, tch*2, kernel_size=3, stride=1, padding=1, norm=norm, Res=True),
            MyConv2d(tch*2, tch*2, kernel_size=3, stride=1, padding=1, norm=norm, Res=True),
        ]

        self.bodyB1 = nn.Sequential(*bodyB1)
        self.bodyB2 = nn.Sequential(*bodyB2)
        self.bodyB3 = nn.Sequential(*bodyB3)

        tailB = [ nn.Conv2d(tch*2, output_dim_b, kernel_size=1, stride=1, padding=0) ]
        self.tailB = nn.Sequential(*tailB)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.apply(gaussian_weights_init)

    @staticmethod
    def get_kwargs(cfg, conv=None):
        kwargs = {
            'output_dim_a': cfg.input_dim_a,
            'output_dim_b': cfg.input_dim_b,
            'input_dim_b': cfg.input_dim_b,
            'norm': cfg.gen_norm,
            'nl_layer': get_non_linearity(layer_type='lrelu'),
        }
        return kwargs

    def forward(self, HR):
        tres = self.avgpool2(HR)
        out = self.headB(HR)
        
        res = out
        out = self.bodyB1(out)
        out += res

        out = self.bodyB2(out)

        res = out
        out = self.bodyB3(out)
        out += res

        out = self.tailB(out)
        out += tres

        return out

REPRESENTATIVE = Generator
