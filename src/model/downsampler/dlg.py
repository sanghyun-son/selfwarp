from torch import nn
from model import common


class DeepLinearGenerator(nn.Module):

    def __init__(self, scale=2, n_feats=64, struct=None, conv=common.default_conv):
        super().__init__()
        # First layer - Converting RGB image to latent space
        self.first_layer = conv(1, n_feats, struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [conv(n_feats, n_feats, struct[layer], bias=False)]

        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = conv(n_feats, 1, struct[-1], stride=scale, bias=False)

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        kwargs = {
            'scale': cfg.scale,
            'n_feats': cfg.n_feats,
            'struct': [7, 5, 3, 1, 1, 1],
        }
        return kwargs

    def forward(self, x):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        b, c, h, w = x.size()
        x = x.view(-1, 1, h, w)
        x = self.first_layer(x)
        x = self.feature_block(x)
        x = self.final_layer(x)
        _, _, hh, ww = x.size()
        x = x.view(b, c, hh, ww)
        return x
