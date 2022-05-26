from trainer.gan import dcgan
from model.utils import forward_utils as futils
from ops import shave
from ops import filters
from ops import transformation

_parent_class = dcgan.GANTrainer


class DownTrainer(_parent_class):

    def __init__(
            self, *args,
            scale=2, x8=False, test_period=1, blur_sigma=0.5,
            **kwargs):

        super().__init__(*args, **kwargs)
        self.scale = scale
        self.x8 = x8
        self.test_period = test_period
        self.blur_sigma = blur_sigma

    @staticmethod
    def get_kwargs(cfg):
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['scale'] = cfg.scale
        kwargs['x8'] = cfg.x8
        kwargs['test_period'] = cfg.test_period
        kwargs['blur_sigma'] = cfg.blur_sigma
        return kwargs

    def forward(self, **samples):
        if self.training:
            samples = self.split_batch(**samples)
            lr_d = samples['d']['lr']
            lr_g = samples['g']['lr']
            hr_d = samples['d']['hr']
            hr = samples['g']['hr']

            down = self.pforward(hr)
            hr_t, idx = transformation.random_transformation(hr)
            down_t = self.pforward(hr_t)
            down_tt, _ = transformation.random_transformation(
                down_t, inverse=idx,
            )

            if self.blur_sigma > 0:
                down_filtered = filters.gaussian_filtering(
                    down, sigma=self.blur_sigma,
                )
            else:
                down_filtered = down

            loss = self.loss(
                g=self.model,                   # Generator
                hr_d=hr_d,                      # Input HR for D
                lr_d=lr_d,                      # Real LR for D
                hr=hr,                          # Input HR for G
                down=down,                      # Generated LR
                down_filtered=down_filtered,    # Fake LR for G
                lr_g=lr_g,                      # Real LR for G
                down_tt=down_tt,                # Output from transformed input
            )
        else:
            hr = samples['hr']
            if self.test_period > 0:
                down = self.pforward(hr)
                hr_t, idx = transformation.random_transformation(hr)
                down_t = self.pforward(hr_t)
                down_tt, _ = transformation.random_transformation(down_t, inverse=idx)

                if self.blur_sigma > 0:
                    down_filtered = filters.gaussian_filtering(
                        down, sigma=self.blur_sigma,
                    )
                else:
                    down_filtered = down

                loss = self.loss(
                    g=None,                         # Generator
                    hr_d=None,                      # Input HR for D
                    lr_d=None,                      # Real LR for D
                    hr=hr,                          # Input HR for G
                    down=down,                      # Fake LR for G
                    down_filtered=down_filtered,    # Fake LR for G
                    lr_g=samples['lr'],             # Real LR for G
                    down_tt=down_tt,                # Output from transformed input
                )
            else:
                if self.x8:
                    down = futils.x8_forward(self.model, hr)
                else:
                    down = self.pforward(hr)

                loss = 0

        return loss, {'{:0>2}'.format(self.get_epoch()): down}
