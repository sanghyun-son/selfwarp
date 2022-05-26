from trainer.sr import base
from misc.gpu_utils import parallel_forward as pforward


class SRTrainer(base.SRTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **samples):
        if self.training:
            samples = self.split_batch(**samples)
            lr_d = samples['d']['lr']
            lr_g = samples['g']['lr']
            hr_d = samples['d']['hr']
            hr = samples['g']['hr']
            mask_d = samples['d']['mask']
            mask_g = samples['g']['mask']

            sr = pforward(self.model, lr_g)
            loss = self.loss(
                g=self.model,
                lr_d=lr_d,
                hr_d=hr_d,
                sr=sr,
                hr=hr,
                mask_d=mask_d,
                mask_g=mask_g,
            )
        else:
            sr = pforward(self.model, samples['lr'])
            loss = self.loss(
                g=None,
                lr_d=None,
                hr_d=None,
                sr=sr,
                hr=samples['hr'],
                mask_d=None,
                mask_g=None,
            )

        return loss, sr

