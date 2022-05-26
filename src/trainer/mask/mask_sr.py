from trainer.sr import base

import torch
from torch.nn import functional as F


class MaskedSRTrainer(base.SRTrainer):

    def __init__(self, *args, **kwargs):
        super(MaskedSRTrainer, self).__init__(*args, **kwargs)

    def forward(self, last=True, **samples):
        '''
        Define forward behavior here
        
        Args:
            sample (tuple): an input-target pair

        Return:
            if self.training:
                Tensor: final loss value for back-propagation
            else:
                An arbitrary type: output result(s)
        '''
        #x8 = self.x8 and (not self.training)
        lr = samples['lr']
        hr = samples['hr']
        mask = samples['mask']

        sr = self.model(lr, x8=self.x8)
        loss = self.loss({'sr': sr, 'hr': hr, 'mask': mask})
        return loss, sr
