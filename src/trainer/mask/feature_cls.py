import trainer

from model.mask import discriminator
from misc import mask_utils

import torch

def make_trainer(cfg, ckp):
    modules = trainer.prepare_modules(cfg, ckp)
    return FeatureClassifier(ckp, n_classes=cfg.n_classes, **modules)


class FeatureClassifier(trainer.BaseTrainer):

    def __init__(self, *args, n_classes=11, **kwargs):
        super(FeatureClassifier, self).__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.load_discriminator()

    def forward(self, last=True, **samples):
        '''
        Define forward behavior here
        
        Args:
            sample (dict): Input samples

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

        with torch.no_grad():
            f = self.extractor(hr)
            b, c, h, w = f.size()
            f = f.permute(0, 2, 3, 1).contiguous()
            f = f.view(b * h * w, c)

            acc = mask.long().sum(dim=1)
            mask = mask_utils.channel2idx(mask)
            mask[acc > 1] = 0
            mask -= 1
            mask = mask.view(-1)

        pred = self.model(f)
        loss = self.loss(pred, mask, track=True)

        if self.training:
            return loss, None
        else:
            mask = mask.view(b, h, w)
            mask_color = mask_utils.idx2color(mask + 1, n=self.n_classes)
            _, pred_idx = pred.max(dim=1)
            pred_idx = pred_idx.view(b, h, w)
            pred_color = mask_utils.idx2color(pred_idx + 1, n=self.n_classes)

            return loss, {'gt': mask_color, 'pred': pred_color}

    def load_discriminator(self, pretrained=None):
        from os import path
        load_from = path.join(
            '..',
            'experiment',
            #'ost_edsr_x4_mgan',
            'face_edsr_x8_mgan',
            'loss',
            'loss.pt',
        )
        dis = discriminator.Discriminator(
            n_classes=self.n_classes,
            ignore_bg=False,
            mask_scale=16,
        )
        dis.cuda()

        pt = torch.load(load_from)
        state = pt['state']
        state = {
            k.replace('OUT1.MGAN.func.dis.', ''): v for k, v in state.items()
        }
        dis.load_state_dict(state, strict=False)
        self.extractor = dis.features
        self.extractor.eval()
