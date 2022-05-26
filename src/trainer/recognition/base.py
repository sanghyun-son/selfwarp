from trainer import base_trainer
from misc.gpu_utils import parallel_forward as pforward

_parent_class = base_trainer.BaseTrainer


class RecogTrainer(_parent_class):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **samples):
        img = samples['img']
        label = samples['label']
        pred = pforward(self.model, img)
        loss = self.loss(pred=pred, label=label)

        return loss, None
