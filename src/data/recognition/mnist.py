import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


class MNIST(datasets.MNIST):

    def __init__(self, *args, n_z=100, batch_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_z = n_z
        if self.train:
            self.noise = None
        else:
            self.noise = torch.randn(batch_size, n_z)

    @staticmethod
    def get_kwargs(cfg, train=True):
        # So that pixel values can lie in between -1 and 1
        composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        kwargs = {
            'root': cfg.dpath,
            'train': train,
            'transform': composed,
            'download': True,
        }
        return kwargs

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        if self.train:
            z = torch.randn(self.n_z)
        else:
            z = self.noise[idx % len(self.noise)]

        return {'z': z, 'img': img, 'label': label, 'name': 'mnist'}

