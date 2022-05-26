import types
import typing

import torch
from torch import nn
from torchvision import models
from torchvision.models import resnet


class KernelEstimator(nn.ModuleDict):

    def __init__(self, n_feats: int) -> None:
        # We use different classfier for different degradation kernels
        super().__init__()
        # Kernel types (classification)
        # (Gaussian, Generalized Gaussian, Plateau, Sinc)
        self['kernel_type'] = nn.Linear(n_feats, 4)
        # Determine whether the blur is applied
        self['is_blur'] = nn.Linear(n_feats, 2)

        # Kernel size (classification)
        # Possible sizes: (7, 9, 11, 13, 15, 17, 19, 21)
        self['kernel_size_gaussian'] = nn.Linear(n_feats, 8)
        self['kernel_size_ggaussian'] = nn.Linear(n_feats, 8)
        self['kernel_size_plateau'] = nn.Linear(n_feats, 8)
        self['kernel_size_sinc'] = nn.Linear(n_feats, 8)

        # Kernel sigma (regression)
        # Estimate 2x2 matrix
        self['kernel_sigma_gaussian'] = nn.Linear(n_feats, 4)
        self['kernel_sigma_ggaussian'] = nn.Linear(n_feats, 4)
        self['kernel_sigma_plateau'] = nn.Linear(n_feats, 4)

        # For sinc, we have omega rather than sigma (regression)
        self['kernel_omega_sinc'] = nn.Linear(n_feats, 1)

        # For Generalized Gaussian and plateau, we have beta (regression)
        self['kernel_beta_ggaussian'] = nn.Linear(n_feats, 1)
        self['kernel_beta_plateau'] = nn.Linear(n_feats, 1)
        return

    def forward(self, x) -> typing.Mapping[str, torch.Tensor]:
        ret_dict = {}
        for k, v in self.items():
            ret_dict[k] = v(x)

        return ret_dict


class ResizeEstimator(nn.ModuleDict):

    def __init__(self, n_feats: int) -> None:
        # We use different classfier for different noise types
        super().__init__()
        # Interpolation types (classification)
        # (area, bilinear, bicubic)
        self['resize_type'] = nn.Linear(n_feats, 3)
        # Resize scale (regression)
        self['resize_scale_area'] = nn.Linear(n_feats, 1)
        self['resize_scale_bilinear'] = nn.Linear(n_feats, 1)
        self['resize_scale_bicubic'] = nn.Linear(n_feats, 1)
        return

    def forward(self, x) -> typing.Mapping[str, torch.Tensor]:
        ret_dict = {}
        for k, v in self.items():
            ret_dict[k] = v(x)

        return ret_dict


class NoiseEstimator(nn.ModuleDict):

    def __init__(self, n_feats: int) -> None:
        # We use different classfier for different noise types
        super().__init__()
        # Noise types (classification)
        # (Gaussian, Poisson)
        self['noise_type'] = nn.Linear(n_feats, 2)
        # Noise parameter (regression)
        self['noise_sigma_gaussian'] = nn.Linear(n_feats, 1)
        self['noise_scale_poisson'] = nn.Linear(n_feats, 1)

        # Gray noise (classification)
        # For simplicity, use 2-way classifier
        self['noise_gray_gaussian'] = nn.Linear(n_feats, 2)
        self['noise_gray_poisson'] = nn.Linear(n_feats, 2)
        return

    def forward(self, x) -> typing.Mapping[str, torch.Tensor]:
        ret_dict = {}
        for k, v in self.items():
            ret_dict[k] = v(x)

        return ret_dict


class JPEGEstimator(nn.ModuleDict):

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        # JPEG Q (regression)
        self['jpeg_q'] = nn.Linear(n_feats, 1)
        return

    def forward(self, x) -> typing.Mapping[str, torch.Tensor]:
        ret_dict = {}
        for k, v in self.items():
            ret_dict[k] = v(x)

        return ret_dict


class FinalEstimator(nn.ModuleDict):

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        # Check if JPEG compression comes first (classification)
        self['jpeg_first'] = nn.Linear(n_feats, 2)

        # Kernel type (classification)
        # (Sinc, Pulse)
        self['kernel_type_jf'] = nn.Linear(n_feats, 2)
        self['kernel_type_jl'] = nn.Linear(n_feats, 2)

        # Kernel size (classification)
        # Possible sizes: (7, 9, 11, 13, 15, 17, 19, 21)
        self['kernel_size_sinc_jf'] = nn.Linear(n_feats, 8)
        self['kernel_size_sinc_jl'] = nn.Linear(n_feats, 8)

        self['kernel_omega_sinc_jf'] = nn.Linear(n_feats, 1)
        self['kernel_omega_sinc_jl'] = nn.Linear(n_feats, 1)

        self['resize_type_jf'] = nn.Linear(n_feats, 3)
        self['resize_type_jl'] = nn.Linear(n_feats, 3)

        '''
        self['resize_scale_area_jf'] = nn.Linear(n_feats, 1)
        self['resize_scale_bilinear_jf'] = nn.Linear(n_feats, 1)
        self['resize_scale_bicubic_jf'] = nn.Linear(n_feats, 1)
        self['resize_scale_area_jl'] = nn.Linear(n_feats, 1)
        self['resize_scale_bilinear_jl'] = nn.Linear(n_feats, 1)
        self['resize_scale_bicubic_jl'] = nn.Linear(n_feats, 1)
        '''
        self['jpeg_q_jf'] = nn.Linear(n_feats, 1)
        self['jpeg_q_jl'] = nn.Linear(n_feats, 1)
        return

    def forward(self, x) -> typing.Mapping[str, torch.Tensor]:
        ret_dict = {}
        for k, v in self.items():
            ret_dict[k] = v(x)

        return ret_dict


class DegClassfier(models.ResNet):

    def __init__(self, no_srfeat: bool=False) -> None:
        block = resnet.Bottleneck
        # ResNet-50 configuration
        super().__init__(block, [3, 4, 6, 3])
        if no_srfeat:
            n_feats = 3
        else:
            n_feats = 15

        self.conv1 = nn.Conv2d(
            n_feats, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.deg_estimators = nn.ModuleDict()
        self.deg_estimators['kernel_1'] = KernelEstimator(512 * block.expansion)
        self.deg_estimators['resize_1'] = ResizeEstimator(512 * block.expansion)
        self.deg_estimators['noise_1'] = NoiseEstimator(512 * block.expansion)
        self.deg_estimators['jpeg'] = JPEGEstimator(512 * block.expansion)

        self.deg_estimators['kernel_2'] = KernelEstimator(512 * block.expansion)
        self.deg_estimators['resize_2'] = ResizeEstimator(512 * block.expansion)
        self.deg_estimators['noise_2'] = NoiseEstimator(512 * block.expansion)
        self.deg_estimators['final'] = FinalEstimator(512 * block.expansion)
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = {
            'no_srfeat': cfg.no_srfeat,
        }
        return kwargs

    def _forward_impl(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        ret_dict = {k: v(x) for k, v in self.deg_estimators.items()}
        ret_dict['features'] = x
        return ret_dict


if __name__ == '__main__':
    ke = KernelEstimator(16)
    ke.cuda()
    x = torch.randn(1, 16)
    x = x.cuda()
    y = ke(x)
    for k, v in y.items():
        print(k, v.size())


REPRESENTATIVE = DegClassfier
