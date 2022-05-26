import typing

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class Degradation(nn.Module):

    def __init__(self, name: 'degradation') -> None:
        super().__init__()
        self.name = name
        return

    def _make_logits(self, idx: int) -> torch.Tensor:
        logits = torch.tensor(
            [idx],
            dtype=torch.long,
            device=torch.device('cuda'),
        )
        return logits

    def _make_params(self, value: float) -> torch.Tensor:
        params = torch.tensor(
            [[value]],
            dtype=torch.float32,
            device=torch.device('cuda'),
        )
        return params

    def _make_params_sigma(self, sigma_matrix: np.array) -> torch.Tensor:
        sigma_matrix = torch.from_numpy(sigma_matrix).view(1, 4)
        sigma_matrix = sigma_matrix.float()
        sigma_matrix = sigma_matrix.cuda()
        return sigma_matrix

    def _forward_blur(
            self,
            idx: int,
            pred: dict,
            label: dict,
            blur_idx: int) -> torch.Tensor:

        kernel_pred = pred[f'kernel_{blur_idx}']

        ignore_blur = False
        if blur_idx == 2:
            if label['second_blur']:
                kernel_is_blur_logits = 1
            else:
                kernel_is_blur_logits = 0
                ignore_blur = True
        else:
            kernel_is_blur_logits = 1

        kernel_is_blur_logits = self._make_logits(kernel_is_blur_logits)
        kernel_is_blur_pred = kernel_pred['is_blur'][idx]
        kernel_is_blur_pred = kernel_is_blur_pred.unsqueeze(0)
        kernel_is_blur_loss = F.cross_entropy(kernel_is_blur_pred, kernel_is_blur_logits)
        loss_dict = {
            'kernel_is_blur': kernel_is_blur_loss,
        }

        if ignore_blur:
            return loss_dict

        kernel_size_gt = label[f'kernel_size_{blur_idx}']
        kernel_type_gt = label[f'kernel_type_{blur_idx}']
        kernel_param_dict_gt = label[f'kernel_param_{blur_idx}']

        if kernel_type_gt == 'sinc':
            kernel_type_logits = 0
        elif kernel_type_gt == 'iso' or kernel_type_gt == 'aniso':
            kernel_type_logits = 1
        elif 'generalized' in kernel_type_gt:
            kernel_type_logits = 2
        elif 'plateau' in kernel_type_gt:
            kernel_type_logits = 3

        # (1, 1)
        kernel_type_logits = self._make_logits(kernel_type_logits)
        kernel_type_pred = kernel_pred['kernel_type'][idx]
        kernel_type_pred = kernel_type_pred.unsqueeze(0)
        kernel_type_loss = F.cross_entropy(kernel_type_pred, kernel_type_logits)

        kernel_size_logits = (kernel_size_gt - 7) // 2
        kernel_size_logits = self._make_logits(kernel_size_logits)
        if kernel_type_gt == 'sinc':
            kernel_size_pred = kernel_pred['kernel_size_sinc'][idx]
        elif kernel_type_gt == 'iso' or kernel_type_gt == 'aniso':
            kernel_size_pred = kernel_pred['kernel_size_gaussian'][idx]
        elif 'generalized' in kernel_type_gt:
            kernel_size_pred = kernel_pred['kernel_size_ggaussian'][idx]
        elif 'plateau' in kernel_type_gt:
            kernel_size_pred = kernel_pred['kernel_size_plateau'][idx]

        kernel_size_pred = kernel_size_pred.unsqueeze(0)
        kernel_size_loss = F.cross_entropy(kernel_size_pred, kernel_size_logits)

        if kernel_type_gt == 'sinc':
            kernel_param_pred = kernel_pred['kernel_omega_sinc'][idx]
            kernel_sigma_pred = None
            kernel_param_gt = kernel_param_dict_gt['omega_c']
            kernel_sigma_gt = None
        elif kernel_type_gt == 'iso' or kernel_type_gt == 'aniso':
            kernel_param_pred = None
            kernel_sigma_pred = kernel_pred['kernel_sigma_gaussian'][idx]
            kernel_param_gt = None
            kernel_sigma_gt = kernel_param_dict_gt['sigma_matrix']
        elif 'generalized' in kernel_type_gt:
            kernel_param_pred = kernel_pred['kernel_beta_ggaussian'][idx]
            kernel_sigma_pred = kernel_pred['kernel_sigma_ggaussian'][idx]
            kernel_param_gt = kernel_param_dict_gt['beta']
            kernel_sigma_gt = kernel_param_dict_gt['sigma_matrix']
        elif 'plateau' in kernel_type_gt:
            kernel_param_pred = kernel_pred['kernel_beta_plateau'][idx]
            kernel_sigma_pred = kernel_pred['kernel_sigma_plateau'][idx]
            kernel_param_gt = kernel_param_dict_gt['beta']
            kernel_sigma_gt = kernel_param_dict_gt['sigma_matrix']

        if kernel_param_gt is not None:
            kernel_param_pred = kernel_param_pred.unsqueeze(0)
            kernel_param_gt = self._make_params(kernel_param_gt)
            kernel_param_loss = F.mse_loss(kernel_param_pred, kernel_param_gt)
        else:
            kernel_param_loss = 0

        if kernel_sigma_gt is not None:
            kernel_sigma_pred = kernel_sigma_pred.unsqueeze(0)
            kernel_sigma_gt = self._make_params_sigma(kernel_sigma_gt)
            kernel_sigma_loss = F.mse_loss(kernel_sigma_pred, kernel_sigma_gt)
        else:
            kernel_sigma_loss = 0

        kernel_param_total_loss = kernel_param_loss + kernel_sigma_loss
        loss_dict['kernel_type'] = kernel_type_loss
        loss_dict['kernel_size'] = kernel_size_loss
        loss_dict['kernel_param'] = kernel_param_total_loss
        return loss_dict

    def _forward_noise(
            self,
            idx: int,
            pred: dict,
            label: dict,
            noise_idx: int) -> torch.Tensor:

        noise_pred = pred[f'noise_{noise_idx}']
        noise_type_gt = label[f'noise_type_{noise_idx}']
        noise_param_dict_gt = label[f'noise_param_{noise_idx}']
        noise_is_gray_gt = noise_param_dict_gt['gray_noise']

        if noise_type_gt == 'gaussian':
            noise_type_logits = 0
        elif noise_type_gt == 'poisson':
            noise_type_logits = 1

        noise_type_logits = self._make_logits(noise_type_logits)
        noise_type_pred = noise_pred['noise_type'][idx]
        noise_type_pred = noise_type_pred.unsqueeze(0)
        noise_type_loss = F.cross_entropy(noise_type_pred, noise_type_logits)

        if noise_type_gt == 'gaussian':
            noise_param_pred = noise_pred['noise_sigma_gaussian'][idx]
            noise_param_gt = noise_param_dict_gt['sigma']
        elif noise_type_gt == 'poisson':
            noise_param_pred = noise_pred['noise_scale_poisson'][idx]
            noise_param_gt = noise_param_dict_gt['scale']

        noise_param_gt = self._make_params(noise_param_gt)
        noise_param_loss = F.mse_loss(noise_param_pred, noise_param_gt)

        if noise_is_gray_gt == 0.0:
            noise_is_gray_logits = 0
        else:
            noise_is_gray_logits = 1

        if noise_type_gt == 'gaussian':
            noise_is_gray_pred = noise_pred['noise_gray_gaussian'][idx]
        elif noise_type_gt == 'poisson':
            noise_is_gray_pred = noise_pred['noise_gray_poisson'][idx]

        noise_is_gray_logits = self._make_logits(noise_is_gray_logits)
        noise_is_gray_pred = noise_is_gray_pred.unsqueeze(0)
        noise_is_gray_loss = F.cross_entropy(noise_is_gray_pred, noise_is_gray_logits)

        noise_param_total_loss = noise_param_loss + noise_is_gray_loss
        loss_dict = {
            'noise_type': noise_type_loss,
            'noise_param': noise_param_total_loss,
        }
        return loss_dict

    def _forward_jpeg(
            self,
            idx: int,
            pred: dict,
            label: dict,
            jpeg_idx: int) -> torch.Tensor:

        jpeg_pred = pred['jpeg']

        jpeg_q_param_gt = self._make_params(label[f'jpeg_q_{jpeg_idx}'])
        jpeg_q_param_pred = jpeg_pred['jpeg_q'][idx]
        jpeg_q_param_pred = jpeg_q_param_pred.unsqueeze(0)
        jpeg_q_loss = F.mse_loss(jpeg_q_param_pred, jpeg_q_param_gt)

        loss_dict = {
            'jpeg_q': jpeg_q_loss,
        }
        return loss_dict
        
    def _forward_resize(
            self,
            idx: int,
            pred: dict,
            label: dict,
            resize_idx: int) -> torch.Tensor:

        resize_pred = pred[f'resize_{resize_idx}']

        resize_type_gt = label[f'resize_{resize_idx}']
        if resize_type_gt == 'bilinear':
            resize_type_logits = 0
            resize_scale_pred = resize_pred['resize_scale_bilinear'][idx]
        elif resize_type_gt == 'bicubic':
            resize_type_logits = 1
            resize_scale_pred = resize_pred['resize_scale_bicubic'][idx]
        elif resize_type_gt == 'area':
            resize_type_logits = 2
            resize_scale_pred = resize_pred['resize_scale_area'][idx]

        resize_type_logits = self._make_logits(resize_type_logits)
        resize_type_pred = resize_pred['resize_type'][idx]
        resize_type_pred = resize_type_pred.unsqueeze(0)
        resize_type_loss = F.cross_entropy(resize_type_pred, resize_type_logits)

        resize_scale_gt = label[f'scale_{resize_idx}']
        resize_scale_gt = self._make_params(resize_scale_gt)
        resize_scale_loss = F.mse_loss(resize_scale_pred, resize_scale_gt)

        loss_dict = {
            'resize_type': resize_type_loss,
            'resize_scale': resize_scale_loss,
        }
        return loss_dict

    def _forward_final(
            self,
            idx: int,
            pred: dict,
            label: dict) -> torch.Tensor:

        loss_dict = {}
        final_pred = pred[f'final']
        final_is_jpeg_first_gt = label['final_jpeg_first']
        if final_is_jpeg_first_gt:
            final_is_jpeg_first_logits = 1
            postfix = 'jf'
        else:
            final_is_jpeg_first_logits = 0
            postfix = 'jl'

        final_is_jpeg_first_logits = self._make_logits(final_is_jpeg_first_logits)
        final_is_jpeg_first_pred = final_pred['jpeg_first'][idx]
        final_is_jpeg_first_pred = final_is_jpeg_first_pred.unsqueeze(0)
        final_is_jpeg_first_loss = F.cross_entropy(
            final_is_jpeg_first_pred,
            final_is_jpeg_first_logits,
        )

        final_kernel_type_gt = label['kernel_type_final']
        if final_kernel_type_gt == 'sinc':
            final_kernel_type_logits = 0
        else:
            final_kernel_type_logits = 1

        final_kernel_type_logits = self._make_logits(final_kernel_type_logits)
        final_kernel_type_pred = final_pred[f'kernel_type_{postfix}'][idx]
        final_kernel_type_pred = final_kernel_type_pred.unsqueeze(0)
        final_kernel_type_loss = F.cross_entropy(
            final_kernel_type_pred,
            final_kernel_type_logits,
        )

        if final_kernel_type_gt == 'sinc':
            final_kernel_size_gt = label['kernel_size_final']
            final_kernel_size_logits = (final_kernel_size_gt - 7) // 2
            final_kernel_size_logits = self._make_logits(final_kernel_size_logits)
            final_kernel_size_pred = final_pred[f'kernel_size_sinc_{postfix}'][idx]
            final_kernel_size_pred = final_kernel_size_pred.unsqueeze(0)
            final_kernel_size_loss = F.cross_entropy(
                final_kernel_size_pred,
                final_kernel_size_logits,
            )

            final_kernel_omega_pred = final_pred[f'kernel_omega_sinc_{postfix}'][idx]
            final_kernel_omega_pred = final_kernel_omega_pred.unsqueeze(0)
            final_kernel_param_gt = label['kernel_param_final']
            final_kernel_omega_gt = final_kernel_param_gt['omega_c']
            final_kernel_omega_gt = self._make_params(final_kernel_omega_gt)
            final_kernel_omega_loss = F.mse_loss(
                final_kernel_omega_pred,
                final_kernel_omega_gt,
            )
            loss_dict['kernel_size'] = final_kernel_size_loss
            loss_dict['kernel_omega'] = final_kernel_omega_loss

        final_resize_type_gt = label['resize_final']
        if final_resize_type_gt == 'bilinear':
            final_resize_type_logits = 0
        elif final_resize_type_gt == 'bicubic':
            final_resize_type_logits = 1
        elif final_resize_type_gt == 'area':
            final_resize_type_logits = 2

        final_resize_type_logits = self._make_logits(final_resize_type_logits)
        final_resize_type_pred = final_pred[f'resize_type_{postfix}'][idx]
        final_resize_type_pred = final_resize_type_pred.unsqueeze(0)
        final_resize_type_loss = F.cross_entropy(
            final_resize_type_pred,
            final_resize_type_logits,
        )

        final_jpeg_q_gt = label['jpeg_q_final']
        final_jpeg_q_gt = self._make_params(final_jpeg_q_gt)
        final_jpeg_q_pred = final_pred[f'jpeg_q_{postfix}'][idx]
        final_jpeg_q_pred = final_jpeg_q_pred.unsqueeze(0)
        final_jpeg_q_loss = F.mse_loss(
            final_jpeg_q_pred,
            final_jpeg_q_gt,
        )

        loss_dict['jpeg_first'] = final_is_jpeg_first_loss
        loss_dict['kernel_type'] = final_kernel_type_loss
        loss_dict['resize_type'] = final_resize_type_loss
        loss_dict['jpeg_q'] = final_jpeg_q_loss
        return loss_dict

    def _forward_split_batch(
            self,
            idx: int,
            pred: dict,
            label: dict) -> torch.Tensor:

        # First blur kernel
        loss_dict_kernel_1 = self._forward_blur(idx, pred, label, 1)
        loss_dict_kernel_2 = self._forward_blur(idx, pred, label, 2)

        loss_dict_noise_1 = self._forward_noise(idx, pred, label, 1)
        loss_dict_noise_2 = self._forward_noise(idx, pred, label, 2)

        loss_dict_resize_1 = self._forward_resize(idx, pred, label, 1)
        loss_dict_resize_2 = self._forward_resize(idx, pred, label, 2)

        loss_dict_jpeg = self._forward_jpeg(idx, pred, label, 1)
        loss_dict_final = self._forward_final(idx, pred, label)

        ret_dict = {
            'kernel_1': loss_dict_kernel_1,
            'kernel_2': loss_dict_kernel_2,
            'noise_1': loss_dict_noise_1,
            'noise_2': loss_dict_noise_2,
            'resize_1': loss_dict_resize_1,
            'resize_2': loss_dict_resize_2,
            'jpeg': loss_dict_jpeg,
            'final': loss_dict_final,
        }
        return ret_dict

    def forward(self, pred: dict, label: typing.List[dict]) -> torch.Tensor:
        loss_list = []
        for idx, sub_label in enumerate(label):
            loss = self._forward_split_batch(idx, pred, sub_label)
            loss_list.append(loss)

        loss_dict = {}
        loss_total = []
        for idx, loss in enumerate(loss_list):
            if idx == len(loss_list) - 1:
                is_last = True
            else:
                is_last = False

            for k1 in loss.keys():
                if k1 not in loss_dict:
                    loss_dict[k1] = {}

                for k2 in loss[k1].keys():
                    if k2 not in loss_dict[k1]:
                        loss_dict[k1][k2] = []

                    loss_dict[k1][k2].append(loss[k1][k2])
                    if is_last:
                        n = len(loss_dict[k1][k2])
                        loss_k1_k2 = sum(loss_dict[k1][k2]) / n
                        loss_total.append(loss_k1_k2)

        loss_total = sum(loss_total)
        return loss_total
