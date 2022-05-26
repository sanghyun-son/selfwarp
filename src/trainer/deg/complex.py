import random

import types
import typing
import numpy as np

import torch
from torch.nn import functional as F
import tqdm

from model.sr import edsr
from trainer import base_trainer
from misc import downloader
from ops import filters
from diff_jpeg import DiffJPEG
import deg_pipeline

import torch

_parent_class = base_trainer.BaseTrainer


class DegPredictor(_parent_class):

    def __init__(
            self,
            *args: typing.Optional[typing.List[typing.Any]],
            no_srfeat: bool=False,
            normalize: bool=False,
            synth_output: bool=False,
            test_specific: bool=False,
            gaussian_only: bool=False,
            no_random_resize: bool=False,
            get_features: bool=False,
            use_predefined: bool=False,
            save_pred: bool=False,
            test_lr: bool=False,
            **kwargs: typing.Optional[typing.Mapping[str, typing.Any]]) -> None:

        super().__init__(*args, **kwargs)
        self.scale = 2
        pretrained = downloader.download(f'edsr-baseline-x{self.scale}')
        self.net_sr = edsr.EDSR(scale=self.scale)
        self.net_sr.eval()
        self.net_sr.cuda()
        self.net_sr.load_state_dict(pretrained['model'], strict=True)

        cfg = deg_pipeline.parse_yaml()
        self.kernel_synthesizer = deg_pipeline.Kernel(
            cfg,
            gaussian_only=gaussian_only,
            no_random_resize=no_random_resize,
        )
        self.deg_synthesizer = deg_pipeline.Degrader(
            cfg,
            no_random_resize=no_random_resize,
        )

        self.no_srfeat = no_srfeat
        self.normalize = normalize
        self.synth_output = synth_output
        self.test_specific = test_specific

        self.get_features = get_features
        if get_features:
            self.list_features = []
            self.list_labels = []
        else:
            self.list_features = None
            self.list_labels = None

        if use_predefined:
            self.use_predefined = torch.load('degradations.pth')
        else:
            self.use_predefined = None

        self.save_pred = save_pred
        if save_pred:
            self.list_pred = []
            self.list_labels = []
        else:
            self.list_pred = None
            self.list_labels = None

        self.test_lr = test_lr
        self.global_idx = 0
        return

    @staticmethod
    def get_kwargs(cfg: types.SimpleNamespace) -> dict:
        kwargs = _parent_class.get_kwargs(cfg)
        kwargs['no_srfeat'] = cfg.no_srfeat
        kwargs['normalize'] = cfg.normalize
        kwargs['synth_output'] = cfg.synth_output
        kwargs['test_specific'] = cfg.test_specific

        kwargs['gaussian_only'] = cfg.gaussian_only
        kwargs['no_random_resize'] = cfg.no_random_resize

        kwargs['get_features'] = cfg.get_features
        kwargs['use_predefined'] = cfg.use_predefined
        kwargs['save_pred'] = cfg.save_pred
        kwargs['test_lr'] = cfg.test_lr
        return kwargs

    def _normalize(self, v: float, vmin: float, vmax: float) -> float:
        # Normalize to -1 ~ 1
        ret = 2 * (v - vmin) / (vmax - vmin) - 1
        return ret

    def _unnormalize(self, v: float, vmin: float, vmax: float) -> float:
        # Normalize to the original range
        ret = (v + 1) * (vmax - vmin) / 2 + vmin
        return ret

    def _normalize_labels(self, labels: typing.List[dict]) -> dict:
        for label in labels:
            for k, v in label.items():
                if k == 'scale_1':
                    label[k] = self._normalize(v, 0.15, 1.5)
                elif k == 'scale_2':
                    label[k] = self._normalize(v, 0.3, 1.2)
                elif 'noise_param' in k:
                    param_dict = {'gray_noise': v['gray_noise']}
                    if 'sigma' in v:
                        if '1' in k:
                            param_dict['sigma'] = self._normalize(
                                v['sigma'],
                                1,
                                30,
                            )
                        elif '2' in k:
                            param_dict['sigma'] = self._normalize(
                                v['sigma'],
                                1,
                                25,
                            )

                    elif 'scale' in v:
                        if '1' in k:
                            param_dict['scale'] = self._normalize(
                                v['scale'],
                                0.05,
                                3,
                            )
                        elif '2' in k:
                            param_dict['scale'] = self._normalize(
                                v['scale'],
                                0.05,
                                2.5,
                            )

                    label[k] = param_dict

                elif 'kernel_param' in k:
                    param_dict = {}
                    if 'omega_c' in v:
                        param_dict['omega_c'] = self._normalize(
                            v['omega_c'],
                            np.pi / 3,
                            np.pi,
                        )
                    elif 'final' not in k:
                        param_dict['sigma_matrix'] = v['sigma_matrix']

                    if 'beta' in v:
                        if '1' in k:
                            kernel_idx = 1
                        elif '2' in k:
                            kernel_idx = 2

                        if 'plateau' in label[f'kernel_type_{kernel_idx}']:
                            param_dict['beta'] = self._normalize(
                                v['beta'],
                                1,
                                2,
                            )
                        else:
                            param_dict['beta'] = self._normalize(
                                v['beta'],
                                0.5,
                                4,
                            )

                    label[k] = param_dict

                elif 'jpeg_q' in k:
                    label[k] = self._normalize(v, 30, 95)

        return labels

    def _unnormalize_labels(self, pred: dict) -> dict:
        for k, v in pred.items():
            if k == 'scale_1':
                pred[k] = self._unnormalize(v, 0.15, 1.5)
            elif k == 'scale_2':
                pred[k] = self._unnormalize(v, 0.3, 1.2)
            elif 'noise_param' in k:
                param_dict = {'gray_noise': v['gray_noise']}
                if 'sigma' in v:
                    if '1' in k:
                        param_dict['sigma'] = self._unnormalize(
                            v['sigma'],
                            1,
                            30,
                        )
                    elif '2' in k:
                        param_dict['sigma'] = self._unnormalize(
                            v['sigma'],
                            1,
                            25,
                        )

                elif 'scale' in v:
                    if '1' in k:
                        param_dict['scale'] = self._unnormalize(
                            v['scale'],
                            0.05,
                            3,
                        )
                    elif '2' in k:
                        param_dict['scale'] = self._unnormalize(
                            v['scale'],
                            0.05,
                            2.5,
                        )

                pred[k] = param_dict

            elif 'kernel_param' in k:
                param_dict = {}
                if 'omega_c' in v:
                    param_dict['omega_c'] = self._unnormalize(
                        v['omega_c'],
                        np.pi / 3,
                        np.pi,
                    )
                elif 'final' not in k:
                    param_dict['sigma_matrix'] = v['sigma_matrix']

                if 'beta' in v:
                    if '1' in k:
                        kernel_idx = 1
                    elif '2' in k:
                        kernel_idx = 2

                    if 'plateau' in pred[f'kernel_type_{kernel_idx}']:
                        param_dict['beta'] = self._unnormalize(
                            v['beta'],
                            1,
                            2,
                        )
                    else:
                        param_dict['beta'] = self._unnormalize(
                            v['beta'],
                            0.5,
                            4,
                        )

                pred[k] = param_dict

            elif 'jpeg_q' in k:
                pred[k] = self._unnormalize(v, 30, 95)

        return pred

    def _get_label(self, pred: torch.Tensor) -> int:
        _, idx = pred.max(dim=1)
        idx = idx.item()
        return idx

    def _get_value(self, pred: torch.Tensor) -> float:
        return pred.item()

    def _get_sigma_matrix(self, pred: torch.Tensor) -> np.array:
        arr = np.array(
            [[pred[0, 0].item(), pred[0, 1].item()],
            [pred[0, 2].item(), pred[0, 3].item()]],
            dtype=np.float32,
        )
        return arr

    def _pred2dict(self, pred: dict) -> dict:
        meta_dict = {}
        for k, v in pred.items():
            if '1' in k:
                idx = 1
            elif '2' in k:
                idx = 2
            else:
                idx = 'final'

            if 'kernel' in k:
                is_blur = self._get_label(v['is_blur'])
                if is_blur == 0:
                    meta_dict['second_blur'] = False
                else:
                    meta_dict['second_blur'] = True

                    kernel_type_idx = self._get_label(v['kernel_type'])
                    if kernel_type_idx == 0:
                        kernel_type = 'sinc'
                        kernel_str = 'sinc'
                    elif kernel_type_idx == 1:
                        kernel_type = 'aniso'
                        kernel_str = 'gaussian'
                    elif kernel_type_idx == 2:
                        kernel_type = 'generalized_aniso'
                        kernel_str = 'ggaussian'
                    elif kernel_type_idx == 3:
                        kernel_type = 'plateau_aniso'
                        kernel_str = 'plateau'
                    else:
                        raise ValueError('Wrong kernel type index!')

                    meta_dict[f'kernel_type_{idx}'] = kernel_type

                    kernel_size_idx = self._get_label(
                        v[f'kernel_size_{kernel_str}'],
                    )
                    kernel_size = 2 * kernel_size_idx + 7
                    meta_dict[f'kernel_size_{idx}'] = kernel_size

                    kernel_param = {}
                    if kernel_type == 'sinc':
                        kernel_param['omega_c'] = self._get_value(v['omega_c'])
                    else:
                        sigma_matrix = self._get_sigma_matrix(
                            v[f'kernel_sigma_{kernel_str}'],
                        )
                        kernel_param['sigma_matrix'] = sigma_matrix
                        if 'generalized' in kernel_type or 'plateau' in kernel_type:
                            beta = self._get_value(v[f'kernel_beta_{kernel_str}'])
                            kernel_param['beta'] = beta

                    meta_dict[f'kernel_param_{idx}'] = kernel_param

            elif 'jpeg' in k:
                meta_dict['jpeg_q_1'] = self._get_value(v['jpeg_q'])
            elif 'noise' in k:
                noise_type_idx = self._get_label(v['noise_type'])
                if noise_type_idx == 0:
                    noise_type = 'gaussian'
                    noise_param_str = 'sigma'
                elif noise_type_idx == 1:
                    noise_type = 'poisson'
                    noise_param_str = 'scale'
                else:
                    raise ValueError('Wrong noise type index!')

                meta_dict[f'noise_type_{idx}'] = noise_type
                noise_param = {
                    noise_param_str: self._get_value(
                        v[f'noise_{noise_param_str}_{noise_type}'],
                    ),
                    'gray_noise': self._get_label(v[f'noise_gray_{noise_type}']),
                }
                meta_dict[f'noise_param_{idx}'] = noise_param
            elif 'resize' in k:
                resize_type_idx = self._get_label(v['resize_type'])
                if resize_type_idx == 0:
                    resize_type = 'bilinear'
                elif resize_type_idx == 1:
                    resize_type = 'bicubic'
                elif resize_type_idx == 2:
                    resize_type = 'area'
                else:
                    raise ValueError('Wrong resize type index!')

                meta_dict[f'resize_{idx}'] = resize_type
                meta_dict[f'scale_{idx}'] = self._get_value(
                    v[f'resize_scale_{resize_type}'],
                )
            elif 'final' in k:
                is_jpeg_first = self._get_label(v['jpeg_first'])
                if is_jpeg_first == 0:
                    postfix = 'jl'
                elif is_jpeg_first == 1:
                    postfix = 'jf'
                else:
                    raise ValueError('Wrong jpeg_first flag!')

                meta_dict['final_jpeg_first'] = (is_jpeg_first == 1)

                kernel_type_idx = self._get_label(v[f'kernel_type_{postfix}'])
                if kernel_type_idx == 0:
                    kernel_type = 'sinc'
                elif kernel_type_idx == 1:
                    kernel_type = 'pulse'
                else:
                    raise ValueError('Wrong kernel type index!')

                meta_dict['kernel_type_final'] = kernel_type

                kernel_param = {}
                if kernel_type == 'sinc':
                    kernel_size_idx = self._get_label(
                        v[f'kernel_size_sinc_{postfix}'],
                    )
                    kernel_size = 2 * kernel_size_idx + 7
                    meta_dict[f'kernel_size_final'] = kernel_size
                    kernel_param['omega_c'] = self._get_value(
                        v[f'kernel_omega_sinc_{postfix}'],
                    )

                meta_dict['kernel_param_final'] = kernel_param

                resize_type_idx = self._get_label(v[f'resize_type_{postfix}'])
                if resize_type_idx == 0:
                    resize_type = 'bilinear'
                elif resize_type_idx == 1:
                    resize_type = 'bicubic'
                elif resize_type_idx == 2:
                    resize_type = 'area'
                else:
                    raise ValueError('Wrong resize type index!')

                meta_dict['resize_final'] = resize_type
                meta_dict['jpeg_q_final'] = self._get_value(
                    v[f'jpeg_q_{postfix}']
                )

        return meta_dict

    def forward(
            self,
            **samples: dict) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        if self.test_lr:
            lrs = samples['img']
            meta_dicts = None
            label = int(samples['name'][0])
        else:
            hrs = samples['img']
            hrs = (hrs + 1) / 2

            lrs = []
            meta_dicts = []
            for hr in hrs.split(1, dim=0):
                if self.use_predefined is None:
                    kernel_dict, meta_dict_kernel = self.kernel_synthesizer.get_kernel()
                    lr, meta_dict_deg = self.deg_synthesizer(hr, **kernel_dict)
                    meta_dict = {**meta_dict_kernel, **meta_dict_deg}
                    label = 0
                else:
                    label = random.randrange(10)
                    meta_dict = self.use_predefined[label]
                    kernel_dict = self.kernel_synthesizer.get_kernel_from_meta(meta_dict)
                    lr = self.deg_synthesizer.degrade_from_meta(
                        hr,
                        meta_dict,
                        kernel_dict['kernel1'],
                        kernel_dict['kernel2'],
                        kernel_dict['sinc_kernel'],
                    )

                lrs.append(lr)
                meta_dicts.append(meta_dict)

            if self.normalize:
                meta_dicts = self._normalize_labels(meta_dicts)

            lrs = torch.cat(lrs, dim=0)
            lrs = 2 * lrs - 1

        #self.pause(count_max=10, hr=(2 * hrs - 1), lr=lrs)

        if self.no_srfeat:
            x = lrs
        else:
            sr = self.net_sr(lrs)
            sr_unshuffle = F.pixel_unshuffle(sr, self.scale)
            x = torch.cat((sr_unshuffle, lrs), dim=1)

        pred = self.pforward(x)
        if self.get_features:
            features = pred.pop('features')
            self.list_features.append(features)
            self.list_labels.append(label)

            if self.save_pred:
                pred = self._pred2dict(pred)
                pred = self._unnormalize_labels(pred)
                self.list_pred.append(pred)

        if meta_dicts is None:
            loss = 0
        else:
            loss = self.loss(
                pred=pred,
                label=meta_dicts,
            )

        if not self.training and self.synth_output:
            meta_dict_pred = self._pred2dict(pred)

            kernel_dict = self.kernel_synthesizer.get_kernel_from_meta(
                meta_dict_pred,
            )
            lr_synth = self.deg_synthesizer.degrade_from_meta(
                hrs, meta_dict_pred, **kernel_dict,
            )
            lr_synth = 2 * lr_synth - 1
            hrs = 2 * hrs - 1
            ret_dict = {
                'lr_synth': lr_synth,
                'hrs': hrs,
                'lrs': lrs,
            }
        else:
            ret_dict = {}

        return loss, ret_dict

    def evaluation(self) -> None:
        super().evaluation()
        save_dict = {
            'features': self.list_features,
            'labels': self.list_labels,
        }
        torch.save(save_dict, 'complex_normalize_schedule_gg_nr_add.pth')

        save_dict_pred = {}
        for p, l in zip(self.list_pred, self.list_labels):
            save_dict_pred[l] = p

        torch.save(save_dict_pred, 'complex_normalize_schedule_gg_nr_pred.pth')
        return
