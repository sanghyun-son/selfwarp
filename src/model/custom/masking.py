from torch import autograd
from torch.nn import functional as F
import numpy as np

class Masking(autograd.Function):

    @staticmethod
    def forward(ctx, x, mask, n_classes):
        ctx.mask = mask
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Boolean masking
        grad_x = grad_output * ctx.mask.float()
        return grad_x, None, None


class ClassMasking(autograd.Function):

    @staticmethod
    def forward(ctx, x, mask, n_classes):
        ctx.mask = mask
        ctx.n_classes = n_classes
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Boolean masking
        grad_x = ctx.n_classes * ctx.mask.float() * grad_output
        return grad_x, None, None


class LegacyScaledMasking(autograd.Function):
    '''
    Backpropagated gradients are scaled so that
    each classifier can be trained effectively.
    '''

    @staticmethod
    def forward(ctx, x, mask):
        '''
        Args:
            x (torch.Tensor): ... x H x W
            mask (torch.BoolTensor): ... x H x W
        '''
        ctx.mask = mask
        return x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Args:
            grad_output (torch.Tensor): ... x H x W
        '''
        eps = 1e-8
        mask = ctx.mask.float()
        n = mask.size(-1) * mask.size(-2)
        eff = mask.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        scaler = n / (eff + eps)
        # Boolean masking
        grad_x = scaler * mask * grad_output
        return grad_x, None


class ScaledMasking(autograd.Function):
    '''
    Backpropagated gradients are scaled so that
    each classifier can be trained effectively.
    '''

    @staticmethod
    def forward(ctx, x, mask, n_classes):
        '''
        Args:
            x (torch.Tensor): ... x H x W
            mask (torch.BoolTensor): ... x H x W
        '''
        ctx.mask = mask
        ctx.n_classes = n_classes
        return x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Args:
            grad_output (torch.Tensor): ... x H x W
        '''
        eps = 1e-8
        mask = ctx.mask.float()
        scaler = np.prod(mask.size()) / (mask.sum() + eps)
        # Boolean masking
        grad_x = ctx.n_classes * scaler * mask * grad_output
        return grad_x, None, None

class UnnormalizedScaledMasking(autograd.Function):
    '''
    Backpropagated gradients are scaled so that
    each classifier can be trained effectively.
    '''

    @staticmethod
    def forward(ctx, x, mask, n_classes):
        '''
        Args:
            x (torch.Tensor): ... x H x W
            mask (torch.BoolTensor): ... x H x W
        '''
        ctx.mask = mask
        return x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Args:
            grad_output (torch.Tensor): ... x H x W
        '''
        eps = 1e-8
        mask = ctx.mask.float()
        scaler = np.prod(mask.size()) / (mask.sum() + eps)
        # Boolean masking
        grad_x = scaler * mask * grad_output
        return grad_x, None, None


masking = Masking.apply
masking_c = ClassMasking.apply
masking_s = ScaledMasking.apply
masking_us = UnnormalizedScaledMasking.apply
masking_legacy = LegacyScaledMasking.apply

def loss_ref(x, y, mask):
    '''
    Args:
        x (torch.Tensor): B x C x H x W
        y (torch.LongTensor): B x C x H x W
        mask (torch.BoolTensor): B x C x H x W
    
    '''

    loss = []
    for i in range(x.size(1)):
        x_sub = x[:, i]         # B x H x W
        y_sub = y[:, i]         # B x H x W
        mask_sub = mask[:, i]   # B x H x W

        eff = x_sub[mask_sub]
        eff_label = y_sub[mask_sub]
        loss.append(
            F.binary_cross_entropy_with_logits(eff, eff_label)        
        )

    return sum(loss)

def loss_old(x, y):
    return F.binary_cross_entropy_with_logits(x, y)

def main():
    import torch
    from torch import nn
    n_classes = 3
    b = 2
    c = 3
    h = 4
    w = 4
    torch.manual_seed(20191016)
    x = torch.randn(b, c, h, w)
    x.requires_grad = True
    mask = torch.randn(b, n_classes, h, w) > 0
    mask_split = mask.split(1, dim=1)

    multi_cls = [nn.Conv2d(c, 1, 1, padding=0) for _ in range(n_classes)]
    label = [cls(x) for cls in multi_cls]
    label = [masking_s(l, m, n_classes) for l, m in zip(label, mask_split)]
    label = torch.cat(label, dim=1)

    gt = torch.ones_like(label)
    #ref = loss_ref(label, gt, mask)
    #ref.backward()
    #print(multi_cls[0].weight.grad)
    #print(ref.item())
    old = loss_old(label, gt)
    old.backward()
    print(multi_cls[0].weight.grad)
    print(old.item())

if __name__ == '__main__':
    main()

