import typing

import torch
from torch import autograd
from torch.autograd import function

import srwarp_cuda

_T = typing.Optional[torch.Tensor]

def projective_grid(
        sizes: typing.Tuple[int, int],
        m: torch.Tensor,
        eps_y: float=0,
        eps_x: float=0) -> torch.Tensor:

    '''
    Args:
        sizes (int, int): Target domain size.
        m (torch.Tensor): Target to source transform.
        eps_y (float, optional): Perturbation along y-axis.
        eps_x (float, optional): Perturbation alogn x-axis.
    '''
    # Must be done on GPU
    m = m.cuda().float()
    grid = m.new(sizes[0] * sizes[1], 2)
    srwarp_cuda.projective_grid(m, sizes[0], sizes[1], grid, eps_y, eps_x)
    grid = grid.t().contiguous()
    return grid


class SVF(autograd.Function):

    @staticmethod
    def forward(
            ctx: function._ContextMethodMixin,
            x: torch.Tensor,
            weight: torch.Tensor,
            sizes: typing.Tuple[int, int],
            k: int,
            xi: torch.Tensor,
            yi: torch.Tensor,
            fill_value: float,
            is_half: bool) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                'x should be 4-dim Tensor! (got {})'.format(x.dim()),
            )

        wk = weight.size(-1)
        if wk != k**2:
            raise ValueError(
                'Incorrect kernel size! (Expected {}, got {})'.format(
                    k**2, wk,
                ),
            )

        if weight.dim() == 3:
            wc = weight.size(1)
            xc = x.size(1)
            if wc != 1 and wc != xc:
                raise ValueError(
                    'Incorrect weight channels! (Expected 1 or {}, got {})'.format(
                        xc, wc,
                    )
                )

        if is_half:
            x = x.half()
            weight = weight.half()

        xi = xi.int()
        yi = yi.int()
        # Backup for backpropagation
        ctx.save_for_backward(x, weight, xi, yi)
        ctx.k = k
        ctx.is_half = is_half
        # Return memory allocation
        y = x.new_full((x.size(0), x.size(1), *sizes), fill_value)
        srwarp_cuda.forward(x, weight, y, k, xi, yi, is_half)
        return y

    @staticmethod
    def backward(
            ctx: function._ContextMethodMixin,
            grad_output: torch.Tensor) -> typing.List[_T]:

        x, weight, xi, yi = ctx.saved_tensors
        k = ctx.k
        is_half = ctx.is_half
        if is_half:
            grad_output = grad_output.half()

        # Return memory allocation
        dx = torch.zeros_like(x)
        dweight = torch.zeros_like(weight)
        srwarp_cuda.backward(
            x, dx, weight, dweight, grad_output, k, xi, yi, is_half,
        )
        return dx, dweight, None, None, None, None, None, None, None, None, None


svf_forward = SVF.apply

