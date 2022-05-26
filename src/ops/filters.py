import math
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

import tqdm
from PIL import Image
from sklearn import linear_model

def gaussian_kernel(sigma: float=1, kernel_size: int=None) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = 2 * math.ceil(3 * sigma) + 1

    k = kernel_size // 2
    if kernel_size % 2 == 0:
        r = torch.linspace(-k + 0.5, k - 0.5, kernel_size)
    else:
        r = torch.linspace(-k, k, kernel_size)

    r = r.view(1, -1)
    r = r.repeat(kernel_size, 1)
    r = r**2
    # Squared distance from origin
    r = r + r.t()

    exp = -r / (2 * sigma**2)
    coeff = exp.exp()
    coeff = coeff / coeff.sum()
    return coeff

def filtering(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k = k.to(x.device)
    kh, kw = k.size()
    if x.dim() == 3:
        x = x.unsqueeze(0)

    if x.dim() == 4:
        c = x.size(1)
        k = k.view(1, 1, kh, kw)
        k = k.repeat(c, c, 1, 1)
        e = torch.eye(c, c)
        e = e.to(x.device)
        e = e.view(c, c, 1, 1)
        k *= e
    else:
        raise ValueError('x.dim() == {}! It should be 3 or 4.'.format(x.dim()))

    x = F.pad(x, (kh // 2, kh // 2, kw // 2, kw // 2), mode='replicate')
    y = F.conv2d(x, k, padding=0)
    return y

def downsampling(x: torch.Tensor, k: torch.Tensor, scale: int) -> torch.Tensor:
    x_f = filtering(x, k)
    # Offset for nearest downsampling
    offset = scale - 1
    x_f = F.pad(x_f[..., offset:, offset:], pad=(0, offset, 0, offset))
    y = F.interpolate(
        x_f,
        scale_factor=(1 / scale),
        mode='nearest',
        recompute_scale_factor=False,
    )
    return y

def gaussian_filtering(x: torch.Tensor, sigma: float=1) -> torch.Tensor:
    k = gaussian_kernel(sigma=sigma)
    y = filtering(x, k)
    return y

@torch.no_grad()
def find_kernel(
        x: torch.Tensor,
        y: torch.Tensor,
        scale: int,
        k: int,
        max_patches: int=-1,
        threshold: float=1e-5,
        lasso: float=0,
        ridge: float=0,
        reweight: float=0,
        gradient: bool=False,
        verbose: bool=False) -> torch.Tensor:
    '''
    Args:
        x (torch.Tensor): (B x C x H x W or C x H x W) A high-resolution image.
        y (torch.Tensor): (B x C x H x W or C x H x W) A low-resolution image.
        scale (int): Downsampling scale.
        k (int): Kernel size.
        max_patches (int, optional): Maximum number of patches to use.
            If not specified, use all possible patches.
            You will get a better result with more patches.

        threshold (float, optional): Ignore values smaller than the threshold.

    Return:
        torch.Tensor: (k x k) The calculated kernel.
    '''
    # For even scales, the kernel size should be even, too (but not smaller one)
    if scale % 2 == 0 and k % 2 != 0:
        k += 1
    # The same holds for odd scales
    elif scale % 2 != 0 and k % 2 == 0:
        k += 1

    if x.dim() == 3:
        x = x.unsqueeze(0)

    if y.dim() == 3:
        y = y.unsqueeze(0)

    bx, cx, hx, wx = x.size()
    by, cy, hy, wy = y.size()

    # If y is larger than x
    if hx < hy:
        return find_kernel(y, x)

    # We convert RGB images to grayscale
    def luminance(rgb):
        coeff = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        l = torch.sum(coeff * rgb, dim=1, keepdim=True)
        return l

    if cx == 3:
        x = luminance(x)
    if cy == 3:
        y = luminance(y)

    # Some pixels in HR and LR images should be omitted
    # to compute an accurate kernel.
    # This offset is very important!
    offset_l = math.ceil(0.5 * (k / scale - 1))
    offset_h = offset_l * scale + (scale - k) // 2

    hx_new = hx - offset_h
    wx_new = wx - offset_h
    hx_new -= (hx_new - k) % scale
    wx_new -= (wx_new - k) % scale
    x = x[..., offset_h:(offset_h + hx_new), offset_h:(offset_h + wx_new)]

    hy_new = (hx_new - k) // scale + 1
    wy_new = (wx_new - k) // scale + 1
    y = y[..., offset_l:(offset_l + hy_new), offset_l:(offset_l + wy_new)]

    if verbose:
        print('Unfolding patches...')

    # Flatten
    y = y.reshape(by, -1)
    x = F.unfold(x, k, stride=scale)
    x_spatial = x.view(bx, k, k, -1)

    if max_patches == -1:
        x_sampled = x[0]
        y_sampled = y[0]
    else:
        '''
        Gradient-based sampling
        Caculate the gradient to determine which patches to use
        '''
        if verbose:
            print('Gradient-based sampling...')

        k_half = k // 2
        gx = x.new_zeros(1, k, k, 1)
        gx[:, k_half - 1, k_half - 1, :] = -1
        gx[:, k_half - 1, k_half, :] = 1
        grad_x = x_spatial * gx
        grad_x = grad_x.view(bx, k * k, -1)

        gy = x.new_zeros(1, k, k, 1)
        gy[:, k_half - 1, k_half - 1, :] = -1
        gy[:, k_half, k_half - 1, :] = 1
        grad_y = x_spatial * gy
        grad_y = grad_y.view(by, k * k, -1)

        grad = grad_x.sum(dim=1).pow(2) + grad_y.sum(dim=1).pow(2)
        grad_order = grad.view(-1).argsort(dim=-1, descending=True)

        # We need at least k^2 patches
        max_patches = max(k**2, max_patches)
        grad_order = grad_order[:max_patches].view(-1)
        # We use only one sample in the given batch
        x_sampled = x[0, ..., grad_order]
        y_sampled = y[0, ..., grad_order]

    '''
    Increase precision for numerical accuracy.
    You will get wrong results with FLOAT32!!!
    '''
    x_sampled = x_sampled.double()
    y_sampled = y_sampled.double()
    x_t = x_sampled.t()
    y = y_sampled.unsqueeze(0)

    if verbose:
        print('Solving equation...')

    if lasso > 0:
        solver = linear_model.Lasso(alpha=(lasso / y.nelement()))
        '''
        solver = linear_model.ElasticNet(
            alpha=(100 / y.nelement()),
            l1_ratio=0.75,
        )
        '''
        solver.fit(x_t.cpu().numpy(), y.squeeze(0).cpu().numpy())
        kernel = x.new_tensor(solver.coef_)
    elif ridge > 0:
        solver = linear_model.Ridge(alpha=2 * ridge)
        solver.fit(x_t.cpu().numpy(), y.squeeze(0).cpu().numpy())
        kernel = x.new_tensor(solver.coef_)
    elif gradient:
        x = x_sampled.float()
        y = y.float()
        with torch.enable_grad():
            kernel_params = nn.Parameter(
                x.new_zeros(1, k**2).normal_(),
                requires_grad=True,
            )
            optimizer = optim.Adam((kernel_params,), lr=1e-3)
            '''
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, [5, 20, 40, 60, 80], gamma=0.1,
            )
            '''
            def calc_loss():
                mm = torch.mm(kernel_params, x)
                loss = F.mse_loss(y, mm)
                return loss

            for i in tqdm.trange(100, ncols=80):
                loss = calc_loss()
                loss.backward()
                if (i + 1) % 1 == 0:
                    print('Step {:0>3}\tloss: {:.4f}'.format(i + 1, loss))

                optimizer.step()
                #scheduler.step()

        kernel = kernel_params.data
    else:
        solver = linear_model.LinearRegression()
        solver.fit(x_t.cpu().numpy(), y.squeeze(0).cpu().numpy())
        kernel = x.new_tensor(solver.coef_)
        '''
        kernel = torch.matmul(y, x_t)
        kernel_c = torch.matmul(x_sampled, x_t)
        kernel_c = torch.inverse(kernel_c)
        kernel = torch.matmul(kernel, kernel_c)
        '''
    # For debugging
    #from scipy import io
    #io.savemat('tensor.mat', {'x_t': x_t.numpy(), 'y': y.numpy(), 'kernel': kernel.numpy()})

    # Kernel thresholding and normalization
    kernel = kernel * (kernel.abs() > threshold).float()
    if reweight > 0:
        if k % 2 == 0:
            r = torch.linspace(-k // 2 + 0.5, k // 2 - 0.5, k)
        else:
            r = torch.linspace(-k // 2, k // 2, k)

        r = r.view(1, -1)
        r = r.repeat(k, 1)
        r = r**2
        # Squared distance from origin
        r = r + r.t()
        weight = torch.exp(-reweight * r)
        weight = weight.view(1, -1)
        kernel = kernel * weight

    kernel = kernel / kernel.sum()
    kernel = kernel.view(k, k).float()
    return kernel

def visualize_kernel(k: torch.Tensor) -> Image:
    with torch.no_grad():
        kh, kw = k.size()
        normalized = k / k.abs().max()
        r = -normalized * (normalized < 0).float()
        g = normalized * (normalized > 0).float()
        rgb = torch.stack([r, g, torch.zeros_like(normalized)], dim=0)

    pil = TF.to_pil_image(rgb.cpu())
    pil = pil.resize((20 * kw, 20 * kh), resample=Image.NEAREST)
    return pil


if __name__ == '__main__':
    import numpy as np
    import imageio
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=16, linewidth=200)
    a = imageio.imread('../example/butterfly.png')
    a = np.transpose(a, (2, 0, 1))
    a = torch.from_numpy(a).unsqueeze(0).float()
    #b = gaussian_filtering(a, sigma=0.3)
    k = gaussian_kernel(sigma=0.4)
    b = downsampling(a, k, 2)
    b = b.round().clamp(min=0, max=255).byte()
    b = b.squeeze(0)
    b = b.numpy()
    b = np.transpose(b, (1, 2, 0))
    imageio.imwrite('gaussian_04.png', b)
    input()
    #x = torch.arange(64).view(1, 1, 8, 8).float()
    #y = torch.arange(16).view(1, 1, 4, 4).float()
    #x = Image.open('../../dataset/DIV2K/DIV2K_train_HR/0001.png')
    x = Image.open('butterfly.png')
    x = TF.to_tensor(x)
    #y = Image.open('../../dataset/DIV2K/DIV2K_train_LR_d4/X2/0001.png')
    y = Image.open('bicubic.png')
    y = TF.to_tensor(y)
    scale = 2
    k = 16
    kernel = find_kernel(x, y, scale, k, max_patches=2**18, verbose=True)

    y_recon = downsampling(x, kernel, scale=scale)
    y_recon = (255 * y_recon).round().clamp(min=0, max=255)
    y_recon = y_recon / 255
    pil_y = TF.to_pil_image(y_recon.squeeze(0))
    pil_y.save('y_recon.png')

    '''
    kernel_double = kernel.double()
    u, s, v = kernel.svd()
    print(s)
    r = 2
    kernel_low = torch.matmul(torch.matmul(u[:, :r], torch.diag_embed(s[:r])), v[:, :r].t())
    kernel_low /= kernel_low.abs().max()
    k_pos = kernel_low * (kernel_low > 0).float()
    k_neg = kernel_low * (kernel_low < 0).float()
    k_rgb = torch.stack([-k_neg, k_pos, torch.zeros_like(k_pos)], dim=0)
    pil = TF.to_pil_image(k_rgb.cpu())
    pil = pil.resize((k * 20, k * 20), resample=Image.NEAREST)
    pil.save('kernel_low.png')
    '''
    pil = visualize_kernel(kernel)
    pil.save('kernel.png')
    #pil.show()
    #gau = gaussian_kernel(sigma=1.2, kernel_size=20)
    #pil = visualize_kernel(gau)
    #pil.save('gaussian.png')
