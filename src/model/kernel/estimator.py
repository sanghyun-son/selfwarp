import typing

import torch
from torch import nn
from torch.nn import init

from srwarp import utils

_II = typing.Tuple[int, int]


class Kernel(nn.Module):

    def __init__(self, kernel_size_max: int=20, n_feats: int=128) -> None:
        super().__init__()
        self.__kernel_size_max = kernel_size_max
        m = [
            nn.Linear(2, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, n_feats),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats, 1),
        ]
        self.net = nn.Sequential(*m)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()

        return

    @property
    def kernel_size_max(self) -> int:
        return self.__kernel_size_max

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size_max // 2
        x = utils.padding(x, -2, pad, pad, padding_type='reflect')
        x = utils.padding(x, -1, pad, pad, padding_type='reflect')
        return x

    def get_continuous(
            self,
            size: typing.Union[int, str]=256,
            rgb: bool=False) -> torch.Tensor:

        if size == 'exact':
            size = self.kernel_size_max
            is_exact = True
        else:
            is_exact = False

        with torch.no_grad():
            r = torch.arange(size**2)
            r = r.float()
            r = r.cuda()

            rh = r // size
            rw = r % size
            p = torch.stack((rh, rw), dim=1)
            offset = size // 2 - 0.5
            if is_exact:
                den = self.kernel_size_max // 2
            else:
                den = offset

            p = (p - offset) / den
            p.unsqueeze_(1)

        k = self.net(p)
        k = k.view(size, size)
        k = k / k.sum()

        if not rgb:
            return k

        with torch.no_grad():
            k_pos = k * (k >= 0).float()
            k_neg = -k * (k < 0).float()
            k_zero = k * 0
            k_rgb = torch.stack((k_neg, k_pos, k_zero), dim=0)
            k_rgb /= k_rgb.abs().max()

            k_rgb *= 255
            k_rgb.round_()
            k_rgb.clamp_(min=0, max=255)
            k_rgb = k_rgb.byte()
            k_rgb = k_rgb.cpu()

        return k_rgb

    def get_com(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        k_cont = self.get_continuous(size='exact')

        r = torch.arange(self.kernel_size_max**2)
        r = r.view(self.kernel_size_max, self.kernel_size_max)
        r = r.float()
        r = r.to(k_cont.device)

        offset = self.kernel_size_max // 2 - 0.5
        rh = r // self.kernel_size_max - offset
        rw = r % self.kernel_size_max - offset

        cy = rh * k_cont
        cy = cy.sum()

        cx = rw * k_cont
        cx = cx.sum()

        return cy, cx

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        '''
        grid: (N, k^2, 2)
        '''
        n, k, _ = grid.size()
        grid = grid.view(-1, 2)
        with torch.no_grad():
            side = self.kernel_size_max // 2
            grid = grid / side

        weight = self.net(grid)
        weight = weight.view(n, k)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        return weight        

    @torch.no_grad()
    def get_sampling_grid(
            self,
            sizes_input: _II,
            sizes_output: _II,
            device: typing.Optional[torch.device]=None) -> torch.Tensor:

        '''
        Return:
            torch.Tensor: (N, 2)
        '''
        hi, wi = sizes_input
        ho, wo = sizes_output

        grid_h = torch.arange(ho)
        grid_h = grid_h.float()
        if device is not None:
            grid_h = grid_h.to(device)

        grid_h = (grid_h + 0.5) * (hi / ho) - 0.5

        grid_w = torch.arange(wo)
        grid_w = grid_w.float()
        if device is not None:
            grid_w = grid_w.to(device)

        grid_w = (grid_w + 0.5) * (wi / wo) - 0.5

        grid_h = grid_h.view(-1, 1)
        grid_h = grid_h.repeat(1, wo)
        grid_h = grid_h.view(-1, 1)

        grid_w = grid_w.view(1, -1)
        grid_w = grid_w.repeat(ho, 1)
        grid_w = grid_w.view(-1, 1)

        grid = torch.cat((grid_h, grid_w), dim=1)
        return grid

    @torch.no_grad()
    def get_sampling_position(
            self,
            grid: torch.Tensor,
            kernel_size: typing.Optional[int]=None) -> torch.Tensor:

        if kernel_size is None:
            kernel_size = self.kernel_size_max

        if kernel_size % 2 == 0:
            comp = 1
        else:
            comp = 0.5

        grid_discrete = grid + comp
        grid_discrete.floor_()
        grid_discrete = grid_discrete.long()
        return grid_discrete

    @torch.no_grad()
    def get_kernel_coordinates(
            self,
            grid: torch.Tensor,
            kernel_size: typing.Optional[int]=None) -> torch.Tensor:

        if kernel_size is None:
            kernel_size = self.kernel_size_max

        grid_floor = grid.floor()
        grid_offset = grid - grid_floor
        grid_offset = grid_offset.view(-1, 1, 2)

        r = torch.arange(kernel_size**2)
        r = r.float()
        r = r.to(grid.device)

        window_h = r // kernel_size
        window_h = window_h.view(1, -1, 1)
        window_w = r % kernel_size
        window_w = window_w.view(1, -1, 1)
        local_coord = torch.cat((window_h, window_w), dim=-1)
        if kernel_size % 2 == 0:
            local_coord -= (kernel_size // 2 - 1)
        else:
            local_coord -= kernel_size // 2

        kernel_coordinates = local_coord - grid_offset
        return kernel_coordinates


if __name__ == '__main__':
    torch.set_printoptions(precision=3, linewidth=120, sci_mode=False)

    k = Kernel(8)
    k.cuda()
    grid = k.get_sampling_grid((4, 4), (4, 4), device=torch.device('cuda'))
    coords = k.get_kernel_coordinates(grid, 4)
    w = k(coords)
    #k_cont = k.get_continuous(size=8, rgb=True)
    #print(k_cont.size())

    k_raw = k.get_continuous(size='exact', rgb=False)
    print(k_raw)

    from torchvision import io
    io.write_png(k_cont, 'kernel.png')
