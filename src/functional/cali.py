import cv2
import torch
import numpy as np

from srwarp import transform
from srwarp import grid
from srwarp import warp
from srwarp import functional

from model.sr import rrdb
from model.srwarp import baseline
from misc import image_utils

@torch.no_grad()
def main() -> None:
    net = baseline.SuperWarpF(
        max_scale=4,
        #backbone='mrdb',
        backbone='mdsr',
        residual=True,
        #residual=False,
        kernel_size_up=3,
        kernel_net=True,
        kernel_net_multi=True,
        kernel_depthwise=True,
        #kernel_depthwise=False,

        fill=-255,
    )
    net.cuda()
    #state = torch.load('../experiment/srwarp/fixed_mrdb_pt/latest.ckpt')
    #state = torch.load('../experiment/dualwarp/dual_reg/latest.ckpt')
    state = torch.load('../experiment/srwarp/fixed_mdsr_pt/latest.ckpt')
    state = state['model']
    net.load_state_dict(state)
    #x = image_utils.get_img('example/butterfly.png')
    x = image_utils.get_img('../experiment/srwarp/input_3.png')
    x = x.cuda()

    m = np.load('../experiment/srwarp/parameters.npz')
    mtx = m['mtx']
    dist = m['dist'][0]
    #f = functional.scaling(functional.barrel(hp=256, wp=256), scale=2)
    #f = functional.scaling(functional.spiral(hp=256, wp=256, k=4), scale=1)
    f = functional.calibration(
        mtx,
        dist[0],
        dist[1],
        dist[2],
        dist[3],
        k3=dist[4],
        offset_x=x.size(-1) // 2,
        offset_y=x.size(-2) // 2,
    )
    sizes = (2 * x.size(-2), 2 * x.size(-1))
    sizes_source = (x.size(-2), x.size(-1))
    box_x = 483
    box_y = 326
    box_width = 80
    box_height = 80
    scale = 1
    crop = False

    # net_sr = rrdb.RRDB(scale=4)
    # net_sr.cuda()
    # state = torch.load('../../../.cache/torch/hub/checkpoints/rrdb_x4_new-9d40f7f7.pth')
    # state = state['model']
    # net_sr.load_state_dict(state)
    # sr = net_sr(x)
    grid_raw, yi = grid.get_safe_functional_grid(f, sizes, (4 * sizes_source[0], 4 * sizes_source[1]), scale=0.25)
    # j = transform.jacobian(f, sizes, yi)
    # j = None
    # srcv = warp.warp_by_grid(sr, grid_raw, yi, sizes=sizes, fill=-255, j=j)
    # if crop:
    #     srcv = srcv[..., box_y:(box_y + box_height), box_x:(box_x + box_width)]

    # image_utils.save_img(srcv, 'example/cali_srcv.png', scale=scale)

    y, _ = net(x, f, sizes=sizes)
    if crop:
        y = y[..., box_y:(box_y + box_height), box_x:(box_x + box_width)]

    image_utils.save_img(y, 'example/cali_srwarp.png', scale=scale)

    grid_raw, yi = grid.get_safe_functional_grid(f, sizes, sizes_source)
    cv = warp.warp_by_grid(x, grid_raw, yi, sizes=sizes, fill=-255)
    if crop:
        cv = cv[..., box_y:(box_y + box_height), box_x:(box_x + box_width)]

    image_utils.save_img(cv, 'example/cali_cv2.png', scale=scale)

    '''
    box = grid.draw_boundary(x, grid_raw, yi, sizes, box_x, box_y, box_width, box_height, 10)
    image_utils.save_img(box, 'example/cali_box.png')
    '''
    return

if __name__ == '__main__':
    main()

