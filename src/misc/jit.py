import os
from os import path
import sys
import glob
import shutil

from data import common
from model.utils import forward_utils
import misc

import torch
from torch import jit
import imageio
import tqdm


def autocomplete(name):
    if 'exp-' in name:
        name = name.replace('exp-', '')
        name = path.join('..', 'experiment', name, 'model', 'latest')
    if path.basename(name) is name:
        name = path.join('..', 'models', name)
    if not path.splitext(name)[1]:
        name = name + '.pt'

    return name


def execute_jit(
        script, path_input, path_output,
        rep=1, mod=None, x8=False, exts='.png'):

    if not isinstance(exts, (list, tuple)):
        exts = (exts,)

    pre = common.Preprocessing()
    saver = misc.SRMisc()
    scan_exts = ['.png', 'jpg', 'jpeg', 'bmp']
    scan_exts.extend([scan_ext.upper() for scan_ext in scan_exts])
    scan = lambda x: sorted(glob.glob(path.join(path_input, '*' + x)))
    imgs = sorted(sum([scan(scan_ext) for scan_ext in scan_exts], []))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        print('Loading a JIT script from', script)
        script = jit.load(script, map_location=device)
    except Exception as e:
        print('Please load a JIT script:', e)
        sys.exit(1)

    #slicer = forward_utils.Slicer(model=script, max_size=80000)

    script.eval()
    with torch.no_grad():
        tq = tqdm.tqdm(imgs, ncols=80)
        for img in tq:
            basename = path.basename(img)
            tq.set_description('{: <15}'.format(basename))
            name = path.splitext(basename)[0]

            is_exist = True
            for ext in exts:
                if not path.isfile(path.join(path_output, name + ext)):
                    is_exist = False
                    break

            if is_exist:
                continue

            x = imageio.imread(img)
            scale = 2**rep
            if mod is not None:
                scale = max(scale, mod)

            h, w, _ = x.shape
            x = x[:h - h % scale, :w - w % scale]
            x = pre.set_color(x=x)
            x = pre.np2Tensor(**x)
            x = x['x'].unsqueeze(0).to(device)
            if x8:
                x = forward_utils.forward_x8(script, x, rep=rep)
            else:
                for _ in range(rep):
                    x = script(x)
            #y = slicer(x)
            if isinstance(x, (list, tuple)):
                x = x[0]

            saver.save(x, path_output, name, exts=exts)

        saver.end_background()
        saver.join_background()

