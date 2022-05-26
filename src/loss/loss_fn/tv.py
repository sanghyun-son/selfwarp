import torch
import torch.nn as nn


def total_variation(x: torch.Tensor, reduce_var: bool=True) -> torch.Tensor:
    '''
    Total variation on 2D images.

    Args:
        x (torch.Tensor): B x C x H x W
        y (torch.Tensor or None): This value is ignored.
        kwargs (dict): A placeholder for dummy keyword arguments.
        reduce_var (bool, optional):

    Note:
        Please refere to an anisotropic version in
        https://en.wikipedia.org/wiki/Total_variation_denoising

        This implementation is also compatible with tf.image.total_variation
        when reduce_var is set to True.
    '''
    b, c, h, w = x.size()
    var_horz = torch.abs(x[..., 1:] - x[..., :-1])
    var_vert = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    # To avoid too large values
    if reduce_var:
        n_horz = c * h * (w - 1)
        n_vert = c * (h - 1) * w
    else:
        n_horz = 1
        n_vert = 1

    var = var_horz.sum().div(n_horz) + var_vert.sum().div(n_vert)
    # Batch-wise average
    var = var / b
    return var


class TV(nn.Module):

    def __init__(self, cfg: str=None):
        super(TV, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return total_variation(x)


if __name__ == '__main__':
    import imageio
    import numpy as np
    img = imageio.imread('~/dataset/DIV2K/DIV2K_train_HR/0001.png')
    img = img.astype(np.float32)
    img = img / 127.5 - 1

    import tensorflow as tf
    from tensorflow import image

    tf_tensor = tf.constant(img)
    tf_tv = image.total_variation(tf_tensor)
    with tf.Session() as sess:
        print(sess.run(tf_tv))

    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    pt_tensor = torch.from_numpy(img).unsqueeze(0)
    print(total_variation(pt_tensor))
