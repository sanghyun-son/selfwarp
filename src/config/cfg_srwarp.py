from argparse import ArgumentParser
from email.policy import default

def add_argument(group: ArgumentParser):
    group.add_argument('--transform_type', type=str, default='fixed')
    group.add_argument('--no_adaptive_down', action='store_true')
    group.add_argument('--no_adaptive_up', action='store_true')
    group.add_argument('--kernel_size_up', type=int, default=3)
    group.add_argument('--kernel_bottleneck', type=int)

    group.add_argument('--depth_blending', type=int, default=6)
    group.add_argument('--depth_recon', type=int, default=10)

    group.add_argument('--adversarial', action='store_true')

    group.add_argument('--cv2_naive', action='store_true')
    group.add_argument('--cv2_interpolation', type=str, default='bicubic')

    group.add_argument('--scale_min', type=float, default=1.1)
    group.add_argument('--scale_max', type=float, default=4)

    group.add_argument('--reset_kernel', action='store_true')
    group.add_argument('--reset_sampler', action='store_true')

    # DualWarping
    group.add_argument('--patch_test', type=int, default=96)
    group.add_argument('--shuffle_updown', action='store_true')
    group.add_argument('--w_up', type=float, default=1)
    group.add_argument('--w_down', type=float, default=1)

    group.add_argument('--keep_prob', type=float, default=0)
    group.add_argument('--reg_prob', type=float, default=0.1)
    group.add_argument('--normalize_adl', action='store_true')
    group.add_argument('--gt_hat', action='store_true')
    group.add_argument('--grow', type=str, default='none')

    # SelfWarp
    group.add_argument('--m_path_eval', type=str)
    group.add_argument('--w_self', type=float, default=1)
    group.add_argument('--w_id', type=float, default=1)
    group.add_argument('--w_dual', type=float, default=1)
    group.add_argument('--pretrained_srwarp', type=str)
    group.add_argument('--resize_only', type=str, default=None)
    group.add_argument('--test_sr', action='store_true')
