def add_argument(group):
    group.add_argument('--m_path', type=str)
    group.add_argument('--patch_max', type=int, default=128)

    group.add_argument('--min_scale', type=float, default=1)
    group.add_argument('--max_scale', type=float, default=4)
    group.add_argument('--ms', action='store_true')
    group.add_argument('--mss', action='store_true')
    group.add_argument('--no_position', action='store_true')
    group.add_argument('--log_scale', action='store_true')
    group.add_argument('--backbone', type=str, default='edsr')

    group.add_argument('--residual', action='store_true')
    group.add_argument('--normal_upsample', type=str, default='bicubic')
    group.add_argument('--kernel_size', type=int, default=4)
    group.add_argument('--elliptical', action='store_true')
    group.add_argument('--elliptical_upsample', action='store_true')

    group.add_argument('--ms_blend', type=str, default='net')

    group.add_argument('--identity_p', type=float, default=0)
    group.add_argument('--kernel_net', action='store_true')
    group.add_argument('--kernel_net_multi', action='store_true')
    group.add_argument('--kernel_net_size', type=int, default=4)
    group.add_argument('--kernel_net_regular', action='store_true')
    group.add_argument('--kernel_net_n_feats', type=int, default=48)
    group.add_argument('--kernel_noreg', action='store_true')
    group.add_argument('--kernel_depthwise', action='store_true')
    group.add_argument('--kernel_bilinear', action='store_true')
    group.add_argument('--kernel_no_mul', action='store_true')

    group.add_argument('--kernel_reset', action='store_true')
    group.add_argument('--sampler_reset', action='store_true')

    group.add_argument('--abl_awl', action='store_true')
    group.add_argument('--abl_multi', action='store_true')
    group.add_argument('--abl_recon', action='store_true')
