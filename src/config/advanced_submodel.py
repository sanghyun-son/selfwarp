def add_argument(group):
    # Masked adversarial loss configurations
    group.add_argument('--n_classes', type=int, default=11)
    group.add_argument('--ignore_bg', action='store_true')

    group.add_argument('--dsr', type=str)
    group.add_argument('--dump_intermediate', type=str)
    group.add_argument('--pretrained_sub', type=str)
    group.add_argument('--anchor', type=str)

    group.add_argument('--width_sub', type=int, default=64)
    group.add_argument('--depth_sub', type=int, default=8)
    group.add_argument('--mask_scale', type=int, default=16)

