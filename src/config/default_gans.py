def add_argument(group):
    group.add_argument('--dis', type=str, default='srgan.discriminator')
    group.add_argument('--dpatch', type=int, default=0)
    group.add_argument('--lr_d', type=float)
    group.add_argument('--milestones_d', nargs='+', type=int)
    group.add_argument('--n_z', type=int, default=100)
    group.add_argument('--gan_k', type=int, default=0)
    group.add_argument('--gp', type=float, default=10)

    group.add_argument('--dis_early_fork', type=int, default=0)
    group.add_argument('--no_sn', action='store_true')
    group.add_argument('--warmup', type=int, default=0)

    group.add_argument('--smoothing', action='store_true')
