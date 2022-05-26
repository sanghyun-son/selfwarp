def add_argument(group):
    # Concat uniform random noise to the input
    group.add_argument('--stochastic', action='store_true')

    # Path to the unpaired dataset
    group.add_argument('--unpaired_hr', type=str)
    group.add_argument('--unpaired_lr', type=str)
    group.add_argument('--unpaired_split', action='store_true')

    # Statistics (.npy) files for the unpaired dataset
    group.add_argument('--stat_hr', type=str)
    group.add_argument('--stat_lr', type=str)

    group.add_argument('--no_repeat', action='store_true')

    # Apply Gaussian blur to the generated LR image
    group.add_argument('--blur_sigma', type=float, default=0)

    # For joint training
    group.add_argument('--exp_down', type=str)
    group.add_argument('--shave_down', type=int, default=4)

    group.add_argument('--kernel_gt', type=str)
    group.add_argument('--initial_sigma', type=float, default=1.2)
    group.add_argument('--adjust_weight', type=float, default=0.01)
    group.add_argument('--lasso', type=float, default=0)