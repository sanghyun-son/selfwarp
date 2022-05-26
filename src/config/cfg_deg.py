import argparse

def add_argument(group: argparse.ArgumentParser):
    group.add_argument('--no_srfeat', action='store_true')
    group.add_argument('--regression', action='store_true')
    group.add_argument('--normalize', action='store_true')
    group.add_argument('--synth_output', action='store_true')
    group.add_argument('--test_specific', action='store_true')

    group.add_argument('--gaussian_only', action='store_true')
    group.add_argument('--no_random_resize', action='store_true')
    group.add_argument('--specific_deg_train', type=int, default=-1)
    group.add_argument('--specific_deg_eval', type=int, default=-1)

    group.add_argument('--use_predefined', action='store_true')
    group.add_argument('--get_features', action='store_true')
    group.add_argument('--save_pred', action='store_true')
    group.add_argument('--test_lr', action='store_true')