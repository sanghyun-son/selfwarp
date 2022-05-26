from optimizer import warm_multi_step_lr

def get_kwargs(cfg):
    return {
        'milestones': cfg.milestones,
        'milestones_d': cfg.milestones_d,
        'gamma': cfg.gamma,
        'linear': cfg.linear,
    }

def make_scheduler(opt, sub: bool=False, **kwargs):
    if sub:
        kwargs['milestones'] = kwargs['milestones_d']

    kwargs.pop('milestones_d')
    return warm_multi_step_lr.WarmMultiStepLR(opt, **kwargs)
