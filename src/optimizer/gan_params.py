import types

def set_params(cfg_ref):
    cfg = types.SimpleNamespace()
    attrs = (
        'optimizer',
        'lr',
        'milestones',
        'weight_decay',
        'epsilon',
        'gamma',
        'linear',
        'resume',
        'epochs',
    )
    for attr in attrs:
        attr_ref = getattr(cfg_ref, attr, None)
        setattr(cfg, attr, attr_ref)

    if cfg_ref.lr_d is not None:
        cfg.lr = cfg_ref.lr_d

    if cfg_ref.milestones_d is not None:
        cfg.milestones = cfg_ref.milestones_d

    cfg.beta1 = 0.5
    #cfg.beta1 = 0.9
    cfg.beta2 = 0.999
    cfg.grad_clip = 0
    return cfg

