from torchvision import models

def make_model(cfg):
    kwargs = {'num_classes': cfg.n_classes}
    try:
        model_class = getattr(models, 'resnet{}'.format(cfg.depth))
    except:
        return models.resnet18(**kwargs)

    model_pt = model_class(pretrained=True)
    model_pt_state = model_pt.state_dict()
    model_pt_state.pop('fc.weight')
    model_pt_state.pop('fc.bias')
    model_target = model_class(**kwargs)
    model_target.load_state_dict(model_pt_state, strict=False)
    return model_target


