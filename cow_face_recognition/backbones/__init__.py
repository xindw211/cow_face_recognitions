from .new_gam_iresnet import igam_iresnet100,igam_iresnet50

def get_model(name, **kwargs):
    # resnet
    if name == 'igam_ir100':
        return igam_iresnet100(False,**kwargs)
    else:
        raise ValueError()
