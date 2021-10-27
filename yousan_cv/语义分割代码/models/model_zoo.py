from .fcn import *
from .deeplabv3 import *
from .deeplabv3_plus import *
from .pspnet import *
__all__ = ['get_model', 'get_model_list', 'get_segmentation_model']

_models = {
    'fcn32s_vgg16_voc': get_fcn32s_vgg16_voc,
    'fcn16s_vgg16_voc': get_fcn16s_vgg16_voc,
    'fcn8s_vgg16_voc': get_fcn8s_vgg16_voc,
    'deeplabv3_resnet50_voc': get_deeplabv3_resnet50_voc,
    'deeplabv3_resnet101_voc': get_deeplabv3_resnet101_voc,
    'deeplabv3_resnet152_voc': get_deeplabv3_resnet152_voc,
    'deeplabv3_plus_xception_voc': get_deeplabv3_plus_xception_voc,
    'psp_resnet50_voc': get_psp_resnet50_voc,
    'psp_resnet101_voc': get_psp_resnet101_voc,
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    return _models.keys()


def get_segmentation_model(model, **kwargs):
    models = {
        'fcn32s': get_fcn32s,
        'fcn16s': get_fcn16s,
        'fcn8s': get_fcn8s,
        'deeplabv3': get_deeplabv3,
        'deeplabv3_plus': get_deeplabv3_plus,
        'pspnet':get_psp,
    }
    return models[model](**kwargs)