import torchvision.models as pytorch_model
from models.resnet import (DASNet34,update_resnet_network)
from models.darts import (NetworkCIFAR,update_network_DARTS)
from models.SqueezeNet import (SqueezeNet,update_SqueezeNet_network)
from models.mobilenetv2 import MobileNetV2, update_mobilenet
from models.ShuffleNetV2 import shufflenetv2, update_shuffle_network

def get_model(model_type, **kwargs):
    models = {
        # 'resnet18': pytorch_model.resnet18,
        'resnet34': DASNet34,
        'darts': update_network_DARTS,
        'mobilenetv2': MobileNetV2,
        'shufflenetv2': shufflenetv2,
        # 'mobilenetv2': MobileNetV2,
        'squeezeNet': SqueezeNet 
    }
    return models[model_type](**kwargs)

def update_network(model_type, **kwargs):
    models = {
        # 'resnet18': pytorch_model.resnet18,
        'resnet34': update_resnet_network,
        'darts': update_network_DARTS,
        'squeezeNet': update_SqueezeNet_network,
        'mobilenetv2': update_mobilenet,
        'shufflenetv2': update_shuffle_network

    }
    return models[model_type](**kwargs)

