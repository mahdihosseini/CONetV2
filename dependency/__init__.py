from dependency.getDependency import getDependency
from dependency.LLADJ import ResNetAdj, DartsAdj, SQNetAdj, ResNextAdj, MobileNetAdj, SQNetAdjforHandCraft,MobileNetAdjForLL, ShuffleADJ

options = { 'resnet18': ResNetAdj,
            'resnet34': ResNetAdj,
            'mobilenetv2': MobileNetAdj,
            'squeezeNet': SQNetAdjforHandCraft,
            'shufflenetv2': ShuffleADJ,
            'darts': DartsAdj

}
    
def DependencyList(model_type, model):
    OrderedList, convIdx = options[model_type](model)
    return getDependency(model,OrderedList,convIdx)