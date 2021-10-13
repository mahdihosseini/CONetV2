import sys
sys.path.append('/home/helen/Documents/projects/AdaS-private/Channel_Search')
from dependency.LinkedListConstructor import LinkedList, LinkedListPrinter, DependencySubList, DependencyRunTime
from dependency.LLADJ import ResNetAdj

from RevisedVisualizer import PassiveVisualizer, ActiveVisualizer, PrimaryVisualizer
from torchvision import models
import pandas as pd
import os


model = models.resnet34()
OrderedList, convIdx = ResNetAdj(model)

LAYER_SET = ["Conv2d", "Squeeze-Conv2d","Expand3-Conv2d","Expand1-Conv2d"]
# MainStream
LAYER_4D_SET = {"conv":LAYER_SET[0],
                "downsample":LAYER_SET[0],  #ResNet Pop
                "squeeze":LAYER_SET[1],     #SqueezeNet Pop
                
                
                }

# Stream 1
S1_LAYER_4D_SET = {"expand3":LAYER_SET[2]      #SqueezeNet Pop
				}

# Stream 2
S2_LAYER_4D_SET = {"expand1":LAYER_SET[3]    #SqueezeNet Pop
				}

def LayerDetection(string):
    for elem in (LAYER_4D_SET.keys()):
        if elem in string:
            return LAYER_4D_SET[elem]
    for elem in (S1_LAYER_4D_SET.keys()):
        if elem in string:
            return S1_LAYER_4D_SET[elem]
    
    for elem in (S2_LAYER_4D_SET.keys()):
        if elem in string:
            return S2_LAYER_4D_SET[elem]

    return LAYER_SET[0]

def getTrialConvSizes(excel_path):
    data = pd.read_excel(excel_path)
    data_np = data.to_numpy()
    return data_np

   



if __name__ == "__main__":
    counter = 0
    ModelDict = {}
    ArchitectureName = "Resnet-34"
    for name, param in model.named_parameters():
        Type = LayerDetection(name)
        print(name, param.shape)
        if(counter in OrderedList.keys()):
            ModelDict[counter] = [Type, list(param.shape)]
        elif(counter == convIdx[-1]):
            ModelDict[counter] = [Type, list(param.shape)]
        counter += 1
    FinalLayer = convIdx[-1]
    LL = LinkedList(OrderedList, ModelDict, model, FinalLayer)
    LinkedListPrinter(LL)
    DL = DependencyRunTime(LL)


    

    # Practice for Running Trials:
    LastTrial = 24
    excel_path = '/home/helen/Documents/projects/AdaS-private/Channel_Search/adas_search/initEpoch=2_        epochpert=2_searching=greedy_        epoch_sel=avg_metric=MQC_        adaptnum=25_        gamma=0.8_1/Trials/adapted_architectures.xlsx'
    experiment_dir, _ = os.path.split(excel_path)
    conv_sizes_by_trial = getTrialConvSizes(excel_path)

    for i in range(LastTrial):
        # Get the convolution sizes
        conv_list = conv_sizes_by_trial[i]
        
        conv_list = conv_list[1:]
        
        for id,elem in zip(LL.keys(), LL.values()):
            try:
                # print(id)
                elem.size[0] = conv_list[int(id/3)]
                # elem.size[0] = conv_list[list(LL.keys()).index(id)]
            except:
                print(id)
                continue
       
        if (i==0):
            preLoadedValues = PrimaryVisualizer(LL, ModelDict, DL, ArchitectureName, "Tester", i,Folder_Path = os.path.join(experiment_dir, "ChannelEvolutionFolder"))

        else:

            ActiveVisualizer(*preLoadedValues, ModelDict, "Tester", i, LastTrial, LL)