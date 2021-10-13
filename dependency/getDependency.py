from dependency.LinkedListConstructor import LinkedList, LinkedListPrinter, DependencySubList, DependencyRunTime



def LayerDetection(string):
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

def getDependency(model,OrderedList,convIdx):
    counter = 0
    ModelDict = {}
    for name, param in model.named_parameters():
        Type = LayerDetection(name)
        if(counter in OrderedList.keys()):
            # print(counter, Type)
            ModelDict[counter] = [Type, list(param.shape)]
        elif(counter == convIdx[-1]):
            ModelDict[counter] = [Type, list(param.shape)]
        counter += 1
    
  
    FinalLayer = convIdx[-1]
    
    LL = LinkedList(OrderedList, ModelDict, model, FinalLayer)
    
    # LinkedListPrinter(LL)

    DL = DependencyRunTime(LL)
    
    return DL