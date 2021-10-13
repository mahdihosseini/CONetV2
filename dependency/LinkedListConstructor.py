
import networkx as nx
import matplotlib.pyplot as plt
import random
from torchvision import models
# import tensorflow as tf
import torch
import numpy as np

CLR_LAYER_TYPE = {"Conv2d":"red", "BatchNorm2d":"green", "Linear":"blue", "AdaptiveAvgPool2d":"red", "*":"green"}
INDEP_LIST = ["Sep-Conv"]

def LinkedList(OrderedList, ModelDict, Model, FinalLayer):

    class Flowpath:

        def __init__(self):
            self.start = None
            self.list = []

        def Insertion(self, node):
            if self.start is None:
                self.start = node
            self.list.append(node)
            return
        
        def PreInsertion(self, node, index):
            if self.start is None:
                self.start = node
                return
            n = 0
            for item in self:
                if (n == index): 
                    break
                pass
            item.next = node
    class Layer:
        def __init__(self, Name, nConnect, Size, index):
            self.type = Name
            self.size = Size
            self.prev = {}
            self.next = nConnect
            self.id = index
            self.inFlow = Name + "_" + str(index) + "_in"
            self.outFlow = Name + "_" + str(index) + "_out"
        

    class Connection:
        def __init__(self, Name, pLayer, index, nLayer):
            self.type = Name
            self.prev = pLayer
            self.next = nLayer
            self.id = index

    
    def Next(Dict, Key, Val):
        Dict[Key] = Val
        return Dict

    # List Conversion:
    DAG = Flowpath()
    LayerIndices4D = [] # Layer Numbers
    LayerInsertions = {} # List of Layers Dict
    # for elem in OrderedList:
    for idx, param in enumerate(Model.parameters()):
        NextConnections = {}
        if ((idx not in LayerIndices4D) and (idx in ModelDict.keys())):
           
            tempL = Layer(ModelDict[idx][0],None,ModelDict[idx][1],idx)
            LayerInsertions[idx] = tempL
            LayerIndices4D.append(idx)
        
        # Case for not all Layers Present:
        if ((idx in ModelDict.keys()) and (idx is not FinalLayer)):
            for e in OrderedList[idx]:
                
                if ((e not in LayerIndices4D)):
                    
                    tempA = Layer(ModelDict[e][0],None,ModelDict[e][1],e)
                    LayerInsertions[e] = tempA
                    LayerIndices4D.append(e)
                    # print(LayerInsertions[e].size)
                
                if(LayerInsertions[e].size == None):

                    tempC = Connection("Non-Conv", {idx:LayerInsertions[idx]}, e, {e:LayerInsertions[e]})
                
                elif((LayerInsertions[e].size)[1]== 1):
                    tempC = Connection("Sep-Conv", {idx:LayerInsertions[idx]}, e, {e:LayerInsertions[e]})
                
                elif((LayerInsertions[idx].size)[0] != (LayerInsertions[e].size)[1]):
                    tempC = Connection("Concat", {idx:LayerInsertions[idx]}, e, {e:LayerInsertions[e]})
                else:
                    #  Justify Conv based on the Size
                    tempC = Connection("Conv", {idx:LayerInsertions[idx]}, e, {e:LayerInsertions[e]})
                (LayerInsertions[e].prev)[idx] = tempC
                NextConnections = Next(NextConnections, e, tempC)
        
        if((idx in ModelDict.keys())): 
            # print("INDEX" + str(idx)) 
            LayerInsertions[idx].next = NextConnections
            DAG.Insertion(LayerInsertions[idx])

        for key in NextConnections:
            DAG.Insertion(NextConnections[key])
    return LayerInsertions
    
def LinkedListPrinter(LayerList): # LayerList = LayerInsertions

    temp = LayerList[0]
    Next = temp.next
    while(Next is not []):
        print(temp.type, temp.id)
        Prev = temp.prev
        Next = temp.next
        print(((Next).keys()))
        temp = list((Next).values())
        if(temp == []):
            break
        else:
            temp = list((Next).values())[0]

    # Channel Dependencies:
    
def DependencySubList(FirstElem, Visited, DependencyList, BackProp):
    subList = []
    # print("Dep-SubList-Start")
    for connects in (FirstElem.next).values():
        # elem Connections to Layer First
        # print("Forward-Propogate-Trigger")
        # print((connects).type)
        for elem in (connects.next).values():
            # idx Next Layers Per Connection:
            if(elem.inFlow in Visited):
                # print("BACK-Propogate-Trigger")
                BackProp = [True, elem]
            else:
                if(connects.type in INDEP_LIST):
                    Visited.append(elem.inFlow)
                    DependencyList.append([{elem.inFlow : elem}])
                    continue
                elif(len(elem.size)==2): #FC Layer
                    Visited.append(elem.inFlow)
                    continue


                Visited.append(elem.inFlow)
                subList.append({elem.inFlow : elem})
   
    subList.insert(0, {FirstElem.outFlow : FirstElem})
    Visited.append(FirstElem.outFlow)
    
    if(BackProp[0]):
        
        PrevConn = (BackProp[1].prev)
        for i in PrevConn.keys():
            if (FirstElem.id is not i):
                break
        PrevLay = list((PrevConn[i].prev).values())[0]
        Item = {PrevLay.outFlow : PrevLay}
        for block,ls in enumerate(DependencyList):
            # Block is Index of Inner List, ls is inner List of Dictionaries
            
            if Item in ls:
                for elem in subList:
                    
                    DependencyList[block].append(elem)
                # print(DependencyList[block])
                # print("Backward-Propogation-Complete")
                break
    else:
        if(len(subList) > 1):
            DependencyList.append(subList)
        # print(subList)
        # print("Forward-Propogation-Complete")

def DependencyRunTime(LayerList, DepRunTime=False): # LayerList = LayerInsertions
    V = []
    DependencyList = []
    temp = LayerList[0]
    Vals = list(LayerList.values())
    for id in sorted((LayerList.keys())):
        BackProp = [False, 0]
        # print(LayerList[id].type, LayerList[id].id, LayerList[id].next)
        if((LayerList[id].next == None) or (LayerList[id].next == {})): break
        DependencySubList(LayerList[id], V, DependencyList, BackProp)
    
    # Post Printing
    if(DepRunTime): return DependencyList
    else:
        DepList2 = []
        for el in DependencyList:
            if (len(el) != 1):
                DepList2.append(el)

        return DepList2
    
# def DependencyRunTime(LayerList): # LayerList = LayerInsertions
#     V = []
#     DependencyList = []
#     temp = LayerList[0]
#     Vals = list(LayerList.values())
#     for id in sorted((LayerList.keys())):
#         BackProp = [False, 0]
#         print(LayerList[id].type, LayerList[id].id)
#         if((LayerList[id].next == None) or (LayerList[id].next == {})): break
#         DependencySubList(LayerList[id], V, DependencyList, BackProp)
    
#     # Post Printing
    
#     for el in DependencyList:
#         print("DEP:")
#         print(el)
#     return DependencyList
    