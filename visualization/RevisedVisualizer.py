
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


import networkx as nx
import matplotlib.pyplot as plt
import random
import re
import cv2
import os

			
CLR_CONNECTOR_TYPE = {"Sep-Conv":[0,1,0], "Conv":[0,0,0], "Concat":[1,0,0], "Non-Conv":[1,0,1], "*":"green"}


def Vis_Init(LayerList, DependencyList):
    Graph = nx.DiGraph()
    LL = {}
    for key in sorted(LayerList):
        LL[key] = LayerList[key]
    
    
    PARAM_CLR = [1,1,1]
    MajorColorAssignment = {}
    for idx,elem in enumerate(DependencyList):
        PARAM_CLR[0] = random.uniform(0, 1)
        PARAM_CLR[1] = random.uniform(0.5, 1)
        PARAM_CLR[2] = random.uniform(0.5, 1)

        MajorColorAssignment[idx] = [[PARAM_CLR[0],PARAM_CLR[1],PARAM_CLR[2], 1],[PARAM_CLR[0],PARAM_CLR[1],PARAM_CLR[2], 0.2]]
    return Graph, LL, MajorColorAssignment


def NodeUpdateRoutine(node_size_map,layer,border_color_map):
    node_size_map.append(layer.size[0])

    if(layer.size[0] != layer.size[1]):
        border_color_map.append("r")
    else:
        border_color_map.append("black")
    return node_size_map,border_color_map

def NodeDescriptor(nodeList, node_size_map, border_color_map, node_color_map, DependencyList, MajorColorAssignment, layer, idx):
    nodeList.append(idx)

    node_size_map, border_color_map = NodeUpdateRoutine(node_size_map,layer,border_color_map)
    
    
    node_color_map[idx] = [1,1]
    
    for num, Dependency in enumerate(DependencyList):
        for loc,lay in enumerate(Dependency):
            
            Name, _ = list(lay.items())[0]
            Components = re.split('_+', Name)
            
            if (str(idx) == Components[1]): #Only 1 Key:
                
                if (Components[2] == 'in'):
                    
                    if(loc == 0):
                            node_color_map[idx][0] = MajorColorAssignment[num][0]   
                    else:
                        node_color_map[idx][0] = MajorColorAssignment[num][1]
                    
                
                else:
                    
                    if(loc == 0):
                            node_color_map[idx][1] = MajorColorAssignment[num][0]
                            
                            
                    else:
                        node_color_map[idx][1] = MajorColorAssignment[num][1]
                    
    return nodeList, node_size_map, border_color_map, node_color_map

def EdgeDescriptors(V, Next, TempDict, pos, edge_color_map, CurvedIndices, elem, idx, i, x):
    if((len(Next) > 1) and (i > list(TempDict.keys())[0])):
        pos[i] = (i + 2, x)
        x*=2
        CurvedIndices[idx] = i
    else:
        pos[i] = (i + 2, 0)

    edge_color_map.append(CLR_CONNECTOR_TYPE[elem.type])
    V.add_edge(idx, i)
    return V, TempDict, pos, edge_color_map, CurvedIndices, x


def PlotInit(V, pos, edge_color_map, node_color_map,idx, ArchitectureName):
        node_color_map[0][0] = node_color_map[0][1]
        node_color_map[idx][1] = node_color_map[idx][0]
        
        
        
        
        plt.rcParams["figure.figsize"] = [18, 8]
        plt.rcParams["figure.autolayout"] = True

        fig,ax = plt.subplots()
        ax.set_title(ArchitectureName)
        Edges = nx.draw_networkx_edges(V, pos = pos, edge_color = edge_color_map)
        return V, fig, ax, Edges, node_color_map

def NodePlotActive(V, ax, pos, node_color_map, node_size_map, border_color_map, ModelDict):
        for id,node in enumerate(V.nodes()):
            
            if((node in ModelDict.keys()) and (node != list(ModelDict)[-1])):
                w = ax.pie(
                    [50,50],
                    startangle=90,
                    center= pos[node],
                    colors=[node_color_map[node][0], node_color_map[node][1]],
                    radius=(node_size_map[id])/320,
                    wedgeprops={
                        'edgecolor':border_color_map[id]
                    }
                )
            else:
                x,y = pos[node]
                rect = plt.Rectangle((x,y-1), 2, 2, color='r', alpha=1)
                w = ax.add_patch(rect)


        return V, ax

def EdgePlotActive(V, ax, pos,CurvedIndices, Edges):
        EdgeIndices = []
        for id,e in enumerate(V.edges):
            a,b = e
            if ((a in CurvedIndices.keys())and (CurvedIndices[a] == b)):

                EdgeIndices.append(id)
            else:
                st, fs = pos[a]
                if((fs > 0)):
                    EdgeIndices.append(id)

        for i, l in enumerate(Edges):
            if (i in EdgeIndices):
                l.set_connectionstyle('arc3,rad=-0.6')
        return V, ax


def UpdateVisualizer(V, pos, edge_color_map,node_color_map, idx, ArchitectureName, CurvedIndices,ModelDict,node_size_map,border_color_map,LL =[],Update=False):
    if(Update):
        # Must Recalculate node_size_map & border_color_map
        print("UPDATING")
        for idx, layer in zip(LL.keys(),LL.values()):
            node_size_map, border_color_map = NodeUpdateRoutine(node_size_map,layer,border_color_map)

    
    V, fig, ax, Edges, node_color_map = PlotInit(V, pos, edge_color_map,node_color_map, idx, ArchitectureName) 

    
    V, ax = NodePlotActive(V, ax, pos, node_color_map, node_size_map, border_color_map, ModelDict)

    
    V, ax = EdgePlotActive(V, ax, pos,CurvedIndices, Edges)
    return V, fig, ax

def Visualizer(LayerList, ModelDict, DependencyList, ArchitectureName): #Add Node COlor Map Here
    
    V, LL, MajorColorAssignment = Vis_Init(LayerList, DependencyList)

    nodeList, edge_color_map, node_size_map, border_color_map = ([] for i in range(4))
    pos, CurvedIndices, node_color_map  = ({} for i in range(3))
    

    for idx, layer in zip(LL.keys(),LL.values()):
       
        ScaledHeight = 8
        Next = (layer.next)
        
        TempDict = {}
        for key in sorted(Next):
            TempDict[key] = Next[key]

        if (idx not in nodeList):
            nodeList, node_size_map, \
            border_color_map, node_color_map \
            = NodeDescriptor(nodeList,node_size_map, 
                             border_color_map, node_color_map, 
                             DependencyList,MajorColorAssignment, 
                             layer, idx)
            pos[idx] = (idx + 2, 0)

            
        for i, elem in zip(TempDict.keys(),TempDict.values()):
            
            if (i not in nodeList):
                nodeList, node_size_map, \
                border_color_map, node_color_map \
                = NodeDescriptor(nodeList, node_size_map, 
                                border_color_map, node_color_map, 
                                DependencyList, MajorColorAssignment, 
                                LL[i], i)


            V, TempDict, pos, edge_color_map, \
            CurvedIndices, ScaledHeight = EdgeDescriptors(V, Next, 
                                                          TempDict, pos, 
                                                          edge_color_map, 
                                                          CurvedIndices, 
                                                          elem, idx, i, ScaledHeight)


    
    V, fig, ax = UpdateVisualizer(V, pos, edge_color_map,node_color_map, idx, ArchitectureName,CurvedIndices,ModelDict,node_size_map,border_color_map,Update=False)

    return V,fig,ax, pos, edge_color_map,node_color_map, idx, ArchitectureName, CurvedIndices
    


def PassiveVisualizer(LayerList, ModelDict, DependencyList, ArchitectureName):

    _,fig,ax,pos, *_ = Visualizer(LayerList, ModelDict, DependencyList, ArchitectureName)
    Buffer = 10
    xStart, xFinish = (pos[list(pos)[0]][0]), (pos[list(pos)[-1]][0]+Buffer)
    ax.set_ylim(-25,25)
    ax.set_xlim(xStart, xFinish)
    print(xStart, xFinish)
    plt.savefig(ArchitectureName + "-ChannelEvolution.jpg", dpi=150)

    plt.show()

def PrimaryVisualizer(LayerList, ModelDict, DependencyList, ArchitectureName, NameIdentifier, Trial,Folder_Path = "ChannelEvolutionFolder" ):
    # Folder_Path = "ChannelEvolutionFolder"   
    if not os.path.exists(Folder_Path):
        os.mkdir(Folder_Path)
    

    V, fig, ax, pos, edge_color_map,node_color_map, idx, ArchitectureName,CurvedIndices = Visualizer(LayerList, ModelDict, DependencyList, ArchitectureName)
    Buffer = 10
    xStart, xFinish = (pos[list(pos)[0]][0]), (pos[list(pos)[-1]][0]+Buffer)
    ax.set_ylim(-25,25)
    ax.set_xlim(xStart, xFinish)
    print(xStart, xFinish)
    plt.savefig(Folder_Path+"/"+NameIdentifier+str(Trial)+"-ChannelEvolution.jpg", dpi=150)
    return V, pos, edge_color_map,node_color_map, idx, ArchitectureName, CurvedIndices, Folder_Path

def ActiveVisualizer(V, pos, edge_color_map,node_color_map, idx, ArchitectureName, CurvedIndices,Folder_Path, ModelDict, NameIdentifier, Trial, FinalTrial, LL):
    
    V,fig, ax = UpdateVisualizer(V, pos, edge_color_map,node_color_map, idx, ArchitectureName,CurvedIndices,ModelDict,[],[],LL=LL, Update=True)
    Buffer = 10
    xStart, xFinish = (pos[list(pos)[0]][0]), (pos[list(pos)[-1]][0]+Buffer)
    ax.set_ylim(-25,25)
    ax.set_xlim(xStart, xFinish)
    print(xStart, xFinish)
    plt.savefig(Folder_Path+"/"+NameIdentifier+str(Trial)+"-ChannelEvolution.jpg", dpi=150)
    if(Trial == FinalTrial-1):
        frames = []
        for i in range(FinalTrial):
            img = cv2.imread(Folder_Path+"/"+NameIdentifier+str(i)+"-ChannelEvolution.jpg")
            h,w,_ = img.shape
            size = (w,h)
            frames.append(img)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(Folder_Path+"/"+NameIdentifier+"ChannelEvolution.mp4", fourcc, 1, size)

        for i in range(FinalTrial):
            out.write(frames[i])
        out.release()

    return




