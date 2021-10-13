
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
#from torchviz import make_dot, make_dot_from_trace


#from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models

import torch.onnx
#from ptflops import get_model_complexity_info
import sys
sys.path.append("..")
# import global_vars as GLOBALS
class Fire(nn.Module):
    def __init__(self, in_planes, sq_intermediate_planes, ex_out_planes,kernel_size_1=1,kernel_size_2=3,stride=1):
        self.in_planes=in_planes
        self.intermediate_planes=sq_intermediate_planes
        self.out_planes=ex_out_planes

        super(Fire,self).__init__()
        self.conv1=nn.Conv2d(
                in_planes,
                sq_intermediate_planes,
                kernel_size=kernel_size_1,
                stride=stride,
                padding=int((kernel_size_1-1)/2),
                bias=False
        )
        self.bn1=nn.BatchNorm2d(sq_intermediate_planes)
        self.relu1=nn.ReLU()

        
        self.conv2=nn.Conv2d(
                sq_intermediate_planes,
                ex_out_planes,
                kernel_size=kernel_size_1,
                stride=1,
                padding=int((kernel_size_2-1)/2),
                bias=False
        )
        self.bn2=nn.BatchNorm2d(ex_out_planes)

        self.conv3=nn.Conv2d(
                sq_intermediate_planes,
                ex_out_planes,
                kernel_size=kernel_size_2,
                stride=1,
                padding=int((kernel_size_2-1)/2),
                bias=False
        )
        self.bn3=nn.BatchNorm2d(ex_out_planes)
        self.relu2=nn.ReLU()

    def forward(self,y):
        x = self.conv1(y)
        #print(x.shape,'post conv1 block')
        x = self.bn1(x)
        x = self.relu1(x)
        stream1 = self.conv2(x)
        stream1 = self.bn2(stream1)
        stream2 = self.conv3(x)
        stream2 = self.bn3(stream2)
        out = torch.cat([stream1, stream2], 1)
        out = self.relu2(out)
        return x



class SqueezeNet_Network(nn.Module):
    def __init__(self, num_classes_input=10,new_output_sizes=None,new_kernel_sizes=None):
        super(SqueezeNet_Network, self).__init__()
        
        # Could assign on basis of SuperBlock Identities
        #(In, Out/InNext, OutNext)
        if(new_output_sizes is not None):
            self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
            self.bn1 = nn.BatchNorm2d(96)
            self.relu = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
            self.fire2 = Fire(96, 16, 64) #(, kernelSize1, kernelSize2)
            self.fire3 = Fire(128, 16, 64)
            self.fire4 = Fire(128, 32, 128)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
            self.fire5 = Fire(256, 32, 128)
            self.fire6 = Fire(256, 48, 192)
            self.fire7 = Fire(384, 48, 192)
            self.fire8 = Fire(384, 64, 256)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
            self.fire9 = Fire(512, 64, 256)
            self.conv2 = nn.Conv2d(512, num_classes_input, kernel_size=1, stride=1)
            self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
            self.softmax = nn.LogSoftmax(dim=1)

        
        elif(new_kernel_sizes is not None):
            self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
            self.bn1 = nn.BatchNorm2d(96)
            self.relu = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
            self.fire2 = Fire(96, 16, 64) #(, kernelSize1, kernelSize2)
            self.fire3 = Fire(128, 16, 64)
            self.fire4 = Fire(128, 32, 128)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
            self.fire5 = Fire(256, 32, 128)
            self.fire6 = Fire(256, 48, 192)
            self.fire7 = Fire(384, 48, 192)
            self.fire8 = Fire(384, 64, 256)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
            self.fire9 = Fire(512, 64, 256)
            self.conv2 = nn.Conv2d(512, num_classes_input, kernel_size=1, stride=1)
            self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
            self.softmax = nn.LogSoftmax(dim=1)


        elif((new_output_sizes is not None) and (new_kernel_sizes is not None)):
            self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
            self.bn1 = nn.BatchNorm2d(96)
            self.relu = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
            self.fire2 = Fire(96, 16, 64) #(, kernelSize1, kernelSize2)
            self.fire3 = Fire(128, 16, 64)
            self.fire4 = Fire(128, 32, 128)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
            self.fire5 = Fire(256, 32, 128)
            self.fire6 = Fire(256, 48, 192)
            self.fire7 = Fire(384, 48, 192)
            self.fire8 = Fire(384, 64, 256)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
            self.fire9 = Fire(512, 64, 256)
            self.conv2 = nn.Conv2d(512, num_classes_input, kernel_size=1, stride=1)
            self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
            self.softmax = nn.LogSoftmax(dim=1)

        else:


            self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
            self.bn1 = nn.BatchNorm2d(96)
            self.relu = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
            self.fire2 = Fire(96, 16, 64) #(, kernelSize1, kernelSize2)
            self.fire3 = Fire(128, 16, 64)
            self.fire4 = Fire(128, 32, 128)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
            self.fire5 = Fire(256, 32, 128)
            self.fire6 = Fire(256, 48, 192)
            self.fire7 = Fire(384, 48, 192)
            self.fire8 = Fire(384, 64, 256)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
            self.fire9 = Fire(512, 64, 256)
            self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
            self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        out = self.softmax(x)

        return out

def SqueezeNet(num_classes_input = 10,new_output_sizes=None,new_kernel_sizes=None,global_config = None):
    # GLOBALS.BLOCK_TYPE='BasicBlock'
    # print('SETTING BLOCK_TYPE TO BasicBlock')
    return SqueezeNet_Network(num_classes_input,new_output_sizes,new_kernel_sizes)

def update_SqueezeNet_network(new_channel_sizes_list,new_kernel_sizes, global_config, class_num = None):

    # class_num = 0
    # if global_config.CONFIG['dataset'] == 'CIFAR10':
    #     class_num = 10
    # elif global_config.CONFIG['dataset'] == 'CIFAR100':
    #     class_num = 100
    if global_config.CONFIG['network']=='SqueezeNet':
        print(new_channel_sizes_list)
        
        new_channel_sizes = None
        # new_channel_sizes = [new_channel_sizes_list[0:7],
        #                     new_channel_sizes_list[7:9] + new_channel_sizes_list[10:16],
        #                     new_channel_sizes_list[16:18] + new_channel_sizes_list[19:29],
        #                     new_channel_sizes_list[29:31] + new_channel_sizes_list[32:]]
        new_network=SqueezeNet(num_classes_input=class_num,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
    return new_network

