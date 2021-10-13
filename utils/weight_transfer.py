from pathlib import Path
import os
import platform
import time
import copy
import pandas as pd
import numpy as np
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from utils.VBMF import EVBMF
#DARTS Model files
# from model import NetworkCIFAR as DARTS
# import genotypes
#ResNet
# from models.own_network import DASNet34,DASNet50

# reshape the net's 4D param using mode_4 unfold: (shape[0],shape[1]*shape[2]*shape[3])
def mode_4_reshape(param):
    shape = param.shape
    mode_4_unfold = param.cpu()
    mode_4_unfold = torch.reshape(mode_4_unfold, [shape[0], shape[1] *
                                                  shape[2] * shape[3]])
    return shape, mode_4_unfold

# restore the unfolded new model's param back to 4D to be loaded into new_net
def param_shape_restore(param,orig_shape):
    param = torch.reshape(param, [orig_shape[0], orig_shape[1],
                                  orig_shape[2],orig_shape[3]])

    return param

    # """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # return param.to(device)
    # """
def svd_phi(param):
    U, S, V = torch.svd(param)
    H, M = param.shape
    U = U[:, :H]
    S = S[:H]
    V = V[:, :H]
    return U,torch.diag(S),V

# take mode_4 unfolded 2D param matrix and transfer param value from old to new
def param_weight_copy(old_param, new_param, wt_type):
    if wt_type == 'EVBMF':
        old_U, old_S, old_V = EVBMF(old_param) # results in all 0 (or mostly 0?)
        old_param = torch.matmul(old_U, torch.matmul(old_S, torch.t(old_V)))
    temp = old_param
    # check if one of the dim is shrinking
    if new_param.shape[0] < old_param.shape[0] or new_param.shape[1] < old_param.shape[1]:
        # get U,S,V of old_param from SVD
        if wt_type == 'SVD':
            old_U, old_S, old_V = svd_phi(old_param)
        # check if both dims are shrinking
        if new_param.shape[0] < old_param.shape[0] and new_param.shape[1] < old_param.shape[1]:
            # reduce U, S and V
            temp_U = old_U[:new_param.shape[0], :new_param.shape[0]]
            temp_S = old_S[:new_param.shape[0], :new_param.shape[1]]
            temp_V = old_V[:new_param.shape[1], :new_param.shape[1]]
        else:
            # check if only row is shrinking
            if new_param.shape[0] < old_param.shape[0]:
                # reduce U, S
                temp_U = old_U[:new_param.shape[0], :new_param.shape[0]]
                temp_S = old_S[:new_param.shape[0], :]
                temp_V = old_V
            # check if only col is shrinking
            elif new_param.shape[1] < old_param.shape[1]:
                # reduce S, V
                temp_U = old_U
                temp_S = old_S[:, :new_param.shape[1]]
                temp_V = old_V[:new_param.shape[1], :new_param.shape[1]]
        # phi = UxSxV^T (Approx of the unfolded new_param)
        temp = torch.matmul(temp_U, torch.matmul(temp_S, torch.t(temp_V)))

    # check if one of the dim is expanding
    if new_param.shape[0] > old_param.shape[0] or new_param.shape[1] > old_param.shape[1]:
        # check if both dim are expanding
        if new_param.shape[0] > old_param.shape[0] and new_param.shape[1] > old_param.shape[1]:
            # first duplicate the rows then the cols
            row_diff = new_param.shape[0] - old_param.shape[0]
            row_from = old_param.shape[0] - row_diff
            temp2 = torch.cat((temp, temp[row_from:, :]), dim=0)

            col_diff = new_param.shape[1] - old_param.shape[1]
            col_from = old_param.shape[1] - col_diff
            new_param = torch.cat((temp2, temp2[:, col_from:]), dim=1)
        else:
            # check if only row is expanding
            if new_param.shape[0] > old_param.shape[0]:
                # duplicate the row
                row_diff = new_param.shape[0] - old_param.shape[0]
                row_from = old_param.shape[0] - row_diff
                new_param = torch.cat((temp, temp[row_from:, :]), dim=0)
            # check if only col is expanding
            elif new_param.shape[1] > old_param.shape[1]:
                # duplicate the col
                col_diff = new_param.shape[1] - old_param.shape[1]
                col_from = old_param.shape[1] - col_diff
                new_param = torch.cat((temp, temp[:, col_from:]), dim=1)
    # either no expansion (shrinking only) or no change in dim at all
    else:
        new_param = temp
    return new_param

# transfer the param value of old_net to the new_net
def weight_transfer(old_net, new_net, wt_type, old_net_prefix = ""):
    with torch.no_grad():
        print(old_net.state_dict().keys())
        for idx, (name, new_param) in enumerate(new_net.named_parameters()):
            # check if layer is conv
            if len(new_param.shape) == 4:
                # new_param_t = copy.deepcopy(new_param)
                
                old_param = old_net.state_dict()[old_net_prefix + name]
                # reshape the param using mode4 unfolding (chan_out,chan_in*kernel*kernel)
                old_shape, old_param_reshape = mode_4_reshape(old_param)
                new_shape, new_param_reshape = mode_4_reshape(new_param)
                # copy old net's weights to new net
                new_param_updated = param_weight_copy(old_param_reshape, new_param_reshape, wt_type)
                # reshape the param back to 4D
                new_param_restored = param_shape_restore(new_param_updated, new_shape)
                # load the weight into new param
                new_param.copy_(new_param_restored)
                # del new_param_t

# use for debugging comparing old_net and new_net value of each layer
def check_param(old_net, new_net):
    check_layer = 0
    layer = 0
    with torch.no_grad():
        for idx, (name, new_param) in enumerate(new_net.named_parameters()):
            # check if layer is conv
            if len(new_param.shape) == 4:
                if layer == check_layer:
                    print("name: ",name, ", idx: ", idx, ", conv layer: ",layer)
                    print("old param val: \n",old_net.state_dict()[name])
                    print("new param val: \n",new_param)
                layer += 1

# use for debugging with different architectures
def practice_test(arch, wt_type):
    if arch == 'ResNet':
        super1_idx = [32, 32, 32, 32, 32, 32, 32]
        super2_idx = [32, 32, 32, 32, 32, 32, 32, 32]
        super3_idx = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
        super4_idx = [32, 32, 32, 32, 32, 32]
        old_conv_sizes = [super1_idx,super2_idx,super3_idx,super4_idx]
        old_network = DASNet34(num_classes_input=10, new_output_sizes=old_conv_sizes)

        super1_idx = [48, 48, 48, 48, 48, 48, 48]
        super2_idx = [28, 28, 28, 28, 28, 28, 28, 28]
        super3_idx = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
        super4_idx = [32, 32, 32, 32, 32, 32]

        new_conv_sizes = [super1_idx, super2_idx, super3_idx, super4_idx]
        new_network = DASNet34(num_classes_input=10, new_output_sizes=new_conv_sizes)
        weight_transfer(old_network, new_network, wt_type)
        check_param(old_network, new_network)


    elif arch == 'DARTS':
        darts_arch = "DARTS"
        genotype = eval("genotypes.%s" % darts_arch)
        DARTS_cell_list_7 = [[32], [32, 32, 32, 32], [32, 32, 32, 32], [32], [32, 32, 32, 32], [32], [32, 32, 32, 32],
                             [32, 32, 32, 32]]
        DARTS_sep_conv_list_7 = [[32, 32, 32, 32, 32], [32, 32, 32, 32, 32], [32, 32, 32, 32, 32], [32, 32, 32, 32, 32],
                                 [32, 32, 32, 32, 32]]
        old_network = DARTS(DARTS_cell_list_7, DARTS_sep_conv_list_7, 10, 7, False, genotype, arch)
        new_cell_list = [[26], [26, 26, 26, 26], [48, 32, 26, 40], [32], [22, 38, 32, 32], [32], [28, 40, 32, 30],
                             [32, 32, 32, 32]]
        new_sep_conv_list = [[32, 32, 32, 32, 32], [48, 48, 48, 48, 48], [20, 32, 48, 20, 30], [32, 26, 40, 26, 34],
                             [32, 32, 32, 32, 32]]
        new_network = DARTS(new_cell_list, new_sep_conv_list, 10, 7, False, genotype, arch)
        weight_transfer(old_network, new_network, wt_type)
        check_param(old_network, new_network)

# use for debugging the weight copying
def weight_copy_test():
    '''TEST WEIGHT COPY'''
    old_param = torch.randn(4,6)
    print("old_param:\n", old_param)
    new_param = torch.randn(4,6)
    print("new_param:\n", new_param)
    new_param = param_weight_copy(old_param, new_param, wt_type='EVBMF')
    print("new_param_updated:\n",new_param)

if __name__ == '__main__':
    practice_test(arch='ResNet', wt_type='EVBMF') # DARTS (7-cell only), ResNet
    # weight_copy_test()
