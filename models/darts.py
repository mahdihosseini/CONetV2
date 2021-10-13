import torch
import torch.nn as nn
from models.operations import *
# from torch.autograd import Variable
from utils.utils import drop_path
import models.genotypes as genotypes


class Cell(nn.Module):

    def __init__(self, arch, genotype, C_prev_prev, C_prev, C_set, sep_conv_set, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if arch == "DARTS_PLUS_CIFAR100":
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C_set[0])
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C_set[0], 1, 1, 0)

            if reduction:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[1], 1, 1, 0)  #This is kinda redundant now
            else:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[1], 1, 1, 0)

            if reduction:
                op_names, indices = zip(*genotype.reduce)
                concat = genotype.reduce_concat
            else:
                op_names, indices = zip(*genotype.normal)
                concat = genotype.normal_concat
            self._compile_DARTS_PLUS(C_set, sep_conv_set, op_names, indices, concat, reduction)

        elif arch == "DARTS":
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C_set[0])
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C_set[0], 1, 1, 0)

            if reduction:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[0], 1, 1, 0)  # This is kinda redundant now
            else:
                self.preprocess1 = ReLUConvBN(C_prev, C_set[1], 1, 1, 0)

            if reduction:
                op_names, indices = zip(*genotype.reduce)
                concat = genotype.reduce_concat
            else:
                op_names, indices = zip(*genotype.normal)
                concat = genotype.normal_concat
            self._compile_DARTS(C_set, sep_conv_set, op_names, indices, concat, reduction)

    def _compile_DARTS(self, C_set, sep_conv_set, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        # This is the final feature map size after concatenation from all intermediate nodes
        # Please be cautious of this in the future
        if reduction:
            self.concat_size = 4 * C_set[0]  # Final concat size is 4x C0 channel size
        else:
            self.concat_size = C_set[0] + C_set[0] + C_set[2] + C_set[3]  # This need to be determined analytically! Draw the genotype on a piece of paper

        self._ops = nn.ModuleList()

        if reduction:
            for name, index in zip(op_names, indices):
                stride = 2 if index < 2 else 1
                # reduction cell only has 1 channel value
                op = OPS[name](C_set[0], C_set[0], stride, True)
                # print(name, index)
                self._ops += [op]
        else:

            node_index = 0
            sep_conv_set_index = 0
            edge_count = 0

            # These are the required output channel for each node
            node_to_input_size = [C_set[2], C_set[3], C_set[0], C_set[0]]

            for name, index in zip(op_names, indices):

                if "sep_conv" in name:
                    op = OPS[name](C_set[index], sep_conv_set[sep_conv_set_index], node_to_input_size[node_index], 1,
                                   True)
                    sep_conv_set_index = sep_conv_set_index + 1
                else:
                    op = OPS[name](C_set[index], node_to_input_size[node_index], 1, True)
                edge_count = edge_count + 1
                # Every node has 2 input edges
                if edge_count == 2:
                    node_index = node_index + 1
                    edge_count = 0
                # print(name, index)
                self._ops += [op]

        self._indices = indices

    def _compile_DARTS_PLUS(self, C_set, sep_conv_set, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        # This is the final feature map size after concatenation from all intermediate nodes
        # Please be cautious of this in the future
        if reduction:
            self.concat_size = 4 * C_set[0]  # Final concat size is 4x C0 channel size
        else:
            self.concat_size = C_set[0] + C_set[0] + C_set[2] + C_set[3]  # This need to be determined analytically! Draw the genotype on a piece of paper

        self._ops = nn.ModuleList()

        if reduction:
            for name, index in zip(op_names, indices):
                stride = 2 if index < 2 else 1
                # reduction cell only has 2 channel values now! update!!
                if index == 1 and name == 'skip_connect':
                    # This becomes a factorized reduce!
                    op = OPS[name](C_set[1], C_set[0], stride, True)
                else:
                    op = OPS[name](C_set[0], C_set[0], stride, True)
                # print(name, index)
                self._ops += [op]
        else:

            node_index = 0
            sep_conv_set_index = 0
            edge_count = 0

            # These are the required output channel for each node
            node_to_input_size = [C_set[0], C_set[0], C_set[2], C_set[3]]

            # These are the required input channel size for each edge
            edge_to_input_size = [C_set[0], C_set[1], C_set[0], C_set[0], C_set[1], C_set[0], C_set[0], C_set[2]]

            for name, index in zip(op_names, indices):

                if "sep_conv" in name:
                    op = OPS[name](edge_to_input_size[edge_count], sep_conv_set[sep_conv_set_index],
                                   node_to_input_size[node_index], 1, True)
                    sep_conv_set_index = sep_conv_set_index + 1
                else:
                    op = OPS[name](edge_to_input_size[edge_count], node_to_input_size[node_index], 1, True)
                edge_count = edge_count + 1
                # Every node has 2 input edges
                if edge_count % 2 == 0:
                    node_index = node_index + 1
                    # edge_count = 0
                # print(name, index)
                self._ops += [op]

        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C_list, sep_conv_list, num_classes, layers, auxiliary, genotype, arch):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        # stem_multiplier = 3
        C_curr = C_list[0][0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        C_prev_prev, C_prev = C_curr, C_curr
        self.cells = nn.ModuleList()
        reduction_prev = False

        # Only increment this for normal cells. Reduction cells do not have sep convs
        sep_conv_list_index = 0
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(arch, genotype, C_prev_prev, C_prev, C_list[i + 1], sep_conv_list[sep_conv_list_index], reduction,
                        reduction_prev)
            if not reduction:
                sep_conv_list_index = sep_conv_list_index + 1

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.concat_size
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

def DARTS(num_classes_input = 10,new_output_sizes=None,new_kernel_sizes=None):
    # GLOBALS.BLOCK_TYPE='BasicBlock'
    # print('SETTING BLOCK_TYPE TO BasicBlock')
    # (self, C_list, sep_conv_list, num_classes, layers, auxiliary, genotype, arch)
    return NetworkCIFAR(C_list, sep_conv_list, num_classes, layers, auxiliary, genotype, arch)
    #NetworkCIFAR(BasicBlock, 3, num_classes=num_classes_input, new_output_sizes=new_output_sizes,new_kernel_sizes=new_kernel_sizes)

def update_network_DARTS(num_classes_input = None, class_num = None, new_channel_sizes_list = None, new_kernel_sizes = None, new_cell_list = None,new_sep_conv_list = None,global_config = None, predefined = False):
    # new_channel_sizes_list,new_kernel_sizes, predefined = False
    
    assert global_config is not None, 'GLOBAL CONFIG CANNOT BE NONE'

    if global_config.CONFIG['network'] == 'DARTS':
        arch = "DARTS"
    elif global_config.CONFIG['network'] == 'DARTSPlus':
        arch = "DARTS_PLUS_CIFAR100"
    genotype = eval("genotypes.%s" % arch)
    
    if num_classes_input != None:

        fc_dim = num_classes_input
    elif class_num != None:
        fc_dim = class_num
    else:
        raise ValueError('Class number cannot be empty!!!!!')
    # if global_config.CONFIG["dataset"] == 'CIFAR10':
    #     fc_dim = 10
    # elif global_config.CONFIG["dataset"] == 'CIFAR100':
    #     fc_dim = 100

    assert global_config.CONFIG["num_cells"] == 7 or global_config.CONFIG["num_cells"] == 14 or global_config.CONFIG["num_cells"] == 20

    print('*'*200, new_channel_sizes_list)
    if new_channel_sizes_list == None:
        if global_config.CONFIG["num_cells"] == 20:
            if arch == "DARTS":
                new_cell_list = [[32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32],
             [32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32],
             [32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32]]
            else:
                new_cell_list = [[32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32, 32],
             [32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32, 32],
             [32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32],[32,32,32,32]]

        elif global_config.CONFIG["num_cells"] == 14:
            if arch == "DARTS":
                new_cell_list = [[32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32],
                      [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32],
                      [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32]]
            else:
                new_cell_list = [[32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32],
                      [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32],
                      [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32]]
        else:
            if arch == "DARTS":
                new_cell_list = [[32], [32, 32, 32, 32], [32, 32, 32, 32], [32], [32, 32, 32, 32], [32], [32, 32, 32, 32],
                 [32, 32, 32, 32]]
            else:
                new_cell_list = [[32], [32, 32, 32, 32], [32, 32, 32, 32], [32, 32], [32, 32, 32, 32], [32, 32], [32, 32, 32, 32],
                 [32, 32, 32, 32]]
    if new_channel_sizes_list == None:
        if global_config.CONFIG["num_cells"] == 20:
            if arch == "DARTS":
                new_sep_conv_list = [[32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32],
                 [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32],
                 [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32], [32,32,32,32,32]]
            else:
                new_sep_conv_list = [[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32], [32,32,32,32,32,32],
                 [32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32], [32,32,32,32,32,32],
                 [32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32], [32,32,32,32,32,32]]
        elif global_config.CONFIG["num_cells"] == 14:
            if arch == "DARTS":
                new_sep_conv_list = [[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],
                          [32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],
                          [32,32,32,32,32], [32,32,32,32,32]]
            else:
                new_sep_conv_list = [[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],
                          [32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],
                          [32,32,32,32,32,32], [32,32,32,32,32,32]]
        else:
            if arch == "DARTS":
                new_sep_conv_list = [[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32]]
            else:
                new_sep_conv_list = [[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32],[32,32,32,32,32,32]]

        
    if new_channel_sizes_list != None:
        print('HERRERER', new_channel_sizes_list)
        adj = new_channel_sizes_list
        new_cell_list = [[adj[0]], 

            [adj[1], adj[2], adj[6], adj[14]],

            [adj[25], adj[26], adj[30], adj[38]],

            [adj[49]],

            [adj[51], adj[53], adj[57], adj[65]],
            
            [adj[76]],

            [adj[78], adj[80], adj[84], adj[92]],

            [adj[103], adj[104], adj[108], adj[116]]]


        # [[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32]]
                    

        new_sep_conv_list = [[adj[4],adj[8],adj[12],adj[16],adj[20]],
                            [adj[28],adj[32],adj[36],adj[40],adj[44]],

                            [adj[55],adj[59],adj[63],adj[67],adj[71]],

                            [adj[82],adj[86],adj[90],adj[94],adj[96]],
                            [adj[106],adj[110],adj[114],adj[118],adj[122]]]

    if predefined:
        #Default Cell 7
        new_cell_list = [[48], [16, 16, 16, 16], [16, 16, 16, 16], [32], [32, 32, 32, 32], [64], [64, 64, 64, 64],
                        [64, 64, 64, 64]]

        #Default sep conv for cell 7
        new_sep_conv_list = [[16,16,16,16,16],[16,16,16,16,16],[32,32,32,32,32],[64,64,64,64,64],[64,64,64,64,64]]
        #2 start        
        # new_cell_list = [[32], [32, 32, 32, 32], [32, 32, 32, 32], [32], [32, 32, 32, 32], [32], [32, 32, 32, 32],
            #  [32, 32, 32, 32]]
        # new_sep_conv_list = [[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32],[32,32,32,32,32]]

    #The 10 is the number of classes in CIFAR10
    # C_list, sep_conv_list, num_classes, layers, auxiliary, genotype, arch
    new_network = NetworkCIFAR(C_list = new_cell_list, sep_conv_list = new_sep_conv_list, num_classes = fc_dim, layers = global_config.CONFIG["num_cells"], auxiliary = global_config.CONFIG['auxiliary'], genotype = genotype, arch = arch)
    # print (GLOBALS.NET)
    new_network.drop_path_prob = 0  # Need to update this
    return new_network
