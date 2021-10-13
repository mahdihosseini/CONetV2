"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import List, Any
import numpy as np
import torch
from copy import deepcopy

from metrics.components import LayerType, IOMetrics
from utils.VBMF import EVBMF, np_EVBMF, np_EVBMF_s_only
# import  global_vars as GLOBALS


class Metrics():
    def __init__(self, parameters: List[Any], p: int, config = None) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        '''
        assert config is not None, 'Please provide a config to metrics'
        self.config = config
        self.net_blocks = net_blocks = parameters
        self.layers_index_todo = np.ones(shape=len(net_blocks), dtype='bool')
        self.layers_info = list()
        self.number_of_conv = 0
        self.number_of_fc = 0
        self.p = p
        self.historical_metrics = list()
        for iteration_block in range(len(net_blocks)):
            block_shape = net_blocks[iteration_block].shape
            if len(block_shape) == 4:
                self.layers_info = np.concatenate(
                    [self.layers_info, [LayerType.CONV]], axis=0)
                self.number_of_conv += 1
            elif len(block_shape) == 2:
                self.layers_info = np.concatenate(
                    [self.layers_info, [LayerType.FC]], axis=0)
                self.number_of_fc += 1
            else:
                self.layers_info = np.concatenate(
                    [self.layers_info, [LayerType.NON_CONV]], axis=0)
        self.final_decision_index = np.ones(
            shape=self.number_of_conv, dtype='bool')
        self.conv_indices = [i for i in range(len(self.layers_info)) if
                             self.layers_info[i] == LayerType.CONV]
        self.fc_indices = [i for i in range(len(self.layers_info)) if
                           self.layers_info[i] == LayerType.FC]
        self.non_conv_indices = [i for i in range(len(self.layers_info)) if
                                 self.layers_info[i] == LayerType.NON_CONV]
        '''
        new metric: temporal W, cumulative KG and Rank
        '''
        self.window_size = self.config['window_size'] # how many cols (iterations) to keep
        self.col_count = 0 # where to insert new col
        self.big_W = list()
        self.cumulative_S = np.zeros(len(self.conv_indices),dtype=float)
        self.cumulative_rank = np.zeros(len(self.conv_indices),dtype=float)
        self.cumulative_cond = np.zeros(len(self.conv_indices),dtype=float)
        '''Initialize big_W'''
        if self.config['sel_metric'] == 'MR-sw':
            for block_index in range(len(self.conv_indices)):
                layer_tensor = self.net_blocks[self.conv_indices[block_index]].data
                N = layer_tensor.shape[0] * layer_tensor.shape[1] * layer_tensor.shape[2] * layer_tensor.shape[3]
                layer_big_W = np.zeros(shape=[self.window_size, N], dtype=float)
                self.big_W.append(deepcopy(layer_big_W))

    def update_big_W(self) -> None:
        '''
        Inserts updated weight of each layer into the big_W matrix
        '''
        for block_index in range(len(self.conv_indices)):
            layer_tensor = self.net_blocks[self.conv_indices[block_index]].data
            N = layer_tensor.shape[0] * layer_tensor.shape[1] * layer_tensor.shape[2] * layer_tensor.shape[3]
            flattened = layer_tensor.cpu()
            flattened = torch.reshape(flattened, [1,N]).numpy()
            if self.col_count >= self.window_size:
                # remove 1st col and shuffle cols to the left by 1
                self.big_W[block_index] = np.concatenate((self.big_W[block_index][1:,:],flattened),axis=0)
            else:
                self.big_W[block_index][self.col_count,:] = deepcopy(flattened.squeeze())
        self.col_count += 1

    # if need to reinitialize
    def clear_big_W(self) -> None:
        self.col_count = 0
        self.big_W = list()
        self.cumulative_S = np.zeros(len(self.conv_indices),dtype=float)
        self.cumulative_rank = np.zeros(len(self.conv_indices),dtype=float)
        self.cumulative_cond = np.zeros(len(self.conv_indices),dtype=float)
        '''Initialize big_W'''
        for block_index in range(len(self.conv_indices)):
            layer_tensor = self.net_blocks[self.conv_indices[block_index]].data
            N = layer_tensor.shape[0] * layer_tensor.shape[1] * layer_tensor.shape[2] * layer_tensor.shape[3]
            layer_big_W = np.zeros(shape=[self.window_size, N], dtype=float)
            self.big_W.append(deepcopy(layer_big_W))

    def evaluate(self, epoch: int) -> IOMetrics:
        '''
        Computes the knowledge gain (S) and mapping condition (condition)
        '''
        input_channel_rank = list()
        output_channel_rank = list()
        mode_12_channel_rank = list()
        input_channel_S = list()
        output_channel_S = list()
        mode_12_channel_S = list()
        input_channel_condition = list()
        output_channel_condition = list()
        mode_12_channel_condition = list()
        input_channel_QC = list()
        output_channel_QC = list()
        input_channel_newQC = list()
        output_channel_newQC = list()

        ''' New metrics '''
        big_W_rank = list()
        big_W_S = list()
        big_W_condition = list()
        if self.config['sel_metric'] != 'MR-sw':
            big_W_rank = np.zeros(len(self.conv_indices), dtype=float)
            big_W_S = np.zeros(len(self.conv_indices), dtype=float)
            big_W_condition = np.zeros(len(self.conv_indices), dtype=float)

        factorized_index_12 = np.zeros(len(self.conv_indices), dtype=bool)
        factorized_index_3 = np.zeros(len(self.conv_indices), dtype=bool)
        factorized_index_4 = np.zeros(len(self.conv_indices), dtype=bool)
        factorized_index_bigW = np.zeros(len(self.conv_indices), dtype=bool)
        for block_index in range(len(self.conv_indices)):
            layer_tensor = self.net_blocks[self.conv_indices[block_index]].data

            tensor_size = layer_tensor.shape
            mode_12_unfold = layer_tensor.permute(3, 2, 1, 0).cpu()
            mode_12_unfold = torch.reshape(
                mode_12_unfold, [tensor_size[3] * tensor_size[2],
                                 tensor_size[1] * tensor_size[0]])

            mode_3_unfold = layer_tensor.permute(1, 0, 2, 3).cpu()
            mode_3_unfold = torch.reshape(
                mode_3_unfold, [tensor_size[1], tensor_size[0] *
                                tensor_size[2] * tensor_size[3]])
            mode_4_unfold = layer_tensor.cpu()
            mode_4_unfold = torch.reshape(
                mode_4_unfold, [tensor_size[0], tensor_size[1] *
                                tensor_size[2] * tensor_size[3]])

            try:
                #Size of input to EVBMF is (tensor_size[2]*tensor_size[3]) x (tensor_size[2]*tensor_size[3])
                U_approx, S_approx, V_approx = EVBMF(torch.matmul(mode_12_unfold, torch.t(mode_12_unfold)))
                mode_12_channel_rank.append(
                    S_approx.shape[0] / (tensor_size[2] * tensor_size[3]))
                low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
                low_rank_eigen = np.sqrt(low_rank_eigen)
                low_rank_eigen = low_rank_eigen ** self.p
                if len(low_rank_eigen) != 0:
                    mode_12_channel_condition.append(
                        low_rank_eigen[0] / low_rank_eigen[-1])
                    sum_low_rank_eigen = low_rank_eigen / \
                        max(low_rank_eigen)
                    sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
                else:
                    mode_12_channel_condition.append(0)
                    sum_low_rank_eigen = 0
                factorized_index_12[block_index] = True
                mode_12_channel_S.append(sum_low_rank_eigen / (tensor_size[2] * tensor_size[3]))
                # NOTE never used (below)
                # mode_12_unfold_approx = torch.matmul(
                #     U_approx, torch.matmul(S_approx, torch.t(V_approx)))
            except Exception:
                U_approx = torch.zeros(mode_12_unfold.shape[0], 0)
                S_approx = torch.zeros(0, 0)
                V_approx = torch.zeros(mode_12_unfold.shape[1], 0)
                if epoch > 0:
                    # mode_12_channel_rank.append(
                    #     variables_performance['mode_12_rank_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # mode_12_channel_S.append(
                    #     variables_performance['mode_12_S_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # mode_12_channel_condition.append(
                    #     variables_performance['mode_12_condition_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    mode_12_channel_rank.append(
                        self.historical_metrics[-1].mode_12_channel_rank[block_index])
                    mode_12_channel_S.append(
                        self.historical_metrics[-1].mode_12_channel_S[block_index])
                    mode_12_channel_condition.append(
                        self.historical_metrics[-1].mode_12_channel_condition[block_index])
                else:
                    mode_12_channel_rank.append(0)
                    mode_12_channel_S.append(0)
                    mode_12_channel_condition.append(0)

            try:
                U_approx, S_approx, V_approx = EVBMF(mode_3_unfold)
                input_channel_rank.append(
                    S_approx.shape[0] / tensor_size[1])
                low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
                low_rank_eigen = low_rank_eigen ** self.p
                if len(low_rank_eigen) != 0:
                    input_channel_condition.append(
                        low_rank_eigen[0] / low_rank_eigen[-1])
                    sum_low_rank_eigen = low_rank_eigen / \
                        max(low_rank_eigen)
                    sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
                else:
                    input_channel_condition.append(0)
                    sum_low_rank_eigen = 0
                factorized_index_3[block_index] = True
                input_channel_S.append(sum_low_rank_eigen / tensor_size[1])
                low_rank = input_channel_rank[-1]
                if input_channel_condition[-1] == 0:
                    MC_p = 1
                else:
                    MC_p = 1 - 1/input_channel_condition[-1]
                if MC_p == 0:
                    MC_p = 0.0001
                input_channel_QC.append(np.arctan2(low_rank, MC_p))
                # print('input qc shape', input_channel_QC)
                # print('low_rank_shape', low_rank)
                #input_channel_newQC.append(np.arctan((1/(low_rank*low_rank_eigen[0]))*np.sum(low_rank_eigen),MC_p))
                input_channel_newQC.append(np.arctan2(input_channel_S[-1],MC_p))
                # print('input qc new shape', input_channel_newQC)
                # NOTE never used (below)
                # mode_3_unfold_approx = torch.matmul(
                #     U_approx, torch.matmul(S_approx, torch.t(V_approx)))
            except Exception:
                # raise NotImplementedError
                U_approx = torch.zeros(mode_3_unfold.shape[0], 0)
                S_approx = torch.zeros(0, 0)
                V_approx = torch.zeros(mode_3_unfold.shape[1], 0)
                if epoch > 0:
                    # input_channel_rank.append(
                    #     variables_performance['in_rank_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # input_channel_S.append(
                    #     variables_performance['in_S_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # input_channel_condition.append(
                    #     variables_performance['in_condition_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    input_channel_rank.append(
                        self.historical_metrics[-1].input_channel_rank[block_index])
                    input_channel_S.append(
                        self.historical_metrics[-1].input_channel_S[block_index])
                    input_channel_condition.append(
                        self.historical_metrics[-1].input_channel_condition[block_index])
                    input_channel_QC.append(
                        self.historical_metrics[-1].input_channel_QC[block_index])
                    input_channel_newQC.append(
                        self.historical_metrics[-1].input_channel_newQC[block_index])
                else:
                    input_channel_rank.append(0)
                    input_channel_S.append(0)
                    input_channel_condition.append(0)
                    input_channel_QC.append(0)
                    input_channel_newQC.append(0)

            try:
                U_approx, S_approx, V_approx = EVBMF(mode_4_unfold)
                output_channel_rank.append(
                    S_approx.shape[0] / tensor_size[0])
                low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
                low_rank_eigen = low_rank_eigen ** self.p
                if len(low_rank_eigen) != 0:
                    output_channel_condition.append(
                        low_rank_eigen[0] / low_rank_eigen[-1])
                    sum_low_rank_eigen = low_rank_eigen / \
                        max(low_rank_eigen)
                    sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
                else:
                    output_channel_condition.append(0)
                    sum_low_rank_eigen = 0
                output_channel_S.append(
                    sum_low_rank_eigen / tensor_size[0])
                low_rank = output_channel_rank[-1]
                if output_channel_condition[-1] == 0:
                    MC_p = 1
                else:
                    MC_p = 1 - 1 / output_channel_condition[-1]
                if MC_p == 0:
                    MC_p = 0.0001
                output_channel_QC.append(np.arctan2(low_rank, MC_p))
                #output_channel_newQC.append(np.arctan((1/(low_rank*low_rank_eigen[0]))*np.sum(low_rank_eigen),MC_p))
                output_channel_newQC.append(np.arctan2(output_channel_S[-1],MC_p))
                
                # NOTE never used (below)
                factorized_index_4[block_index] = True
                # mode_4_unfold_approx = torch.matmul(
                #     U_approx, torch.matmul(S_approx, torch.t(V_approx)))
            except Exception:
                # NOTE never used (below)
                # U_approx = torch.zeros(mode_3_unfold.shape[0], 0)
                S_approx = torch.zeros(0, 0)
                # NOTE never used (below)
                # V_approx = torch.zeros(mode_3_unfold.shape[1], 0)
                if epoch > 0:
                    # output_channel_rank.append(
                    #     variables_performance['out_rank_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # output_channel_S.append(
                    #     variables_performance['out_S_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    # output_channel_condition.append(
                    #     variables_performance['out_condition_epoch_' +
                    #                           str(epoch - 1)][block_index])
                    output_channel_rank.append(
                        self.historical_metrics[-1].output_channel_rank[block_index])
                    output_channel_S.append(
                        self.historical_metrics[-1].output_channel_S[block_index])
                    output_channel_condition.append(
                        self.historical_metrics[-1].output_channel_condition[block_index])
                    output_channel_QC.append(
                        self.historical_metrics[-1].output_channel_QC[block_index])
                    output_channel_newQC.append(
                        self.historical_metrics[-1].output_channel_newQC[block_index])
                else:
                    output_channel_rank.append(0)
                    output_channel_S.append(0)
                    output_channel_condition.append(0)
                    output_channel_QC.append(0)
                    output_channel_newQC.append(0)
            ''' New metrics calculations'''
            if self.config['sel_metric'] == 'MR-sw':
                try:
                    big_W_shape = self.big_W[block_index].shape
                    #U_approx_n, S_approx_n, V_approx_n = np_EVBMF(self.big_W[block_index]) # numpy should save sys mem
                    S_approx_n = np_EVBMF_s_only(self.big_W[block_index])
                    # debug to check if numpy's EVBMF implementation is equal to torch's
                    # U_approx_t, S_approx_t, V_approx_t = EVBMF(torch.tensor(self.big_W[block_index]).cpu())
                    # if (U_approx_n != U_approx_t.data.cpu().numpy()).all() or (S_approx_n != S_approx_t.data.cpu().numpy()).all() or (V_approx_n != V_approx_t.data.cpu().numpy()).all():
                    #     try:
                    #         raise ValueError("np didnt match torch!")
                    #     except:
                    #         if (U_approx_n != U_approx_t.data.cpu().numpy()).all():
                    #             print("U didnt match")
                    #             print("np:", U_approx_n)
                    #             print("torch:", U_approx_t.data.cpu().numpy())
                    #         if (S_approx_n != S_approx_t.data.cpu().numpy()).all():
                    #             print("S didnt match")
                    #             print("np:", S_approx_n)
                    #             print("torch:", S_approx_t.data.cpu().numpy())
                    #         if (V_approx_n != V_approx_t.data.cpu().numpy()).all():
                    #             print("V didnt match")
                    #             print("np:", V_approx_n)
                    #             print("torch:", V_approx_t.data.cpu().numpy())

                    rank_val = S_approx_n.shape[0] / min(big_W_shape[0],big_W_shape[1])
                    # rank
                    big_W_rank.append(rank_val)
                    self.cumulative_rank[block_index] += rank_val
                    # MC
                    # low_rank_eigen = torch.diag(S_approx_t).data.cpu().numpy() # torch uses too much sys mem
                    low_rank_eigen = np.diag(S_approx_n)
                    low_rank_eigen = low_rank_eigen ** self.p
                    if len(low_rank_eigen) != 0:
                        cond_val = low_rank_eigen[0] / low_rank_eigen[-1]
                        big_W_condition.append(cond_val)
                        self.cumulative_cond[block_index] += cond_val
                        sum_low_rank_eigen = low_rank_eigen / max(low_rank_eigen)
                        sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
                    else:
                        big_W_condition.append(0)
                        self.cumulative_cond[block_index] += 0
                        sum_low_rank_eigen = 0
                    # KG
                    KG_val = sum_low_rank_eigen / min(big_W_shape[0],big_W_shape[1])
                    big_W_S.append(KG_val)
                    self.cumulative_S[block_index] += KG_val
                    factorized_index_bigW[block_index] = True
                except Exception:
                    print("COULD NOT COMPUTE EVBMF ON STACKED WEIGHT AT LAYER: ", block_index)
                    if epoch > 0:
                        big_W_rank.append(
                            self.historical_metrics[-1].big_W_rank[block_index])
                        big_W_S.append(
                            self.historical_metrics[-1].big_W_S[block_index])
                        big_W_condition.append(
                            self.historical_metrics[-1].big_W_condition[block_index])
                        self.cumulative_S[block_index] += self.historical_metrics[-1].cumulative_S[block_index]
                        self.cumulative_rank[block_index] += self.historical_metrics[-1].cumulative_rank[block_index]
                        self.cumulative_cond[block_index] += self.historical_metrics[-1].cumulative_cond[block_index]
                    else:
                        big_W_rank.append(0)
                        big_W_S.append(0)
                        big_W_condition.append(0)
                        self.cumulative_S[block_index] = 0
                        self.cumulative_rank[block_index] = 0
                        self.cumulative_cond[block_index] = 0



        false_indices_12 = [i for i in range(len(factorized_index_12))
                            if factorized_index_12[i] is False]
        false_indices_3 = [i for i in range(len(factorized_index_3))
                           if factorized_index_3[i] is False]
        false_indices_4 = [i for i in range(len(factorized_index_4))
                           if factorized_index_4[i] is False]
        for false_index in false_indices_12:
            mode_12_channel_S[false_index] = mode_12_channel_S[false_index - 1]
            mode_12_channel_rank[false_index] = \
                mode_12_channel_rank[false_index - 1]
            mode_12_channel_condition[false_index] = \
                mode_12_channel_condition[false_index - 1]
        for false_index in false_indices_3:
            input_channel_S[false_index] = input_channel_S[false_index - 1]
            input_channel_rank[false_index] = \
                input_channel_rank[false_index - 1]
            input_channel_condition[false_index] = \
                input_channel_condition[false_index - 1]
            input_channel_QC[false_index] = \
                input_channel_QC[false_index - 1]
            input_channel_newQC[false_index] = \
                input_channel_newQC[false_index - 1]
        for false_index in false_indices_4:
            output_channel_S[false_index] = output_channel_S[false_index - 1]
            output_channel_rank[false_index] = \
                output_channel_rank[false_index - 1]
            output_channel_condition[false_index] = \
                output_channel_condition[false_index - 1]
            output_channel_QC[false_index] = \
                output_channel_QC[false_index - 1]
            output_channel_newQC[false_index] = \
                output_channel_newQC[false_index - 1]
        if self.config['sel_metric'] == 'MR-sw':
            false_indices_bigW = [i for i in range(len(factorized_index_bigW))
                                  if factorized_index_bigW[i] is False]
            for false_index in false_indices_bigW:
                big_W_S[false_index] = big_W_S[false_index - 1]
                big_W_rank[false_index] = \
                    big_W_rank[false_index - 1]
                big_W_condition[false_index] = \
                    big_W_condition[false_index - 1]
                self.cumulative_rank[false_index] = \
                    self.cumulative_rank[false_index - 1]
                self.cumulative_S[false_index] = \
                    self.cumulative_S[false_index - 1]
                self.cumulative_cond[false_index] = \
                    self.cumulative_cond[false_index - 1]
        metrics = IOMetrics(input_channel_rank=input_channel_rank,
                            input_channel_S=input_channel_S,
                            input_channel_condition=input_channel_condition,
                            input_channel_QC=input_channel_QC,
                            output_channel_rank=output_channel_rank,
                            output_channel_S=output_channel_S,
                            output_channel_condition=output_channel_condition,
                            output_channel_QC=output_channel_QC,
                            mode_12_channel_rank=mode_12_channel_rank,
                            mode_12_channel_S=mode_12_channel_S,
                            mode_12_channel_condition=mode_12_channel_condition,
                            big_W_rank=big_W_rank,
                            big_W_S=big_W_S,
                            big_W_condition=big_W_condition,
                            cumulative_rank=self.cumulative_rank,
                            cumulative_S=self.cumulative_S,
                            cumulative_cond=self.cumulative_cond,
                            fc_S=output_channel_S[-1],
                            fc_rank=output_channel_rank[-1],
                            input_channel_newQC = input_channel_newQC,
                            output_channel_newQC = output_channel_newQC)
        self.historical_metrics.append(metrics)
        return metrics

def test_update_dummy():
    a = torch.randn((3,1))
    b = list()
    b.append(np.array(([1,2],[3,4],[5,6]),dtype=float))
    a = a.cpu().numpy()
    print(a)
    print(b)
    b[0] = np.concatenate((b[0][:,1:],a),axis=1)
    print(b)
    b[0][:,1] = a.squeeze()
    print(b[0])

def test():
    a = [5,10]
    b = [4,12]
    stack = np.stack([a,b],axis=0)
    print(np.amax(stack,axis=0))

if __name__ == '__main__':
    # test_update_dummy()
    test()