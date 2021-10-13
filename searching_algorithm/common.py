import os
import numpy as np
import pandas as pd
import copy

def epochSelection(in_m, out_m,epoch_selection):
    if epoch_selection == 'max':
        return np.amax(in_m, axis=1), np.amax(out_m, axis=1)
    elif epoch_selection == 'min':
        return np.amin(in_m, axis=1), np.amin(out_m, axis=1)
    elif epoch_selection == 'avg':
        return np.average(in_m, axis=1), np.average(out_m, axis=1)
    elif epoch_selection == 'last':
        return in_m[:,-1], out_m[:,-1]
    else:
        raise ValueError('Please provide an epoch selection value that is supported')

# Given a metric, read the trial excel file to get MAX in metric and out metric for each layer
# returns MAX in and out metric of each layer
def get_layer_conv_trial_metric(trial_dir,cur_trial,sel_metric,epoch_selection = 'max'):
    cur_trial = 'AdaS_adapt_trial=%s'%str(cur_trial)
    print(trial_dir)
    file_list = os.listdir(trial_dir)
    print(file_list)
    print(cur_trial)
    in_m = list()
    out_m = list()
    if sel_metric == 'MKG':
        sel_in_m = 'in_S'
        sel_out_m = 'out_S'
    elif sel_metric == 'MR':
        sel_in_m = 'in_rank'
        sel_out_m = 'out_rank'
    elif sel_metric == 'MQC':
        sel_in_m = 'in_QC'
        sel_out_m = 'out_QC'
    elif sel_metric =='newQC':
        sel_in_m = 'in_newQC'
        sel_out_m = 'out_newQC'
    elif sel_metric == 'MR-sw':
        # stacked weight doesnt have in and out channels, so we set both as the same
        sel_in_m = 'big_W_rank'
        sel_out_m = 'big_W_rank'
    elif sel_metric == 'MKG-sw':
        # stacked weight doesnt have in and out channels, so we set both as the same
        sel_in_m = 'big_W_S'
        sel_out_m = 'big_W_S'

    
    for file in file_list:
        if file.startswith(cur_trial) and file.endswith('.xlsx'):
            file_path = os.path.join(trial_dir,file)
            df_trial = pd.read_excel(file_path, index_col=0)
            # find all in metric at each layer for all epochs
            cols = [col for col in df_trial if col.startswith(sel_in_m)]
            for col in cols:
                temp = df_trial.loc[:, col].tolist()
                in_m.append(copy.deepcopy(np.array(temp)))
            # find all out metric at each layer for all epochs
            cols = [col for col in df_trial if col.startswith(sel_out_m)]
            for col in cols:
                temp = df_trial.loc[:, col].tolist()
                out_m.append(copy.deepcopy(np.array(temp)))
            break
    print('in_m shape', np.array(in_m).shape)
    in_m = np.array(in_m).transpose((1,0)) # [epoch,layer] -> [layer,epoch]
    out_m = np.array(out_m).transpose((1,0))
    # return np.amax(in_m, axis=1), np.amax(out_m, axis=1)
    return epochSelection(in_m, out_m,epoch_selection = epoch_selection)

# update momentum metric
def update_momentum_m(in_m, out_m, gamma,global_config):
    # compute input momentum S
    global_config.in_momentum_m = global_config.in_momentum_m * gamma + in_m
    # compute output momentum S
    global_config.out_momentum_m = global_config.out_momentum_m * gamma + out_m


# find the avg momentum metric (mM) of the entire cell
def get_cell_mM_avg(in_mM, out_mM, first_conv_idx,last_conv_idx):
    cell_avg = 0
    in_mM_true = True # flips from true to false and back to true
    shortcut_layer = first_conv_idx + 2
    count = 0
    # skip layer 0 (gate layer)
    if first_conv_idx == 0:
        first_conv_idx = 1
    for layer_idx in range(first_conv_idx, last_conv_idx+1):
        # skip the shortcut layer
        if layer_idx == shortcut_layer and first_conv_idx != 1:
            continue
        if in_mM_true:
            mM = in_mM[layer_idx]
            in_mM_true = False
            # print("in layer ",layer_idx)
        else:
            mM = out_mM[layer_idx]
            in_mM_true = True
            # print("out layer ", layer_idx)
        cell_avg += mM
        count += 1
    return cell_avg/count

# get the cell layer mM by the dependency (mapping) indicated in paper
# returns avg mM of each adjustable conv sizes in the format of conv_size_list
def get_metric_by_dependency(dependency_list,in_mM,out_mM,mapping):
    # total_layer = in_mM.shape[0]
    # channel_size_avg_mM = list()

    metrics_list = list()
    for dependency_group in dependency_list:
        metrics_sublist = list()
        for _, layer in enumerate(dependency_group):
            try:
                layer_index = int(list(layer.keys())[0].split('_')[-2])
                in_or_out = list(layer.keys())[0].split('_')[-1]
            except:
                layer_index = layer[0]
                in_or_out = layer[1]
            if in_or_out == 'in':
                metrics_sublist.append(in_mM[mapping[layer_index]])
            elif in_or_out == 'out':
                metrics_sublist.append(out_mM[mapping[layer_index]])
        metrics_list.append(np.array(metrics_sublist))
    return metrics_list

                
def getCumulativeMetricPerDependency(list_of_metrics,cumulation_method = 'average'):
    if cumulation_method == 'average':
        return np.average(list_of_metrics)
    elif cumulation_method == 'product':
        return np.product(list_of_metrics)
    else:
        raise ValueError('Provide a valid method selection, choices are {average, product}')

            
# def scaleChannelSize(metric_type,metric_difference, prev_channel_size):

    


#     # for cur_cell_idx, first_conv_idx in enumerate(cell_first_layer_idx):
#     #     cur_cell_avg_mM = list()
#     #     if cur_cell_idx == len(cell_first_layer_idx) - 1:
#     #         last_conv_idx = total_layer - 1
#     #     else:
#     #         last_conv_idx = cell_first_layer_idx[cur_cell_idx+1] - 1
#     #     #print(first_conv_idx, last_conv_idx)
#     #     shortcut_layer_idx = first_conv_idx + 2
#     #     # average of the entire cell mM, used for C0 (as noted in paper)
#     #     cell_avg = get_cell_mM_avg(in_mM, out_mM, first_conv_idx,last_conv_idx)

#     #     # first cell starts with a gate layer and the conv size is C0
#     #     if cur_cell_idx == 0:
#     #         cur_cell_avg_mM.append(copy.deepcopy(cell_avg))
#     #         first_conv_idx = 1
#     #     is_C0 = False # alternates between False and True
#     #     # iterate through each conv layer in a cell
#     #     for layer_idx in range(first_conv_idx, last_conv_idx+1):
#     #         # skip shortcut layers
#     #         if layer_idx == shortcut_layer_idx and cur_cell_idx != 0:
#     #             #print("is shortcut")
#     #             continue
#     #         # if the channel size is C0, use the entire cell rank avg
#     #         if is_C0:
#     #             cur_cell_avg_mM.append(copy.deepcopy(cell_avg))
#     #             is_C0 = False
#     #             #print("is C0")
#     #             continue
#     #         # if its pair is shortcut layer, skip it (shouldn't be invoked?)
#     #         if layer_idx + 1 == shortcut_layer_idx and cur_cell_idx != 0:
#     #             pair_next = layer_idx + 2
#     #         else:
#     #             pair_next = layer_idx + 1
#     #         pair_avg = (out_mM[layer_idx] + in_mM[pair_next])/2
#     #         cur_cell_avg_mM.append(copy.deepcopy(pair_avg))
#     #         #print("out ", layer_idx, ",in ", pair_next)
#     #         is_C0 = True
#     #     channel_size_avg_mM.append(copy.deepcopy(np.array(cur_cell_avg_mM)))
#     # return channel_size_avg_mM


# # round up to nearest even number for chan size
# def round_even(number):
#     return int(round(number / 2) * 2)