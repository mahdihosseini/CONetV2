import math
import numpy as np
import copy
import random
from Channel_Search.scaling_method import getScalingMethod
from searching_algorithm.common import (get_layer_conv_trial_metric,
                                    get_metric_by_dependency,
                                    getCumulativeMetricPerDependency,
                                    update_momentum_m)

def acceptingFunction(delt, Temp):
    return math.exp((-delt/Temp))
def SA_scaling_algorithm(dependency_list,conv_size_list, gamma, delta_scale, sel_metric, initial_step_size, min_conv_size, max_conv_size, trial_dir, cur_trial, temp, totalTrials,mapping,global_config):
    # for first trial, initialize mM and increase all channel size by initial_step_size
    ALPHA = global_config.CONFIG['ALPHA']
    temp = ALPHA * (totalTrials - cur_trial)/(totalTrials)
    if cur_trial == 0:
        # initialize
        global_config.old_conv_size = copy.deepcopy(conv_size_list)
        in_m, out_m = get_layer_conv_trial_metric(trial_dir, cur_trial, sel_metric)
        global_config.in_momentum_m = np.zeros(len(in_m), dtype=float)
        global_config.out_momentum_m = np.zeros(len(out_m), dtype=float)
        update_momentum_m(in_m, out_m, gamma,global_config = global_config) # initialize mM to m
        cell_layer_avg_mM = get_metric_by_dependency(dependency_list,global_config.in_momentum_m, global_config.out_momentum_m, mapping)
        # increase all conv sizes by initial_step_size
        for idx, cell in enumerate(conv_size_list):
            conv_size_list[idx] = copy.deepcopy((np.array(cell) + initial_step_size).tolist())
        return conv_size_list, cell_layer_avg_mM, temp

    # Record metric from prev trial
    in_mM_old = global_config.in_momentum_m
    out_mM_old = global_config.out_momentum_m
    conv_size_avg_mM_old = get_metric_by_dependency(dependency_list, in_mM_old, out_mM_old, mapping)

    # get the new mM
    in_m, out_m = get_layer_conv_trial_metric(trial_dir, cur_trial, sel_metric)
    update_momentum_m(in_m, out_m, gamma,global_config = global_config)
    in_mM_new = global_config.in_momentum_m
    out_mM_new = global_config.out_momentum_m
    conv_size_avg_mM_new = get_metric_by_dependency(dependency_list, in_mM_new, out_mM_new, mapping)

    global_config.old_conv_size = copy.deepcopy(conv_size_list)
    # update each conv sizes based on delta mM
    for dep_index, dep_metric_new in enumerate(conv_size_avg_mM_new):
        # Get the difference in metrics
        metric_difference = getCumulativeMetricPerDependency(dep_metric_new) - getCumulativeMetricPerDependency(conv_size_avg_mM_old[dep_index])
        # Get the current dimension for the dependency list
        try:
            cur_dim = conv_size_list[mapping[int(list(dependency_list[dep_index][0].keys())[0].split('_')[-2])]]
        except:
            cur_dim = conv_size_list[mapping[dependency_list[dep_index][0][0]]]

        # Generate new dimension based on metric_difference

       
        if random.random() < acceptingFunction(1/(abs(metric_difference)), temp):
            
            new_dim = getScalingMethod(method_name = global_config.CONFIG['scaling_method'],metric_type = global_config.CONFIG['sel_metric'], metric_difference = metric_difference + acceptingFunction(1/(abs(metric_difference)), temp), prev_channel_size = cur_dim)
        else:
            new_dim = getScalingMethod(method_name = global_config.CONFIG['scaling_method'],metric_type = global_config.CONFIG['sel_metric'], metric_difference = metric_difference, prev_channel_size = cur_dim)
      

        for layer_channel in dependency_list[dep_index]:
            # Only need to update the output dimension sizes'
            try:
                layer_index = int(list(layer_channel.keys())[0].split('_')[-2])
                in_or_out = list(layer_channel.keys())[0].split('_')[-1]
            except:
                layer_index = layer_channel[0]
                in_or_out = layer_channel[1]
            if in_or_out == 'in':
                continue
            conv_size_list[mapping[layer_index]] = new_dim

    return conv_size_list, conv_size_avg_mM_new, temp

