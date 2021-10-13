from typing import Tuple
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
# import platform
import time
import pandas as pd
# import gc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import copy

from optim import get_optimizer_scheduler
from optim.sls import SLS
from optim.sps import SPS
from optim.AdaS import AdaS
from data import get_data
from metrics import Metrics
from utils.early_stop import EarlyStop
from utils.test import (AverageMeter,accuracy,test_main)


# import configs.global_vars as GLOBALS

def get_loss(loss: str) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss() if loss == 'cross_entropy' else None

def initialize(args: APNamespace, new_network, beta=None, new_threshold=None, new_threshold_kernel=None, scheduler=None, init_lr=None, load_config=False, trial=-1,global_config = None):
    assert global_config is not None, 'Please provide a config file'
    root_path = Path(args.root).expanduser()
    data_path = root_path / Path(args.data).expanduser()

    if scheduler != None:
        global_config.CONFIG['lr_scheduler']='StepLR'

    if init_lr != None:
        global_config.CONFIG['init_lr'] = init_lr

    if beta != None:
        global_config.CONFIG['beta'] = beta

    # Set up optimization variables
    if beta is None:
        beta = global_config.CONFIG['beta']
    if scheduler is None:
        scheduler = global_config.CONFIG['lr_scheduler']
    if init_lr is None:
        init_lr = global_config.CONFIG['init_lr']
    # config_path = Path(args.config).expanduser()
    #parse from yaml
    # if load_config:
    #     with config_path.open() as f:
    #         GLOBALS.CONFIG = parse_config(yaml.load(f))

    # Populate function variables with global variable values
    # scheduler = GLOBALS.CONFIG['lr_scheduler']

    # init_lr = GLOBALS.CONFIG['init_lr']
    
    
    print('~~~GLOBALS.CONFIG:~~~')
    # print(GLOBALS.CONFIG)
    print(global_config.CONFIG)
    print("Adas: Argument Parser Options")
    print("-"*45)
    print(f"    {'config':<20}: {args.config:<40}")
    print(f"    {'data':<20}: {str(Path(args.root) / args.data):<40}")
    print(f"    {'output':<20}: {str(Path(args.root) / args.output):<40}")
    #print(f"    {'checkpoint':<20}: " + No checkpoints used
    #      f"{str(Path(args.root) / args.checkpoint):<40}")
    print(f"    {'root':<20}: {args.root:<40}")
    #print(f"    {'resume':<20}: {'True' if args.resume else 'False':<20}") No checkpoints / resumes used
    
    print("\nAdas: Train: Config")
    print(f"    {'Key':<20} {'Value':<20}")
    print("-"*45)
    
    #for k, v in GLOBALS.CONFIG.items():
    for k, v in global_config.CONFIG.items():
        if isinstance(v, list):
            print(f"    {k:<20} {v}")
        else:
            print(f"    {k:<20} {v:<20}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"AdaS: Pytorch device is set to {device}")
    
    # Initialize Global Best Accuracy
    global_config.BEST_ACC = 0  # best test accuracy - needs to be set to 0 every trial / full train

    '''
    Early Stopping Implementation
    '''
    if np.less(float(global_config.CONFIG['early_stop_threshold']), 0):
        print(
            "AdaS: Notice: early stop will not be used as it was set to " +
            f"{global_config.CONFIG['early_stop_threshold']}, training till " +
            "completion")
    elif global_config.CONFIG['optim_method'] != 'SGD' and \
            global_config.CONFIG['lr_scheduler'] != 'AdaS':
        print(
            "AdaS: Notice: early stop will not be used as it is not SGD with" +
            " AdaS, training till completion")
        global_config.CONFIG['early_stop_threshold'] = -1

    '''
    Set up the dataset
    '''
    train_loader, test_loader = get_data(
                root=data_path,
                dataset=global_config.CONFIG['dataset'],
                mini_batch_size=global_config.CONFIG['mini_batch_size'],
                cutout=global_config.CONFIG['cutout'],
                cutout_length = global_config.CONFIG['cutout_length'])

    global_config.PERFORMANCE_STATISTICS = {}
    #Gets initial conv size list (string) from config yaml file and converts into int list
    # init_conv = [int(conv_size) for conv_size in GLOBALS.CONFIG['init_conv_setting'].split(',')]

    '''if GLOBALS.CONFIG['blocks_per_superblock']==2:
        GLOBALS.super1_idx = [64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64]
    else:
        GLOBALS.super1_idx = [64,64,64,64,64,64,64]
        GLOBALS.super2_idx = [64,64,64,64,64,64]
        GLOBALS.super3_idx = [64,64,64,64,64,64]
        GLOBALS.super4_idx = [64,64,64,64,64,64]'''

    # GLOBALS.index_used = GLOBALS.super1_idx + GLOBALS.super2_idx + GLOBALS.super3_idx + GLOBALS.super4_idx

    """
    if GLOBALS.FIRST_INIT == True and new_network == 0:
        print('FIRST_INIT==True, GETTING NET FROM CONFIG')
        GLOBALS.NET = get_net(
                    GLOBALS.CONFIG['network'], num_classes=10 if
                    GLOBALS.CONFIG['dataset'] == 'CIFAR10' else 100 if
                    GLOBALS.CONFIG['dataset'] == 'CIFAR100'
                    else 1000, init_adapt_conv_size=init_conv)
        GLOBALS.FIRST_INIT = False
    else:
        print('GLOBALS.FIRST_INIT IS FALSE LOADING IN NETWORK FROM UPDATE (Fresh weights)')
        GLOBALS.NET = new_network
    """
    
    '''
    Set up the metrics class
    '''
    global_config.METRICS = Metrics(list(new_network.parameters()),p=global_config.CONFIG['p'],config = global_config.CONFIG)
    # print("Memory before sending model to cuda:", torch.cuda.memory_allocated(0))
    model = new_network.to(device)
    # print("Memory after sending model to cuda:", torch.cuda.memory_allocated(0))
    global_config.CRITERION = get_loss(global_config.CONFIG['loss'])

    # if beta != None:
    #     GLOBALS.CONFIG['beta']=beta

    # if new_threshold != None:
    #     GLOBALS.CONFIG['delta_threshold']=new_threshold

    # if new_threshold_kernel != None:
    #     GLOBALS.CONFIG['delta_threshold_kernel']=new_threshold_kernel

    if args.train_num > 0:
        global_config.CONFIG['train_num'] = args.train_num

    optimizer, scheduler = get_optimizer_scheduler(
        net_parameters=model.parameters(),
        listed_params=list(model.parameters()),
        beta = beta,
        init_lr=init_lr,
        # optim_method=global_config.CONFIG['optim_method'],
        lr_scheduler=scheduler,#global_config.CONFIG['lr_scheduler'],
        train_loader_len=len(train_loader),
        config=global_config)

    global_config.EARLY_STOP = EarlyStop(
        patience=int(global_config.CONFIG['early_stop_patience']),
        threshold=float(global_config.CONFIG['early_stop_threshold']))

    #GLOBALS.OPTIMIZER = optimizer
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    return train_loader,test_loader,device,optimizer,scheduler,model

def run_trials(epochs, output_path_train, new_threshold=None):
    last_operation, factor_scale, delta_percentage, last_operation_kernel, factor_scale_kernel, delta_percentage_kernel = [], [], [], [], [], []
    parameter_type = GLOBALS.CONFIG['parameter_type']
    trial_dir = output_path_train #os.path.join(GLOBALS.OUTPUT_PATH_STRING, 'Trials')
    # print(trial_dir)

    # kernel_begin_trial = 0

    def check_last_operation(last_operation, last_operation_kernel, kernel_begin_trial):
        all_channels_stopped = True
        for blah in last_operation:
            for inner in blah:
                if inner != 0:
                    all_channels_stopped = False
        all_kernels_stopped = True
        # if kernel_begin_trial!=0:
        for blah in last_operation_kernel:
            for inner in blah:
                if inner != 0:
                    all_kernels_stopped = False
        return all_channels_stopped, all_kernels_stopped

    def get_shortcut_indexes(conv_size_list):
        shortcut_indexes = []
        counter = -1
        for j in conv_size_list:
            if len(shortcut_indexes) == len(conv_size_list) - 1:
                break
            counter += len(j) + 1
            shortcut_indexes += [counter]
        return shortcut_indexes

    

    def should_break(i, all_channels_stopped, all_kernels_stopped, kernel_begin_trial, parameter_type):
        break_loop = False
        if (all_channels_stopped == True and kernel_begin_trial == 0) or i == GLOBALS.CONFIG['adapt_trials']:
            GLOBALS.CONFIG['adapt_trials'] = i
            parameter_type = 'kernel'
            kernel_begin_trial = i
            if GLOBALS.CONFIG['adapt_trials_kernel'] == 0 or GLOBALS.CONFIG['kernel_adapt'] == 0:
                print('ACTIVATED IF STATEMENT 1 FOR SOME STUPID REASON')
                break_loop = True

        if (all_kernels_stopped == True or i == kernel_begin_trial + GLOBALS.CONFIG[
            'adapt_trials_kernel']) and kernel_begin_trial != 0:  # and kernel_begin_trial!=0:
            print('ACTIVATED IF STATEMENT 2 FOR SOME EVEN STUPIDER REASON')
            break_loop = True
        return kernel_begin_trial, parameter_type, break_loop

    #####################################################################################################################################
    conv_data, kernel_data, rank_final_data, rank_stable_data, delta_info, delta_info_kernel, conv_size_list, kernel_size_list = initialize_dataframes_and_lists()
    shortcut_indexes = get_shortcut_indexes(conv_size_list)
    # print("Memory before allocation:", torch.cuda.memory_allocated(0))
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    if GLOBALS.CONFIG['dataset'] == 'CIFAR10':
        class_num = 10
    elif GLOBALS.CONFIG['dataset'] == 'CIFAR100':
        class_num = 100
    new_network = DASNet34(num_classes_input=class_num)
    train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, new_network)

    #New: Because we are using weight transfer, we can't just keep using the same initial LR!
    cur_LR = GLOBALS.CONFIG['init_lr']
    LR_trial_step = 20 // GLOBALS.CONFIG['epochs_per_trial']
    if LR_trial_step <= 0:
        LR_trial_step = 1

    interrupted_trial = 0 # Determines at which trial we will resume!
    if args.resume_search is False:
        run_epochs(0, model, range(0,GLOBALS.CONFIG['initial_epochs']), train_loader, test_loader, device, optimizer, scheduler, output_path_train)
        # run_epochs(0, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_train)
        # print("Memory after first trial:", torch.cuda.memory_allocated(0))
    else:
        interrupted_trial = get_latest_completed_trial(trial_dir)

    # del model
    model.to('cpu')
    del train_loader
    del test_loader
    del optimizer
    del scheduler

    free_cuda_memory()

    print('~~~First run_epochs done.~~~')

    if (GLOBALS.CONFIG['kernel_adapt'] == 0):
        GLOBALS.CONFIG['adapt_trials_kernel'] = 0

    GLOBALS.total_trials = GLOBALS.CONFIG['adapt_trials'] + GLOBALS.CONFIG['adapt_trials_kernel']
    INITIAL_TEMPERATURE = 1000
    TEMP = INITIAL_TEMPERATURE
    for i in range(1, GLOBALS.total_trials):
        """
        if (GLOBALS.CONFIG['kernel_adapt'] == 0):
            GLOBALS.CONFIG['adapt_trials_kernel'] = 0
        if kernel_begin_trial != 0:
            if (i > (GLOBALS.total_trials // 2 - kernel_begin_trial)) and all_channels_stopped == True:
                GLOBALS.min_kernel_size_1 = GLOBALS.CONFIG['min_kernel_size']
                GLOBALS.CONFIG['min_kernel_size'] = GLOBALS.CONFIG['min_kernel_size_2']
                
        """
        '------------------------------------------------------------------------------------------------------------------------------------------------'
        """
        last_operation, last_operation_kernel, factor_scale, factor_scale_kernel, new_channel_sizes, new_kernel_sizes, delta_percentage, delta_percentage_kernel, rank_averages_final, rank_averages_stable = delta_scaling(
            conv_size_list, kernel_size_list, shortcut_indexes, last_operation, factor_scale, delta_percentage,
            last_operation_kernel, factor_scale_kernel, delta_percentage_kernel, parameter_type=parameter_type)
        """

        # new_channel_sizes, delta_percentage, last_operation, factor_scale, \
        # cell_list_rank  = \
        #     delta_scaling(conv_size_list, GLOBALS.CONFIG['delta_threshold'], \
        #                     GLOBALS.CONFIG['min_scale_limit'], GLOBALS.CONFIG['mapping_condition_threshold'], \
        #                     GLOBALS.CONFIG['min_conv_size'], GLOBALS.CONFIG['max_conv_size'],
        #                     trial_dir, i - 1, last_operation, factor_scale)
        #conv_size_list, gamma, delta_scale, sel_metric, initial_step_size, min_conv_size, max_conv_size, trial_dir, cur_trial
        if (GLOBALS.CONFIG['optimization_algorithm'] == 'SA'):
            print("--~~[SA Trial]~~--")
            new_channel_sizes, avg_mM, TEMP = getScalingAlgorithm(GLOBALS.CONFIG['optimization_algorithm'],conv_size_list = conv_size_list, gamma = GLOBALS.CONFIG['gamma'],delta_scale =  GLOBALS.CONFIG['delta_scale'], sel_metric = GLOBALS.CONFIG['sel_metric'], \
                                                              initial_step_size = GLOBALS.CONFIG['init_step_size'], min_conv_size =  GLOBALS.CONFIG['min_conv_size'], max_conv_size = GLOBALS.CONFIG['max_conv_size'], trial_dir = trial_dir, cur_trial = i - 1, temp= TEMP, totalTrials = GLOBALS.total_trials)
        #! ADD TEMPERATURE CONDITON
        else:
            new_channel_sizes, avg_mM = getScalingAlgorithm(GLOBALS.CONFIG['optimization_algorithm'],conv_size_list = conv_size_list, gamma = GLOBALS.CONFIG['gamma'],delta_scale =  GLOBALS.CONFIG['delta_scale'], sel_metric = GLOBALS.CONFIG['sel_metric'], \
                                                              initial_step_size = GLOBALS.CONFIG['init_step_size'], min_conv_size =  GLOBALS.CONFIG['min_conv_size'], max_conv_size = GLOBALS.CONFIG['max_conv_size'], trial_dir = trial_dir, cur_trial = i - 1)
        # new_channel_sizes, avg_mM = delta_scaling_algorithm(conv_size_list, GLOBALS.CONFIG['gamma'], GLOBALS.CONFIG['delta_scale'], GLOBALS.CONFIG['sel_metric'], \
        #                                                      GLOBALS.CONFIG['init_step_size'], GLOBALS.CONFIG['min_conv_size'], GLOBALS.CONFIG['max_conv_size'], trial_dir, i - 1)
        delta_percentage, last_operation, factor_scale, cell_list_rank = [], [], [], []
        '------------------------------------------------------------------------------------------------------------------------------------------------'

        # print(last_operation_kernel, 'LAST OPERATION KERNEL FOR TRIAL ' + str(i))

        """
        all_channels_stopped, all_kernels_stopped = check_last_operation(last_operation, last_operation_kernel,
                                                                         kernel_begin_trial)
        print(all_channels_stopped, all_kernels_stopped, 'BREAK VALUES!')
        kernel_begin_trial, parameter_type, break_loop = should_break(i, all_channels_stopped, all_kernels_stopped,
                                                                      kernel_begin_trial, parameter_type)
        if break_loop == True:
            GLOBALS.total_trials = i
            break
        """

        # last_operation_copy, factor_scale_copy, delta_percentage_copy = copy.deepcopy(
        #     last_operation), copy.deepcopy(factor_scale), copy.deepcopy(delta_percentage)
       # last_operation_kernel_copy, factor_scale_kernel_copy, delta_percentage_kernel_copy = copy.deepcopy(
       #     last_operation_kernel), copy.deepcopy(factor_scale_kernel), copy.deepcopy(delta_percentage_kernel)
        conv_size_list = copy.deepcopy(new_channel_sizes)
        print(conv_size_list)
        # old_kernel_size_list = copy.deepcopy(kernel_size_list)
        # kernel_size_list = copy.deepcopy(new_kernel_sizes)

        print('~~~Writing to Dataframe~~~')
        if parameter_type == 'channel':
            conv_data.loc[i] = new_channel_sizes
            delta_info.loc[i] = [GLOBALS.in_momentum_m, GLOBALS.out_momentum_m, avg_mM]
        
        # rank_final_data.loc[i] = rank_averages_final_copy
        # rank_stable_data.loc[i] = rank_averages_stable_copy

        print('~~~Starting Conv parameter_typements~~~')

        new_network = update_network(new_channel_sizes, None)
        weight_transfer(model, new_network, GLOBALS.CONFIG['wt_type'], "module.")
        del model

        print('~~~Initializing the new model~~~')

        # New use new LR
        #if i % LR_trial_step == 0:
        #    cur_LR = cur_LR*0.5 #drop it by halve



        train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, new_network,
                                                                                    new_threshold_kernel=new_threshold, init_lr = cur_LR)

        print("Channel Sizes", new_channel_sizes)
        print("Trial: ", i, " LR ", cur_LR)
        #print("Memory allocated before trial:", torch.cuda.memory_allocated(0))
        epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

        if i < interrupted_trial:
            print('~~~Using previous training data~~~')

        else:
            print('~~~Training with new model~~~')
            run_epochs(i, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_train)
        # print("Memory allocated after trial:", torch.cuda.memory_allocated(0))

        # del model
        model.to('cpu')
        del train_loader
        del test_loader
        del optimizer
        del scheduler

        free_cuda_memory()

        #Use default
        new_kernel_sizes = [GLOBALS.super1_kernel_idx,GLOBALS.super2_kernel_idx,GLOBALS.super3_kernel_idx,GLOBALS.super4_kernel_idx]
    del model
    return kernel_data, conv_data, rank_final_data, rank_stable_data, new_channel_sizes, new_kernel_sizes, delta_info, delta_info_kernel

def epoch_iteration(trial, model, train_loader, test_loader, epoch: int,
                    device, optimizer,scheduler,global_config) -> Tuple[float, float]:
    # logging.info(f"Adas: Train: Epoch: {epoch}")
    # global net, performance_statistics, metrics, adas, config
    ''' Update new metrics: big_W once every x iterations'''
    update_iter = global_config.CONFIG['update_iter']

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    """train CNN architecture"""
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if global_config.CONFIG['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + batch_idx / len(train_loader))
        optimizer.zero_grad()
        # if GLOBALS.CONFIG['optim_method'] == 'SLS':
        if isinstance(optimizer, SLS):
            def closure():
                outputs = model(inputs)
                loss = global_config.CRITERION(outputs, targets)
                return loss, outputs
            loss, outputs = optimizer.step(closure=closure)
        else:

            #TODO: Revert this if statement when creating separate files
            if global_config.CONFIG['network'] == 'DARTS' or global_config.CONFIG['network'] == 'DARTSPlus':
                outputs, outputs_aux = model(inputs)
                loss = global_config.CRITERION(outputs, targets)
                if global_config.CONFIG['auxiliary']:
                    loss_aux = global_config.CRITERION(outputs_aux, targets)
                    loss = loss + global_config.CONFIG['auxiliary_weight'] * loss_aux
                loss.backward()
                if global_config.CONFIG['grad_clip']:
                    nn.utils.clip_grad_norm(model.parameters(), global_config.CONFIG['grad_clip_threshold'])

            else:
                outputs = model(inputs)
                loss = global_config.CRITERION(outputs, targets)
                loss.backward()

            # if GLOBALS.ADAS is not None:
            #     optimizer.step(GLOBALS.METRICS.layers_index_todo,
            #                    GLOBALS.ADAS.lr_vector)
            if isinstance(scheduler, AdaS):
                optimizer.step(global_config.METRICS.layers_index_todo,
                               scheduler.lr_vector)
            # elif GLOBALS.CONFIG['optim_method'] == 'SPS':
            elif isinstance(optimizer, SPS):
                optimizer.step(loss=loss)
            else:
                optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        acc1_temp, acc5_temp = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1_temp[0], inputs.size(0))
        top5.update(acc5_temp[0], inputs.size(0))

        global_config.TRAIN_LOSS = train_loss
        global_config.TRAIN_CORRECT = correct
        global_config.TRAIN_TOTAL = total

        if global_config.CONFIG['lr_scheduler'] == 'OneCycleLR':
            scheduler.step()

        ''' Update new metric '''
        if global_config.CONFIG['sel_metric'] == 'MR-sw':
            if batch_idx % update_iter == 0:
                global_config.METRICS.update_big_W()
        #Update optimizer
        #GLOBALS.OPTIMIZER = optimizer

        # progress_bar(batch_idx, len(train_loader),
        #              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1),
        #                  100. * correct / total, correct, total))

    acc = top1.avg.cpu().item() / 100
    acc5 = top5.avg.cpu().item() / 100

    global_config.PERFORMANCE_STATISTICS[f'train_acc_epoch_{epoch}'] = \
        float(acc)
    global_config.PERFORMANCE_STATISTICS[f'train_acc5_epoch_{epoch}'] = \
        float(acc5)
    global_config.PERFORMANCE_STATISTICS[f'train_loss_epoch_{epoch}'] = \
        train_loss / (batch_idx + 1)

    io_metrics = global_config.METRICS.evaluate(epoch)
    global_config.PERFORMANCE_STATISTICS[f'in_S_epoch_{epoch}'] = \
        io_metrics.input_channel_S
    global_config.PERFORMANCE_STATISTICS[f'out_S_epoch_{epoch}'] = \
        io_metrics.output_channel_S
    global_config.PERFORMANCE_STATISTICS[f'mode12_S_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_S
    global_config.PERFORMANCE_STATISTICS[f'fc_S_epoch_{epoch}'] = \
        io_metrics.fc_S
    global_config.PERFORMANCE_STATISTICS[f'in_rank_epoch_{epoch}'] = \
        io_metrics.input_channel_rank
    global_config.PERFORMANCE_STATISTICS[f'out_rank_epoch_{epoch}'] = \
        io_metrics.output_channel_rank
    global_config.PERFORMANCE_STATISTICS[f'mode12_rank_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_rank
    global_config.PERFORMANCE_STATISTICS[f'fc_rank_epoch_{epoch}'] = \
        io_metrics.fc_rank
    global_config.PERFORMANCE_STATISTICS[f'in_condition_epoch_{epoch}'] = \
        io_metrics.input_channel_condition
    global_config.PERFORMANCE_STATISTICS[f'out_condition_epoch_{epoch}'] = \
        io_metrics.output_channel_condition
    global_config.PERFORMANCE_STATISTICS[f'mode12_condition_epoch_{epoch}'] = \
        io_metrics.mode_12_channel_condition
    ''' New metrics'''
    global_config.PERFORMANCE_STATISTICS[f'in_QC_epoch_{epoch}'] = \
        io_metrics.input_channel_QC
    global_config.PERFORMANCE_STATISTICS[f'out_QC_epoch_{epoch}'] = \
        io_metrics.output_channel_QC
    global_config.PERFORMANCE_STATISTICS[f'in_newQC_epoch_{epoch}'] = \
        io_metrics.input_channel_newQC
    global_config.PERFORMANCE_STATISTICS[f'out_newQC_epoch_{epoch}'] = \
        io_metrics.output_channel_newQC
    global_config.PERFORMANCE_STATISTICS[f'big_W_rank_epoch_{epoch}'] = \
        io_metrics.big_W_rank
    global_config.PERFORMANCE_STATISTICS[f'big_W_S_epoch_{epoch}'] = \
        io_metrics.big_W_S
    global_config.PERFORMANCE_STATISTICS[f'big_W_condition_epoch_{epoch}'] = \
        io_metrics.big_W_condition
    global_config.PERFORMANCE_STATISTICS[f'cumulative_rank_epoch_{epoch}'] = \
        copy.deepcopy(io_metrics.cumulative_rank)
    global_config.PERFORMANCE_STATISTICS[f'cumulative_S_epoch_{epoch}'] = \
        copy.deepcopy(io_metrics.cumulative_S)
    global_config.PERFORMANCE_STATISTICS[f'cumulative_condition_epoch_{epoch}'] = \
        copy.deepcopy(io_metrics.cumulative_cond)

    # print("big_W_rank: ",io_metrics.big_W_rank)
    # print("big_W_S: ", io_metrics.big_W_S)
    # print("big_W_condition: ", io_metrics.big_W_condition)
    # print("cumulative_rank: ", io_metrics.cumulative_rank)
    # print("cumulative_S: ", io_metrics.cumulative_S)
    # print("cumulative_cond: ", io_metrics.cumulative_cond)

    # if GLOBALS.ADAS is not None:

    if isinstance(scheduler, AdaS):
        lrmetrics = scheduler.step(epoch, global_config.METRICS)
        global_config.PERFORMANCE_STATISTICS[f'rank_velocity_epoch_{epoch}'] = \
            lrmetrics.rank_velocity
        global_config.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
            lrmetrics.r_conv
    else:
        # if GLOBALS.CONFIG['optim_method'] == 'SLS' or \
        #         GLOBALS.CONFIG['optim_method'] == 'SPS':
        if isinstance(optimizer, SLS) or isinstance(optimizer, SPS):
            global_config.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.state['step_size']
        else:
            global_config.PERFORMANCE_STATISTICS[f'learning_rate_epoch_{epoch}'] = \
                optimizer.param_groups[0]['lr']
    test_loss, test_accuracy, test_acc5 = test_main(model, test_loader, epoch, device, optimizer,global_config = global_config)

    return (train_loss / (batch_idx + 1), 100. * acc,
            test_loss, 100 * test_accuracy)

def run_epochs(trial, model, epochs, train_loader, test_loader,
               device, optimizer, scheduler, output_path,global_config):
    # if platform.system == 'Windows':
    #     slash = '\\'
    # else:
    #     slash = '/'
    # print('------------------------------' + slash)

    xlsx_name = \
                f"{global_config.CONFIG['lr_scheduler']}_adapt_trial={trial}_" +\
                f"net={global_config.CONFIG['network']}_" +\
                f"{global_config.CONFIG['init_lr']}_dataset=" +\
                f"{global_config.CONFIG['dataset']}.xlsx"
    # if global_config.CONFIG['lr_scheduler'] == 'AdaS':

    #     if global_config.FULL_TRAIN == False:
    #         xlsx_name = \
    #             f"AdaS_adapt_trial={trial}_" +\
    #             f"net={global_config.CONFIG['network']}_" +\
    #             f"{global_config.CONFIG['init_lr']}_dataset=" +\
    #             f"{global_config.CONFIG['dataset']}.xlsx"
    #     else:
    #         if global_config.FULL_TRAIN_MODE == 'last_trial':
    #             xlsx_name = \
    #                 f"AdaS_last_iter_fulltrain_trial={trial}_" +\
    #                 f"net={global_config.CONFIG['network']}_" +\
    #                 f"dataset=" +\
    #                 f"{global_config.CONFIG['dataset']}.xlsx"
    #         elif global_config.FULL_TRAIN_MODE == 'fresh':
    #             xlsx_name = \
    #                 f"AdaS_fresh_fulltrain_trial={trial}_" +\
    #                 f"net={global_config.CONFIG['network']}_" +\
    #                 f"beta={global_config.CONFIG['beta']}_" +\
    #                 f"dataset=" +\
    #                 f"{global_config.CONFIG['dataset']}.xlsx"
    #         else:
    #             print('ERROR: INVALID FULL_TRAIN_MODE | Check that the correct full_train_mode strings have been initialized in main file | Should be either fresh, or last_trial')
    #             sys.exit()
    # else:
    #     if global_config.FULL_TRAIN == False:
    #         xlsx_name = \
    #             f"StepLR_adapt_trial={trial}_" +\
    #             f"net={global_config.CONFIG['network']}_" +\
    #             f"{global_config.CONFIG['init_lr']}_dataset=" +\
    #             f"{global_config.CONFIG['dataset']}.xlsx"
    #     else:
    #         if global_config.FULL_TRAIN_MODE == 'last_trial':
    #             xlsx_name = \
    #                 f"StepLR_last_iter_fulltrain_trial={trial}_" +\
    #                 f"net={global_config.CONFIG['network']}_" +\
    #                 f"dataset=" +\
    #                 f"{global_config.CONFIG['dataset']}.xlsx"
    #         elif global_config.FULL_TRAIN_MODE == 'fresh':
    #             xlsx_name = \
    #                 f"StepLR_fresh_fulltrain_trial={trial}_" +\
    #                 f"net={global_config.CONFIG['network']}_" +\
    #                 f"dataset=" +\
    #                 f"{global_config.CONFIG['dataset']}.xlsx"
    #         else:
    #             print('ERROR: INVALID FULL_TRAIN_MODE | Check that the correct full_train_mode strings have been initialized in main file | Should be either fresh, or last_trial')
    #             sys.exit()
            
    # # if platform.system == 'Windows':
    # #     slash = '\\'
    # # else:
    # #     slash = '/'

    xlsx_path = os.path.join(str(output_path),xlsx_name)

    # if global_config.FULL_TRAIN == False:
    #     filename = \
    #         f"stats_net={global_config.CONFIG['network']}_AdaS_trial={trial}_" +\
    #         f"beta={global_config.CONFIG['beta']}_initlr={global_config.CONFIG['init_lr']}_" +\
    #         f"dataset={global_config.CONFIG['dataset']}.csv"
    # else:
    #     if global_config.FULL_TRAIN_MODE == 'last_trial':
    #         filename = \
    #             f"stats_last_iter_net={global_config.CONFIG['network']}_StepLR_trial={trial}_" +\
    #             f"beta={global_config.CONFIG['beta']}_" +\
    #             f"dataset={global_config.CONFIG['dataset']}.csv"
    #     elif global_config.FULL_TRAIN_MODE == 'fresh':
    #         filename = \
    #             f"stats_fresh_net={global_config.CONFIG['network']}_StepLR_trial={trial}_" +\
    #             f"beta={global_config.CONFIG['beta']}_" +\
    #             f"dataset={global_config.CONFIG['dataset']}.csv"
    
    global_config.EXCEL_PATH = xlsx_path
    print('SET GLOBALS EXCEL PATH', global_config.EXCEL_PATH)
    best_result = {'Epoch': 0, 'Test_Accuracy': 0, 'Test_Loss': 1000, 'Train_Accuracy': 0, 'Train_Loss': 1000}
    for epoch in epochs:
        start_time = time.time()
        # print(f"AdaS: Epoch {epoch}/{epochs[-1]} Started.")
        # New - Drop Path
        if global_config.CONFIG['drop_path']:
            model.drop_path_prob = global_config.CONFIG['drop_path_prob'] * epoch / global_config.CONFIG['max_epoch']
            
        train_loss, train_accuracy, test_loss, test_accuracy = \
            epoch_iteration(trial, model, train_loader, test_loader,epoch, device, optimizer, scheduler,global_config)

        # Store the best set of results:
        if test_accuracy > best_result['Test_Accuracy']:
            best_result['Epoch'] = epoch
            best_result['Test_Accuracy'] = test_accuracy
            best_result['Test_Loss'] = test_loss
            best_result['Train_Accuracy'] = train_accuracy
            best_result['Train_Loss'] = train_loss
        end_time = time.time()

        if global_config.CONFIG['lr_scheduler'] == 'StepLR':
            scheduler.step(epoch)
            #scheduler.step()
        total_time = time.time()
        print(
            f"Trial {trial}/{global_config.total_trials - 1} | " +
            f"Epoch {epoch}/{epochs[-1]} Ended | " +
            "Total Time: {:.3f}s | ".format(total_time - start_time) +
            "Epoch Time: {:.3f}s | ".format(end_time - start_time) +
            "~Time Left: {:.3f}s | ".format(
                (total_time - start_time) * (epochs[-1] - epoch)),
            "Train Loss: {:.4f}% | Train Acc. {:.4f}% | ".format(
                train_loss,
                train_accuracy) +
            "Test Loss: {:.4f}% | Test Acc. {:.4f}%".format(test_loss,
                                                            test_accuracy))
        # print(type(GLOBALS.PERFORMANCE_STATISTICS))
        # print(GLOBALS.PERFORMANCE_STATISTICS.keys())
        df = pd.DataFrame(data=global_config.PERFORMANCE_STATISTICS)

        df.to_excel(xlsx_path)
        if global_config.EARLY_STOP(train_loss):
            print("AdaS: Early stop activated.")
            break
    global_config.best_result = best_result
    

# def run_fresh_full_train(output_sizes,kernel_sizes,epochs,output_path_fulltrain,global_config):
#     class_num=0
#     # Get the class number
#     if global_config.CONFIG['dataset']=='CIFAR10':
#         class_num=10
#     elif global_config.CONFIG['dataset']=='CIFAR100':
#         class_num=100

#     # Reinitialize Model
#     if global_config.GLOBALS.CONFIG['network']=='DASNet34':
#         new_network=DASNet34(num_classes_input=class_num,new_output_sizes=output_sizes,new_kernel_sizes=kernel_sizes)
#     elif global_config.CONFIG['network']=='DASNet50':
#         new_network=DASNet50(num_classes_input=class_num,new_output_sizes=output_sizes,new_kernel_sizes=kernel_sizes)

#     GLOBALS.FIRST_INIT = False

#     #optimizer,scheduler=network_initialize(new_network,train_loader)
#     parser = ArgumentParser(description=__doc__)
#     get_args(parser)
#     args = parser.parse_args()
#     #free_cuda_memory()
#     train_loader,test_loader,device,optimizer,scheduler, model = initialize(args,new_network,beta=GLOBALS.CONFIG['beta_full'],
#                                                                             scheduler='StepLR', init_lr=GLOBALS.CONFIG['init_lr_full'])

#     GLOBALS.FULL_TRAIN = True
#     GLOBALS.PERFORMANCE_STATISTICS = {}
#     GLOBALS.FULL_TRAIN_MODE = 'fresh'
#     GLOBALS.EXCEL_PATH = ''

#     for param_tensor in model.state_dict():
#         val=param_tensor.find('bn')
#         if val==-1:
#             continue
#         print(param_tensor, "\t", model.state_dict()[param_tensor].size(), 'FRESH')
#         #print(param_tensor, "\t", GLOBALS.NET.state_dict()[param_tensor], 'FRESH')
#         break

#     run_epochs(0, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_fulltrain)
#     # print("Memory allocated full train:", torch.cuda.memory_allocated(0))
#     return model