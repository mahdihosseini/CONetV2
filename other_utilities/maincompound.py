'''
- Get model
- Convert Model Summary to DAG
    - Model summary --> Linked list
    - Skip connections
    - Indicies in model 
- Get dependency list of model using DAG
    - Dependency list only contains indicies of input & output of convolution layers
    - Dependency list can be derived from adjacency list
- Get the weights of the model from the model instance by indexing into the model layers
    - Weight Transfer
    - Computing of Metrics
- Metrics computing
    - Input: weight matrix, indicator of whether it is input or output unravelling
    - Returns: computed metrics
- Method of finding a metric for each layer representative of the trial
    - Input: method
    - Output: metric
- Method of accumulating metrics within dependency lists
    - Input: method
    - Output: metric accumulated based on method indicated
- Scaling Method
    - Input: method type (i.e. momentum, default step, etc.), initial step size, metric_prev, metric_curr, metric_type, curr_channel_size, old_channel_size
    - Output: new_channel_size
- Searching algorithm
    - Input: Current channel sizes for dependency list, cumulative metrics, previous cumulative metrics,
            scaling method 
    - Output: list of scaled channel sizes with the same length as dependency list
- Weight Transfer
    - Take dependency list & scaled channel sizes, for each conv layer, get a reshaped matrix
- Insert back into the model
    - Convert dependency list & associated channel sizes into format for model restructuring
    - Initialize new model
    - Insert weight transfer weights into model

'''

from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
import platform
import copy
import configs.global_vars as GLOBALS
from utils.train_helpers import (build_paths,create_trial_data_file,free_cuda_memory,create_train_output_file)
from utils.create_dataframe import initialize_dataframes_and_lists
from models import (get_model,update_network)
from utils.train import (initialize,run_epochs)
from searching_algorithm import getSearchingAlgorithm
from utils.weight_transfer import weight_transfer
from dependency import DependencyList
import numpy as np
from models.resnet import *
import sys
import math

def get_args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("AdaS Train Args")
    # print("---------------------------------\n")
    # sub_parser.add_argument(
    #     '-vv', '--very-verbose', action='store_true',
    #     dest='very_verbose',
    #     help="Set flask debug mode")
    # sub_parser.add_argument(
    #     '-v', '--verbose', action='store_true',
    #     dest='verbose',
    #     help="Set flask debug mode")
    # sub_parser.set_defaults(verbose=False)
    # sub_parser.set_defaults(very_verbose=False)
    sub_parser.add_argument(
        '--config', dest='config',
        default='./configs/config.yaml', type=str,
        help="Set configuration file path: Default = './configs/config.yaml'")
    sub_parser.add_argument(
        '--data', dest='data',
        default='.adas-data', type=str,
        help="Set data directory path: Default = '.adas-data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='adas_search', type=str,
        help="Set output directory path: Default = '.adas-output'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='.adas-checkpoint', type=str,
        help="Set checkpoint directory path: Default = '.adas-checkpoint")
    sub_parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.set_defaults(resume=False)
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training")
    sub_parser.set_defaults(cpu=False)
    sub_parser.add_argument(
        '--resume-search', action='store_true',
        dest='resume_search',
        help="Flag: Resume searching from latest trial"
    )
    sub_parser.set_defaults(resume_search=False)
    sub_parser.add_argument(
        '--train-num', type=int,
        dest='train_num',
        help="Number of times to run full train"
    )
    sub_parser.set_defaults(train_num=-1)
    sub_parser.add_argument(
        '--post_fix', type=int,
        dest='post_fix',
        help="Added to end of results folder"
    )
    sub_parser.set_defaults(post_fix=None)



if __name__ == '__main__':
    parser = ArgumentParser(description = __doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)
    
    '''
    Set up save paths for trials
    '''
    assert args.post_fix is not None, 'POSTFIX SHOULD NOT BE NONE!'
    GLOBALS.POST_FIX = args.post_fix
    save_dir = os.path.join(output_path, f"model={GLOBALS.CONFIG['model']}_dataset={GLOBALS.CONFIG['dataset']}_initEpoch={GLOBALS.CONFIG['initial_epochs']}_epochpert={GLOBALS.CONFIG['epochs_per_trial']}_searching={GLOBALS.CONFIG['optimization_algorithm']}_alpha={GLOBALS.CONFIG['ALPHA']}_epoch_sel={GLOBALS.CONFIG['epoch_selection']}_metric={GLOBALS.CONFIG['sel_metric']}_adaptnum={GLOBALS.CONFIG['adapt_trials']}_gamma={GLOBALS.CONFIG['gamma']}_{GLOBALS.POST_FIX}")
    GLOBALS.OUTPUT_PATH_STRING = str(save_dir)
    # New Checkpoint Path
    GLOBALS.CHECKPOINT_PATH = str(save_dir)

    

    if not os.path.exists(GLOBALS.OUTPUT_PATH_STRING):
        os.mkdir(GLOBALS.OUTPUT_PATH_STRING)

    output_path_string_trials = os.path.join(GLOBALS.OUTPUT_PATH_STRING, 'Trials')
    output_path_string_modelweights = os.path.join(GLOBALS.OUTPUT_PATH_STRING,'model_weights')
    output_path_string_graph_files = os.path.join(GLOBALS.OUTPUT_PATH_STRING,'graph_files')
    output_path_string_full_train = os.path.join(GLOBALS.OUTPUT_PATH_STRING,'full_train')
    if not os.path.exists(output_path_string_trials):
        os.mkdir(output_path_string_trials)

    if not os.path.exists(output_path_string_modelweights):
        os.mkdir(output_path_string_modelweights)

    if not os.path.exists(output_path_string_graph_files):
        os.mkdir(output_path_string_graph_files)

    if not os.path.exists(output_path_string_full_train):
        os.mkdir(output_path_string_full_train)

    '''
    Initialize Dataset
    '''
    if GLOBALS.CONFIG['dataset'] == 'CIFAR10':
        class_num = 10
    elif GLOBALS.CONFIG['dataset'] == 'CIFAR100':
        class_num = 100
    
    
    '''
    Initialize model
    '''
    # args_for_model = getArgsForModel(GLOBALS.CONFIG['model'])
    

    new_kernel_sizes = [[3]*7,
                                [3]*8,
                                [3]*12,
                                [3]*6]


    channel_sizes = [64, 128, 256, 512]

    new_channel_sizes = [[channel_sizes[0]]*7,
                                [channel_sizes[1]]*8,
                                [channel_sizes[2]]*12,
                                [channel_sizes[3]]*6]


    size_targets = [5000000, 10000000, 15000000] 
    size_diff = [sys.maxsize]*3
    size_actual = [sys.maxsize]*3
    sizes_to_use = [[64, 128, 256, 512], [64, 128, 256, 512], [64, 128, 256, 512]]



    model = DASNet34(num_classes_input=class_num,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
  
    """
    idx = 0
    for name, param in model.named_parameters():
        print(idx, name, param.shape)
        idx += 1
    """

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)


    while(total_params > 1000000):
        channel_sizes = [channel_sizes[0]-1, channel_sizes[1]-2, channel_sizes[2]-4, channel_sizes[3]-8]
        new_channel_sizes = [[channel_sizes[0]]*7,
                                [channel_sizes[1]]*8,
                                [channel_sizes[2]]*12,
                                [channel_sizes[3]]*6]
        model = DASNet34(num_classes_input=class_num,new_output_sizes=new_channel_sizes,new_kernel_sizes=new_kernel_sizes)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(total_params) 

        for idx, size in enumerate(size_targets):
            diff = abs(size - total_params)
            #closer size
            if(diff < size_diff[idx]):
                size_diff[idx] = diff 
                size_actual[idx] = total_params
                sizes_to_use[idx] = channel_sizes

    print(size_actual)
    print(sizes_to_use)





    """
    """

    # adjList, convIdx = ResNetAdj(model)
    # print(adjList)
    
    
    """
    Initialize Dataframes
    
    #conv_data, kernel_data, rank_final_data, rank_stable_data, delta_info, delta_info_kernel, conv_size_list, kernel_size_list,conv_layer_indicies = initialize_dataframes_and_lists(model,config = GLOBALS.CONFIG)

    '''
    Initialize Training
    '''
    # Initialize Learning Rate Scheduler and Optimizer
    #train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, model,global_config = GLOBALS)

    '''
    Make Model into linked-list format
    '''
    
    '''
    Convert into Dependency List
    '''
    
    # Get the model index to conv_layer_list_index mapping
    
    # model_to_list_mapping = {}
    # for list_index, model_index in enumerate(conv_layer_indicies):
    #     model_to_list_mapping[model_index] = list_index
    
    # print(model_to_list_mapping)


    '''
    Run Training for the First Trial
    '''
    # interrupted_trial = 0
    # if not args.resume_search:
    #     run_epochs(0, model, range(0,GLOBALS.CONFIG['initial_epochs']), train_loader, test_loader, device, optimizer, scheduler, output_path_string_trials,global_config = GLOBALS)
    # else:
    #     interrupted_trial = get_latest_completed_trial(output_path_string_trials)

    # # del model
    # model.to('cpu')
    # del train_loader
    # del test_loader
    # del optimizer
    # del scheduler

    # free_cuda_memory()

    # print('~~~First run_epochs done.~~~')

    # if (GLOBALS.CONFIG['kernel_adapt'] == 0):
    #     GLOBALS.CONFIG['adapt_trials_kernel'] = 0

    # GLOBALS.total_trials = GLOBALS.CONFIG['adapt_trials'] + GLOBALS.CONFIG['adapt_trials_kernel']

    # INITIAL_TEMPERATURE = 1000
    # TEMP = INITIAL_TEMPERATURE
    # BEST_METRIC = 0
    # BEST_SIZE = [] 

    # def CombineMetric(dep_metrics):
    #     overall_average = []
    #     for dependency_list in dep_metrics:
    #         overall_average.append(np.average(np.array(dependency_list)))
    #     return np.average(np.array(overall_average))

    # for i in range(1, GLOBALS.total_trials):

        
    #     if (GLOBALS.CONFIG['optimization_algorithm'] == 'SA'):
    #         print("--~~[SA Trial]~~--")
    #         new_channel_sizes, avg_mM, TEMP = getSearchingAlgorithm(GLOBALS.CONFIG['optimization_algorithm'],dependency_list = dependency_list, conv_size_list = conv_size_list, gamma = GLOBALS.CONFIG['gamma'],delta_scale =  GLOBALS.CONFIG['delta_scale'], sel_metric = GLOBALS.CONFIG['sel_metric'], \
    #                                                           initial_step_size = GLOBALS.CONFIG['init_step_size'], min_conv_size =  GLOBALS.CONFIG['min_conv_size'], max_conv_size = GLOBALS.CONFIG['max_conv_size'], trial_dir = output_path_string_trials, cur_trial = i - 1, temp= TEMP, totalTrials = GLOBALS.total_trials,mapping = model_to_list_mapping,global_config = GLOBALS)

    #     else:
    #         new_channel_sizes, avg_mM = getSearchingAlgorithm(GLOBALS.CONFIG['optimization_algorithm'],dependency_list = dependency_list,conv_size_list = conv_size_list, gamma = GLOBALS.CONFIG['gamma'],delta_scale =  GLOBALS.CONFIG['delta_scale'], sel_metric = GLOBALS.CONFIG['sel_metric'], \
    #                                                           initial_step_size = GLOBALS.CONFIG['init_step_size'], min_conv_size =  GLOBALS.CONFIG['min_conv_size'], max_conv_size = GLOBALS.CONFIG['max_conv_size'], trial_dir = output_path_string_trials, cur_trial = i - 1, mapping = model_to_list_mapping,global_config = GLOBALS)
    #     CM = CombineMetric(avg_mM)
    #     print("CM CALC:::    " + str(CM) + "  METRIC CALC:::    " + str(BEST_METRIC))
    #     if(CM > BEST_METRIC):
    #         print("CM Overtake:::   " + str(CM)+ "  METRIC CALC:::    " + str(BEST_METRIC))
    #         BEST_METRIC = CM
    #         BEST_SIZE = GLOBALS.old_conv_size.copy()
    #         # BEST_SIZE = new_channel_sizes.copy()


    #     conv_size_list = copy.deepcopy(new_channel_sizes)
    #     print(conv_size_list)

    #     print('~~~Writing to Dataframe~~~')
    #     if GLOBALS.CONFIG['parameter_type'] == 'channel':
    #         conv_data.loc[i] = new_channel_sizes
    #         delta_info.loc[i] = [GLOBALS.in_momentum_m, GLOBALS.out_momentum_m, avg_mM]

    #     print('~~~Starting Conv parameter_typements~~~')
    #     new_network = update_network(GLOBALS.CONFIG['model'],new_channel_sizes_list = new_channel_sizes, new_kernel_sizes = None, global_config = GLOBALS, class_num = class_num)
    #     print(GLOBALS.CONFIG['wt_type'])
    #     weight_transfer(model, new_network, GLOBALS.CONFIG['wt_type'],"module.")
    #     del model

    #     print('~~~Initializing the new model~~~')

    #     train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, new_network,
    #                                                                                 new_threshold_kernel=None, init_lr = GLOBALS.CONFIG['init_lr'],global_config = GLOBALS)
    #     print("Channel Sizes", new_channel_sizes)
    #     # print("Trial: ", i, " LR ", cur_LR)
    #     #print("Memory allocated before trial:", torch.cuda.memory_allocated(0))
    #     epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])

    #     if i < interrupted_trial:
    #         print('~~~Using previous training data~~~')

    #     else:
    #         print('~~~Training with new model~~~')
    #         run_epochs(i, model, epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_string_trials,global_config = GLOBALS)
    #     # print("Memory allocated after trial:", torch.cuda.memory_allocated(0))
    
    # Save the Trial Data
    # kernel_data, conv_data, rank_final_data, rank_stable_data, new_channel_sizes, delta_info, delta_info_kernel
    
    # Save the final search results
    # create_trial_data_file(kernel_data, conv_data, delta_info_kernel, delta_info, rank_final_data, rank_stable_data,
    #                        output_path_string_trials, output_path_string_graph_files, output_path_string_modelweights)
    
    '''
    Run a fresh full train
    '''
    # del model
    # Reinitialize Model
    """
    """
    model = update_network(GLOBALS.CONFIG['model'],new_channel_sizes_list = None, new_kernel_sizes = None, global_config = GLOBALS,class_num = class_num, predefined = True)
    try:
        train_loader,test_loader,device,optimizer,scheduler, model = initialize(args,model,beta=GLOBALS.CONFIG['beta_full'],
                                                                                scheduler=GLOBALS.CONFIG['full_train_scheduler'], init_lr=GLOBALS.CONFIG['init_lr_full'],global_config = GLOBALS)
    except:
        train_loader,test_loader,device,optimizer,scheduler, model = initialize(args,model,beta=GLOBALS.CONFIG['beta_full'],
                                                                                scheduler='StepLR', init_lr=GLOBALS.CONFIG['init_lr_full'],global_config = GLOBALS)
    # train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, model, global_config = GLOBALS)
    full_train_epochs = range(0, GLOBALS.CONFIG['max_epoch'])

    # GLOBALS.FULL_TRAIN = True
    GLOBALS.PERFORMANCE_STATISTICS = {}
    # GLOBALS.FULL_TRAIN_MODE = 'fresh'
    GLOBALS.EXCEL_PATH = ''

    run_epochs(0, model, full_train_epochs, train_loader, test_loader, device, optimizer, scheduler, output_path_string_full_train,global_config = GLOBALS)
    # model = run_fresh_full_train(output_sizes,kernel_sizes,full_train_epochs,output_path_fulltrain)

    output_excel = f"{GLOBALS.CONFIG['lr_scheduler']}_adapt_trial=0_" +\
                f"net={GLOBALS.CONFIG['network']}_" +\
                f"{GLOBALS.CONFIG['init_lr']}_dataset=" +\
                f"{GLOBALS.CONFIG['dataset']}.xlsx"

    # create_train_output_file(model,os.path.join(output_path_string_full_train,f"StepLR_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx"),
    #                                 output_path_string_full_train)
    performance_data = create_train_output_file(model,os.path.join(output_path_string_full_train,output_excel),
                                     output_path_string_full_train)

    print('-'*40)
    print('Full Train Completed')
    print('Gmac: ', performance_data['Gmac'])
    print('parameter count (M): ', performance_data['parameter count (M)'])
    print('GFlop: ', performance_data['GFlop'])
    print(GLOBALS.best_result)
    # for i in range (1, GLOBALS.CONFIG['train_num']):

    #     # Why is there 2 variables???
    #     output_path_fulltrain = output_path / "full_train_{}".format(i)
    #     output_path_string_full_train = GLOBALS.OUTPUT_PATH_STRING + slash + "full_train_{}".format(i)
    #     if not os.path.exists(output_path_string_full_train):
    #         os.mkdir(output_path_string_full_train)

    #     model = run_fresh_full_train(output_sizes, kernel_sizes, full_train_epochs, output_path_fulltrain)

    #     create_train_output_file(model,
    #                                 output_path_string_full_train + slash + f"StepLR_fresh_fulltrain_trial=0_net={GLOBALS.CONFIG['network']}_dataset={GLOBALS.CONFIG['dataset']}.xlsx",
    #                                 output_path_string_full_train)

    # '''
    # Begin Training Trial #1
    # - Update excel metrics
    # '''
    # '''
    # Begin Training Trial #1+
    # - Scale by default scaling factor for #2
    # - For trial #3, begin channel size scaling
    # - Store metrics after scaling
    # - Weight Transfer
    # - Train another trial
    # '''
    # epochs = range(0, GLOBALS.CONFIG['epochs_per_trial'])
    # full_train_epochs = range(0, GLOBALS.CONFIG['max_epoch'])

    # if GLOBALS.CONFIG['full_train_only']==False:
    #     print('Starting Trials')
    #     kernel_data,conv_data,rank_final_data,rank_stable_data,output_sizes,kernel_sizes,delta_info,delta_info_kernel=run_trials(epochs,output_path_string_trials)
    #     create_trial_data_file(kernel_data,conv_data,delta_info_kernel,delta_info,rank_final_data,rank_stable_data,output_path_string_trials,output_path_string_graph_files,output_path_string_modelweights)
    #     print('Done Trials.')
    # else:
    #     k = 3
    #     output_sizes=[GLOBALS.super1_idx, GLOBALS.super2_idx, GLOBALS.super3_idx, GLOBALS.super4_idx]
    #     kernel_sizes = [GLOBALS.super1_kernel_idx,GLOBALS.super2_kernel_idx,GLOBALS.super3_kernel_idx,GLOBALS.super4_kernel_idx]
    """
    