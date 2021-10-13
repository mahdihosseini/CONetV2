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
from random import randint
import configs.global_vars as GLOBALS
from utils.train_helpers import (build_paths,create_trial_data_file,free_cuda_memory,create_train_output_file)
from utils.create_dataframe import initialize_dataframes_and_lists
from models import (get_model,update_network)
from utils.train import (initialize,run_epochs)
from searching_algorithm import getSearchingAlgorithm
from utils.weight_transfer import weight_transfer
from dependency import DependencyList
import numpy as np
from dartsadj import DartsAdj
import torchvision.models as models

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

    sub_parser.add_argument(
        '--model', type=str,
        dest='model',
        help="model type"
    )
    sub_parser.set_defaults(model=None)

    sub_parser.add_argument(
        '--network', type=str,
        dest='network',
        help="network subclass of model"
    )
    sub_parser.set_defaults(network=None)

    sub_parser.add_argument(
        '--gamma', type=float,
        dest='gamma',
        help="momentum tuning factor"
    )
    sub_parser.set_defaults(gamma=None)

    sub_parser.add_argument(
        '--optimization_algorithm', type=str,
        dest='optimization_algorithm',
        help="Searching Algorithm"
    )
    sub_parser.set_defaults(optimization_algorithm=None)

    sub_parser.add_argument(
        '--full_train_scheduler', type=str,
        dest='full_train_scheduler',
        help="scheduler for full train"
    )
    sub_parser.set_defaults(full_train_scheduler=None)


    



if __name__ == '__main__':
    parser = ArgumentParser(description = __doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)
    
    '''
    Set up save paths for trials
    '''
    assert args.post_fix is not None, 'POSTFIX SHOULD NOT BE NONE!'
    GLOBALS.CONFIG['post_fix'] = args.post_fix

    # Overwrite default values if present
    if args.model:
        GLOBALS.CONFIG['model'] = args.model

    if args.network:
        GLOBALS.CONFIG['network'] = args.network

    if args.gamma:
        GLOBALS.CONFIG['gamma'] = args.gamma

    if args.optimization_algorithm:
        GLOBALS.CONFIG['optimization_algorithm'] = args.optimization_algorithm
    
    if args.full_train_scheduler:
        GLOBALS.CONFIG['full_train_scheduler'] = args.full_train_scheduler
    

    save_dir = os.path.join(output_path, f"model={GLOBALS.CONFIG['model']}_dataset={GLOBALS.CONFIG['dataset']}_initEpoch={GLOBALS.CONFIG['initial_epochs']}_epochpert={GLOBALS.CONFIG['epochs_per_trial']}_searching={GLOBALS.CONFIG['optimization_algorithm']}_alpha={GLOBALS.CONFIG['ALPHA']}_epoch_sel={GLOBALS.CONFIG['epoch_selection']}_metric={GLOBALS.CONFIG['sel_metric']}_adaptnum={GLOBALS.CONFIG['adapt_trials']}_gamma={GLOBALS.CONFIG['gamma']}_{GLOBALS.CONFIG['post_fix']}")
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
    model = get_model(GLOBALS.CONFIG['model'],num_classes_input = class_num, global_config = GLOBALS)
    
    print(model)

    idx = 0
    for name, param in model.named_parameters():
        print(idx, name, param.shape)
        idx += 1

    #DartsAdj(model)

    """

    # adjList, convIdx = ResNetAdj(model)
    # print(adjList)
    """
    
    
    # Initialize Dataframes

    conv_data, kernel_data, rank_final_data, rank_stable_data, delta_info, delta_info_kernel, conv_size_list, kernel_size_list,conv_layer_indicies = initialize_dataframes_and_lists(model,config = GLOBALS.CONFIG)

    '''
    Initialize Training
    '''
    # Initialize Learning Rate Scheduler and Optimizer
    # train_loader, test_loader, device, optimizer, scheduler, model = initialize(args, model,global_config = GLOBALS)

    '''
    Make Model into linked-list format
    '''
    
    '''
    Convert into Dependency List
    '''
    # dependency_list_hardcode = [[[0,'out'],[3,'in'],[6,'out'],[9,'in'],
    #                     [12,'out'],[15,'in'],[18,'out'],[21,'in'],[27,'in']],
    #                     [[3,'out'],[6,'in']],[[9,'out'],[12,'in']],[[15,'out'],[18,'in']],
    #                     [[21,'out'],[24,'in']],[[27,'out'],[24,'out'],[30,'in'],[33,'out'],[36,'in'],
    #                     [39,'out'],[42,'in'],[45,'out'],[48,'in'],[54,'in']],[[30,'out'],[33,'in']],
    #                     [[36,'out'],[39,'in']],[[42,'out'],[45,'in']],[[48,'out'],[51,'in']],
    #                     [[54,'out'],[51,'out'],[57,'in'],[60,'out'],[63,'in'],
    #                     [66,'out'],[69,'in'],[72,'out'],[75,'in'],[78,'out'],[81,'in'],[84,'out'],
    #                     [87,'in'],[93,'in']],
    #                     [[57,'out'],[60,'in']],[[63,'out'],[66,'in']],[[69,'out'],[72,'in']],
    #                     [[75,'out'],[78,'in']],[[81,'out'],[84,'in']],[[87,'out'],[90,'in']],
    #                     [[93,'out'],[90,'out'],[96,'in'],[99,'out'],[102,'in'],
    #                     [105,'out']],[[96,'out'],[99,'in']],[[102,'out'],[105,'in']]
    #                     ]
    # # print(model)
    print('*'*20)
    model_to_list_mapping = {}
    for list_index, model_index in enumerate(conv_layer_indicies):

        
        model_to_list_mapping[model_index] = list_index
    
    print(model_to_list_mapping)


    dependency_list = DependencyList(GLOBALS.CONFIG['model'],model)
    print('*'*200, len(dependency_list))

    # for parameter in model.parameters():
    # model = models.resnet34()
    
  
    
    # print(dependency_list)
    # for i,dep in enumerate(dependency_list):
    #     print(len(dep), len(dependency_list_hardcode[i]))
    # print(len(dependency_list[0]))
    
    # for l,p in zip(model.named_modules(), model.named_parameters()):
    #     # a, layer = l
    #     # name,param = p
    #     # print(name)
    # print('*'*40, dependency_list)
    # break
    
    # [[{'Conv2d_0_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75aff38350>},
    #  {'Conv2d_3_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5f850>}, 
    #  {'Conv2d_9_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5f950>}, 
    #  {'Conv2d_6_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5f750>}, 
    #  {'Conv2d_15_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5fbd0>}, 
    #  {'Conv2d_12_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5fd10>}, 
    #  {'Conv2d_21_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5fe10>}, 
    #  {'Conv2d_18_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5ff10>}, 
    #  {'Conv2d_27_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43090>}, 
    #  {'Conv2d_24_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43150>}, 
    #  {'Conv2d_33_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43290>}, 
    #  {'Conv2d_30_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43390>}, 
    #  {'Conv2d_39_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b434d0>}, 
    #  {'Conv2d_36_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b435d0>}, 
    #  {'Conv2d_45_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43710>}, 
    #  {'Conv2d_42_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43810>}, 
    #  {'Conv2d_51_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43950>}, 
    #  {'Conv2d_48_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43a50>}, 
    #  {'Conv2d_57_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43b90>}, 
    #  {'Conv2d_54_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43c90>}, 
    #  {'Conv2d_63_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43dd0>}, 
    #  {'Conv2d_60_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43ed0>}, 
    #  {'Conv2d_69_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49050>}, 
    #  {'Conv2d_66_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49150>}, 
    #  {'Conv2d_75_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49290>}, 
    #  {'Conv2d_72_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49390>}, 
    #  {'Conv2d_81_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b494d0>}, 
    #  {'Conv2d_78_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b495d0>}, 
    #  {'Conv2d_87_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49710>}, 
    #  {'Conv2d_84_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49810>}, 
    #  {'Conv2d_93_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49950>}, 
    #  {'Conv2d_90_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49a50>}, 
    #  {'Conv2d_99_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49b90>}, 
    #  {'Conv2d_96_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49c90>}], 
    #  [{'Conv2d_3_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5f850>}, 
    #  {'Conv2d_6_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5f750>}], 
    #  [{'Conv2d_9_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5f950>}, 
    #  {'Conv2d_12_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5fd10>}], 
    #  [{'Conv2d_15_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5fbd0>}, {'Conv2d_18_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5ff10>}], [{'Conv2d_21_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b5fe10>}, {'Conv2d_24_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43150>}], [{'Conv2d_27_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43090>}, {'Conv2d_30_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43390>}], [{'Conv2d_33_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43290>}, {'Conv2d_36_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b435d0>}], [{'Conv2d_39_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b434d0>}, {'Conv2d_42_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43810>}], [{'Conv2d_45_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43710>}, {'Conv2d_48_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43a50>}], [{'Conv2d_51_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43950>}, {'Conv2d_54_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43c90>}], [{'Conv2d_57_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43b90>}, {'Conv2d_60_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43ed0>}], [{'Conv2d_63_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b43dd0>}, {'Conv2d_66_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49150>}], [{'Conv2d_69_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49050>}, {'Conv2d_72_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49390>}], [{'Conv2d_75_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49290>}, {'Conv2d_78_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b495d0>}], [{'Conv2d_81_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b494d0>}, {'Conv2d_84_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49810>}], [{'Conv2d_87_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49710>}, {'Conv2d_90_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49a50>}], [{'Conv2d_93_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49950>}, {'Conv2d_96_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49c90>}], [{'Conv2d_99_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49b90>}, {'Conv2d_102_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49dd0>}], [{'Conv2d_102_out': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49dd0>}, {'Conv2d_105_in': <dependency.LinkedListConstructor.LinkedList.<locals>.Layer object at 0x7f75a7b49ed0>}]]
    #odict_keys(['module.conv1.weight', 'module.bn1.weight', 'module.bn1.bias', 'module.bn1.running_mean', 'module.bn1.running_var', 'module.bn1.num_batches_tracked', 'module.network.0.conv1.weight', 'module.network.0.bn1.weight', 'module.network.0.bn1.bias', 'module.network.0.bn1.running_mean', 'module.network.0.bn1.running_var', 'module.network.0.bn1.num_batches_tracked', 'module.network.0.conv2.weight', 'module.network.0.bn2.weight', 'module.network.0.bn2.bias', 'module.network.0.bn2.running_mean', 'module.network.0.bn2.running_var', 'module.network.0.bn2.num_batches_tracked', 'module.network.1.conv1.weight', 'module.network.1.bn1.weight', 'module.network.1.bn1.bias', 'module.network.1.bn1.running_mean', 'module.network.1.bn1.running_var', 'module.network.1.bn1.num_batches_tracked', 'module.network.1.conv2.weight', 'module.network.1.bn2.weight', 'module.network.1.bn2.bias', 'module.network.1.bn2.running_mean', 'module.network.1.bn2.running_var', 'module.network.1.bn2.num_batches_tracked', 'module.network.2.conv1.weight', 'module.network.2.bn1.weight', 'module.network.2.bn1.bias', 'module.network.2.bn1.running_mean', 'module.network.2.bn1.running_var', 'module.network.2.bn1.num_batches_tracked', 'module.network.2.conv2.weight', 'module.network.2.bn2.weight', 'module.network.2.bn2.bias', 'module.network.2.bn2.running_mean', 'module.network.2.bn2.running_var', 'module.network.2.bn2.num_batches_tracked', 'module.network.3.conv1.weight', 'module.network.3.bn1.weight', 'module.network.3.bn1.bias', 'module.network.3.bn1.running_mean', 'module.network.3.bn1.running_var', 'module.network.3.bn1.num_batches_tracked', 'module.network.3.conv2.weight', 'module.network.3.bn2.weight', 'module.network.3.bn2.bias', 'module.network.3.bn2.running_mean', 'module.network.3.bn2.running_var', 'module.network.3.bn2.num_batches_tracked', 'module.network.3.shortcut.0.weight', 'module.network.3.shortcut.1.weight', 'module.network.3.shortcut.1.bias', 'module.network.3.shortcut.1.running_mean', 'module.network.3.shortcut.1.running_var', 'module.network.3.shortcut.1.num_batches_tracked', 'module.network.4.conv1.weight', 'module.network.4.bn1.weight', 'module.network.4.bn1.bias', 'module.network.4.bn1.running_mean', 'module.network.4.bn1.running_var', 'module.network.4.bn1.num_batches_tracked', 'module.network.4.conv2.weight', 'module.network.4.bn2.weight', 'module.network.4.bn2.bias', 'module.network.4.bn2.running_mean', 'module.network.4.bn2.running_var', 'module.network.4.bn2.num_batches_tracked', 'module.network.5.conv1.weight', 'module.network.5.bn1.weight', 'module.network.5.bn1.bias', 'module.network.5.bn1.running_mean', 'module.network.5.bn1.running_var', 'module.network.5.bn1.num_batches_tracked', 'module.network.5.conv2.weight', 'module.network.5.bn2.weight', 'module.network.5.bn2.bias', 'module.network.5.bn2.running_mean', 'module.network.5.bn2.running_var', 'module.network.5.bn2.num_batches_tracked', 'module.network.6.conv1.weight', 'module.network.6.bn1.weight', 'module.network.6.bn1.bias', 'module.network.6.bn1.running_mean', 'module.network.6.bn1.running_var', 'module.network.6.bn1.num_batches_tracked', 'module.network.6.conv2.weight', 'module.network.6.bn2.weight', 'module.network.6.bn2.bias', 'module.network.6.bn2.running_mean', 'module.network.6.bn2.running_var', 'module.network.6.bn2.num_batches_tracked', 'module.network.7.conv1.weight', 'module.network.7.bn1.weight', 'module.network.7.bn1.bias', 'module.network.7.bn1.running_mean', 'module.network.7.bn1.running_var', 'module.network.7.bn1.num_batches_tracked', 'module.network.7.conv2.weight', 'module.network.7.bn2.weight', 'module.network.7.bn2.bias', 'module.network.7.bn2.running_mean', 'module.network.7.bn2.running_var', 'module.network.7.bn2.num_batches_tracked', 'module.network.7.shortcut.0.weight', 'module.network.7.shortcut.1.weight', 'module.network.7.shortcut.1.bias', 'module.network.7.shortcut.1.running_mean', 'module.network.7.shortcut.1.running_var', 'module.network.7.shortcut.1.num_batches_tracked', 'module.network.8.conv1.weight', 'module.network.8.bn1.weight', 'module.network.8.bn1.bias', 'module.network.8.bn1.running_mean', 'module.network.8.bn1.running_var', 'module.network.8.bn1.num_batches_tracked', 'module.network.8.conv2.weight', 'module.network.8.bn2.weight', 'module.network.8.bn2.bias', 'module.network.8.bn2.running_mean', 'module.network.8.bn2.running_var', 'module.network.8.bn2.num_batches_tracked', 'module.network.9.conv1.weight', 'module.network.9.bn1.weight', 'module.network.9.bn1.bias', 'module.network.9.bn1.running_mean', 'module.network.9.bn1.running_var', 'module.network.9.bn1.num_batches_tracked', 'module.network.9.conv2.weight', 'module.network.9.bn2.weight', 'module.network.9.bn2.bias', 'module.network.9.bn2.running_mean', 'module.network.9.bn2.running_var', 'module.network.9.bn2.num_batches_tracked', 'module.network.10.conv1.weight', 'module.network.10.bn1.weight', 'module.network.10.bn1.bias', 'module.network.10.bn1.running_mean', 'module.network.10.bn1.running_var', 'module.network.10.bn1.num_batches_tracked', 'module.network.10.conv2.weight', 'module.network.10.bn2.weight', 'module.network.10.bn2.bias', 'module.network.10.bn2.running_mean', 'module.network.10.bn2.running_var', 'module.network.10.bn2.num_batches_tracked', 'module.network.11.conv1.weight', 'module.network.11.bn1.weight', 'module.network.11.bn1.bias', 'module.network.11.bn1.running_mean', 'module.network.11.bn1.running_var', 'module.network.11.bn1.num_batches_tracked', 'module.network.11.conv2.weight', 'module.network.11.bn2.weight', 'module.network.11.bn2.bias', 'module.network.11.bn2.running_mean', 'module.network.11.bn2.running_var', 'module.network.11.bn2.num_batches_tracked', 'module.network.12.conv1.weight', 'module.network.12.bn1.weight', 'module.network.12.bn1.bias', 'module.network.12.bn1.running_mean', 'module.network.12.bn1.running_var', 'module.network.12.bn1.num_batches_tracked', 'module.network.12.conv2.weight', 'module.network.12.bn2.weight', 'module.network.12.bn2.bias', 'module.network.12.bn2.running_mean', 'module.network.12.bn2.running_var', 'module.network.12.bn2.num_batches_tracked', 'module.network.13.conv1.weight', 'module.network.13.bn1.weight', 'module.network.13.bn1.bias', 'module.network.13.bn1.running_mean', 'module.network.13.bn1.running_var', 'module.network.13.bn1.num_batches_tracked', 'module.network.13.conv2.weight', 'module.network.13.bn2.weight', 'module.network.13.bn2.bias', 'module.network.13.bn2.running_mean', 'module.network.13.bn2.running_var', 'module.network.13.bn2.num_batches_tracked', 'module.network.13.shortcut.0.weight', 'module.network.13.shortcut.1.weight', 'module.network.13.shortcut.1.bias', 'module.network.13.shortcut.1.running_mean', 'module.network.13.shortcut.1.running_var', 'module.network.13.shortcut.1.num_batches_tracked', 'module.network.14.conv1.weight', 'module.network.14.bn1.weight', 'module.network.14.bn1.bias', 'module.network.14.bn1.running_mean', 'module.network.14.bn1.running_var', 'module.network.14.bn1.num_batches_tracked', 'module.network.14.conv2.weight', 'module.network.14.bn2.weight', 'module.network.14.bn2.bias', 'module.network.14.bn2.running_mean', 'module.network.14.bn2.running_var', 'module.network.14.bn2.num_batches_tracked', 'module.network.15.conv1.weight', 'module.network.15.bn1.weight', 'module.network.15.bn1.bias', 'module.network.15.bn1.running_mean', 'module.network.15.bn1.running_var', 'module.network.15.bn1.num_batches_tracked', 'module.network.15.conv2.weight', 'module.network.15.bn2.weight', 'module.network.15.bn2.bias', 'module.network.15.bn2.running_mean', 'module.network.15.bn2.running_var', 'module.network.15.bn2.num_batches_tracked', 'module.linear.weight', 'module.linear.bias'])
    # dependency_list = [[[0,'out'],[3,'in'],[6,'out'],[9,'in'],
    #                     [12,'out'],[15,'in'],[18,'out'],[21,'in'],[27,'in']],
    #                     [[3,'out'],[6,'in']],[[9,'out'],[12,'in']],[[15,'out'],[18,'in']],
    #                     [[21,'out'],[24,'in']],[[27,'out'],[24,'out'],[30,'in'],[33,'out'],[36,'in'],
    #                     [39,'out'],[42,'in'],[45,'out'],[48,'in'],[54,'in']],[[30,'out'],[33,'in']],
    #                     [[36,'out'],[39,'in']],[[42,'out'],[45,'in']],[[48,'out'],[51,'in']],
    #                     [[54,'out'],[51,'out'],[57,'in'],[60,'out'],[63,'in'],
    #                     [66,'out'],[69,'in'],[72,'out'],[75,'in'],[78,'out'],[81,'in'],[84,'out'],
    #                     [87,'in'],[93,'in']],
    #                     [[57,'out'],[60,'in']],[[63,'out'],[66,'in']],[[69,'out'],[72,'in']],
    #                     [[75,'out'],[78,'in']],[[81,'out'],[84,'in']],[[87,'out'],[90,'in']],
    #                     [[93,'out'],[90,'out'],[96,'in'],[99,'out'],[102,'in'],
    #                     [105,'out']],[[96,'out'],[99,'in']],[[102,'out'],[105,'in']]
    #                     ]
    
    # Get the model index to conv_layer_list_index mapping
    
    
    

    '''
    Run Training for the First Trial
    '''
    # # interrupted_trial = 0
    # # if not args.resume_search:
    # #     run_epochs(0, model, range(0,GLOBALS.CONFIG['initial_epochs']), train_loader, test_loader, device, optimizer, scheduler, output_path_string_trials,global_config = GLOBALS)
    # # else:
    # #     interrupted_trial = get_latest_completed_trial(output_path_string_trials)

    # # del model
    # model.to('cpu')
    # del train_loader
    # del test_loader
    # del optimizer
    # del scheduler

    # free_cuda_memory()

    print('~~~First run_epochs done.~~~')

    if (GLOBALS.CONFIG['kernel_adapt'] == 0):
        GLOBALS.CONFIG['adapt_trials_kernel'] = 0

    GLOBALS.total_trials = GLOBALS.CONFIG['adapt_trials'] + GLOBALS.CONFIG['adapt_trials_kernel']

    INITIAL_TEMPERATURE = 1000
    TEMP = INITIAL_TEMPERATURE
    BEST_METRIC = 0
    BEST_SIZE = [] 

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
    
    # # Save the Trial Data
    # # kernel_data, conv_data, rank_final_data, rank_stable_data, new_channel_sizes, delta_info, delta_info_kernel
    
    # # Save the final search results
    # create_trial_data_file(kernel_data, conv_data, delta_info_kernel, delta_info, rank_final_data, rank_stable_data,
    #                        output_path_string_trials, output_path_string_graph_files, output_path_string_modelweights)
    
    # '''
    # Run a fresh full train
    # '''
    # del model

    BEST_SIZE = 0
    # 1M - 4M
    # 6M - 10M
    # 12M - 15M
    # 16M - 20M
    rangeMin, rangeMax, i = 4000000, 5000000,  0
    counter = 0
    while ((i > rangeMax) or (i < rangeMin) or (i == 0)):
        conv_size_list = [0]*len(model_to_list_mapping)
        # print("STARTYUY#UY#UY#UY#")


        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)


        # Determine scaling direction for each dependency list
        for dep_index in range(len(dependency_list)):
            # Get the difference in metrics
            # metric_difference = getCumulativeMetricPerDependency(dep_metric_new) - getCumulativeMetricPerDependency(conv_size_avg_mM_old[dep_index])
            
            # Get the current dimension for the dependency list
            # try:
            #     # print(list(dependency_list[dep_index][0].keys())[0].split('_')[-2])
            #     cur_dim = conv_size_list[mapping[int(list(dependency_list[dep_index][0].keys())[0].split('_')[-2])]]
            # except:
            #     cur_dim = conv_size_list[mapping[dependency_list[dep_index][0][0]]]

            # Generate new dimension based on metric_difference
            # new_dim = getScalingMethod(method_name = global_config.CONFIG['scaling_method'],metric_type = global_config.CONFIG['sel_metric'], metric_difference = metric_difference, prev_channel_size = cur_dim)
            new_dim = randint(16,512)
            # Update all elements within dependency list with the new dimension
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
                conv_size_list[model_to_list_mapping[layer_index]] = new_dim
                #[0,3,...]

        # print("8495y439y53459034")
        new_network = update_network(GLOBALS.CONFIG['model'],new_channel_sizes_list = conv_size_list, new_kernel_sizes = None, global_config = GLOBALS, class_num = class_num)
        model = new_network
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)
        i = pytorch_total_params
        BEST_SIZE = conv_size_list
        counter+=1

    print(i)
    print('Num of Iters', counter)

    
    
    # Reinitialize Model
    model = update_network(GLOBALS.CONFIG['model'],new_channel_sizes_list = BEST_SIZE, new_kernel_sizes = None, global_config = GLOBALS,class_num = class_num)
    
    try:
        train_loader,test_loader,device,optimizer,scheduler, model = initialize(args,model,beta=GLOBALS.CONFIG['beta_full'],
                                                                                scheduler=GLOBALS.CONFIG['full_train_scheduler'], init_lr=GLOBALS.CONFIG['init_lr_full'],global_config = GLOBALS)
    except:
        train_loader,test_loader,device,optimizer,scheduler, model = initialize(args,model,beta=GLOBALS.CONFIG['beta_full'],
                                                                                scheduler='StepLR', init_lr=GLOBALS.CONFIG['init_lr_full'],global_config = GLOBALS)
    
    # train_loader,test_loader,device,optimizer,scheduler, model = initialize(args,new_network,beta=GLOBALS.CONFIG['beta_full'],
    #                                                                         scheduler='StepLR', init_lr=GLOBALS.CONFIG['init_lr_full'],global_config = GLOBALS)
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

    # Print best stats
    print('-'*40)
    print('Full Train Completed')
    print('Gmac: ', performance_data['Gmac'])
    print('parameter count (M): ', performance_data['parameter count (M)'])
    print('GFlop: ', performance_data['GFlop'])
    print(GLOBALS.best_result)
    # print('Best Epoch: ')
    # print('Test Accuracy: ')
    # print('Test Loss: ')
    # print('Train Accuracy: ')
    # print('Train Loss: ')



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
    
    