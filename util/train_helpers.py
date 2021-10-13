from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from typing import Tuple
from argparse import Namespace as APNamespace, _SubParsersAction,ArgumentParser
from pathlib import Path
import os
import platform
import time
import pandas as pd
import gc
import numpy as np
from typing import Dict, Union, List
import yaml
import torch
from ptflops import get_model_complexity_info
import configs.global_vars as GLOBALS
from utils.utils import (smart_string_to_float,smart_string_to_int)

# def get_args(sub_parser: _SubParsersAction):
#     # print("\n---------------------------------")
#     # print("AdaS Train Args")
#     # print("---------------------------------\n")
#     # sub_parser.add_argument(
#     #     '-vv', '--very-verbose', action='store_true',
#     #     dest='very_verbose',
#     #     help="Set flask debug mode")
#     # sub_parser.add_argument(
#     #     '-v', '--verbose', action='store_true',
#     #     dest='verbose',
#     #     help="Set flask debug mode")
#     # sub_parser.set_defaults(verbose=False)
#     # sub_parser.set_defaults(very_verbose=False)
#     sub_parser.add_argument(
#         '--config', dest='config',
#         default='config.yaml', type=str,
#         help="Set configuration file path: Default = 'config.yaml'")
#     sub_parser.add_argument(
#         '--data', dest='data',
#         default='.adas-data', type=str,
#         help="Set data directory path: Default = '.adas-data'")
#     sub_parser.add_argument(
#         '--output', dest='output',
#         default='adas_search', type=str,
#         help="Set output directory path: Default = '.adas-output'")
#     sub_parser.add_argument(
#         '--checkpoint', dest='checkpoint',
#         default='.adas-checkpoint', type=str,
#         help="Set checkpoint directory path: Default = '.adas-checkpoint")
#     sub_parser.add_argument(
#         '--root', dest='root',
#         default='.', type=str,
#         help="Set root path of project that parents all others: Default = '.'")
#     sub_parser.set_defaults(resume=False)
#     sub_parser.add_argument(
#         '--cpu', action='store_true',
#         dest='cpu',
#         help="Flag: CPU bound training")
#     sub_parser.set_defaults(cpu=False)
#     sub_parser.add_argument(
#         '--resume-search', action='store_true',
#         dest='resume_search',
#         help="Flag: Resume searching from latest trial"
#     )
#     sub_parser.set_defaults(resume_search=False)
#     sub_parser.add_argument(
#         '--train-num', type=int,
#         dest='train_num',
#         help="Number of times to run full train"
#     )
#     sub_parser.set_defaults(train_num=-1)
def create_train_output_file(new_network, full_fresh_file,output_path_string_full_train, debug=False):
    output_file='default.xlsx'
    performance_output_file = os.path.join(output_path_string_full_train, 'performance.xlsx')
    auxilary_output_file = os.path.join(output_path_string_full_train,'auxilary.xlsx')
    # if platform.system() == 'Windows':
    #     performance_output_file=output_path_string_full_train + '\\' + 'performance.xlsx'
    #     auxilary_output_file = output_path_string_full_train + '\\' + 'auxilary.xlsx'
    # else:
    #     performance_output_file = output_path_string_full_train +'/'+ 'performance.xlsx'
    #     auxilary_output_file = output_path_string_full_train + '/' + 'auxilary.xlsx'
    writer_performance = pd.ExcelWriter(performance_output_file, engine='openpyxl')
    wb_per = writer_performance.book
    writer_auxilary = pd.ExcelWriter(auxilary_output_file, engine='openpyxl')
    wb_aux = writer_auxilary.book

    full_fresh_dfs = pd.read_excel(full_fresh_file)
    final_epoch_fresh = full_fresh_dfs.columns[-1][(full_fresh_dfs.columns[-1].index('epoch_') + 6):]
    performance_data={}
    auxilary_data={}
    if debug==True:
        macs=0
        params=0
    else:
        macs, params = get_model_complexity_info(new_network, (3,32,32), as_strings=False,print_per_layer_stat=False) #verbose=True)
    performance_data['Gmac']=int(macs) / 1000000000
    performance_data['GFlop']=2 * int(macs) / 1000000000
    performance_data['parameter count (M)'] = int(params) / 1000000

    num_layer=len(full_fresh_dfs['train_acc_epoch_' + str(0)])
    layer_list=list(range(0,num_layer))
    auxilary_data['layer_index'] = layer_list

    for i in range(int(final_epoch_fresh)+1):
        performance_data['train_acc_epoch_' + str(i)+ " (%)"] = [full_fresh_dfs['train_acc_epoch_' + str(i)][0] *100]
        performance_data['train_loss_epoch_' + str(i)] = [full_fresh_dfs['train_loss_epoch_' + str(i)][0]]
        performance_data['test_acc_epoch_' + str(i)+" (%)"] = [full_fresh_dfs['test_acc_epoch_' + str(i)][0] *100]
        performance_data['test_loss_epoch_' + str(i)] = [full_fresh_dfs['test_loss_epoch_' + str(i)][0]]

        auxilary_data['in_KG_epcho'+str(i)] = full_fresh_dfs['in_S_epoch_' + str(i)]
        auxilary_data['out_KG_epcho'+str(i)] = full_fresh_dfs['out_S_epoch_' + str(i)]
        auxilary_data['in_rank_epcho'+str(i)] = full_fresh_dfs['in_rank_epoch_' + str(i)]
        auxilary_data['out_rank_epcho'+str(i)] = full_fresh_dfs['out_rank_epoch_' + str(i)]
        auxilary_data['in_condition_epcho'+str(i)] = full_fresh_dfs['in_condition_epoch_' + str(i)]
        auxilary_data['out_condition_epcho'+str(i)] = full_fresh_dfs['out_condition_epoch_' + str(i)]

    df_per = pd.DataFrame(performance_data)
    df_per.to_excel(writer_performance, index=False)
    wb_per.save(performance_output_file)

    df_aux = pd.DataFrame(auxilary_data)
    df_aux.to_excel(writer_auxilary, index=False)
    wb_aux.save(auxilary_output_file)

    # if platform.system == 'Windows':
    #     slash = '\\'
    # else:
    #     slash = '/'
    # output_path, _ = os.path.split(output_path_string_full_train)
    # copyfile(os.path.join(output_path, '..','..' + slash + '.adas-checkpoint' + slash + 'ckpt.pth',
    #          output_path_string_full_train + slash + 'ckpt.pth')

def build_paths(args: APNamespace):

    root_path = Path(args.root).expanduser()
    config_path = Path(args.config).expanduser()
    data_path = root_path / Path(args.data).expanduser()
    output_path = root_path / Path(args.output).expanduser()
    # global checkpoint_path, config
    GLOBALS.CHECKPOINT_PATH = root_path / Path(args.checkpoint).expanduser()
    #checks
    if not config_path.exists():
        # logging.critical(f"AdaS: Config path {config_path} does not exist")
        print(os.getcwd())
        print(config_path)
        raise ValueError(f"Config path {config_path} does not exist")
    if not data_path.exists():
        print(f"Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        print(f"Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not GLOBALS.CHECKPOINT_PATH.exists():
        if args.resume:
            raise ValueError(f"Cannot resume from checkpoint without " +
                             "specifying checkpoint dir")
        GLOBALS.CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)

    with config_path.open() as f:
        GLOBALS.CONFIG = parse_config(yaml.load(f))

    return output_path

def parse_config(
    config: Dict[str, Union[str, float, int]]) -> Dict[
        str, Union[str, float, int]]:
    valid_dataset = ['CIFAR10', 'CIFAR100']
    if config['dataset'] not in valid_dataset:
        raise ValueError(
            f"config.yaml: unknown dataset {config['dataset']}. " +
            f"Must be one of {valid_dataset}")
    valid_models = [
        'VGG16', 'ResNet34', 'PreActResNet18',
        'GoogLeNet', 'densenet_cifar', 'ResNeXt29_2x64d', 'MobileNet',
        'MobileNetV2', 'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2',
        'EfficientNetB0','DASNet34','DASNet50', 'DARTS', 'DARTSPlus']
    if config['network'] not in valid_models:
        raise ValueError(
            f"config.yaml: unknown model {config['network']}." +
            f"Must be one of {valid_models}")

    if config['lr_scheduler'] == 'AdaS' and config['optim_method'] != 'SGD':
        raise ValueError(
            'config.yaml: AdaS can only be used with SGD')


    config['beta'] = smart_string_to_float(
        config['beta'],
        e='config.yaml: beta must be a float')
    e = 'config.yaml: init_lr must be a float or list of floats'
    if not isinstance(config['init_lr'], str):
        if isinstance(config['init_lr'], list):
            for i, lr in enumerate(config['init_lr']):
                config['init_lr'][i] = smart_string_to_float(lr, e=e)
        else:
            config['init_lr'] = smart_string_to_float(config['init_lr'], e=e)
    else:
        if config['init_lr'] != 'auto':
            raise ValueError(e)
    config['max_epoch'] = smart_string_to_int(
        config['max_epoch'],
        e='config.yaml: max_epoch must be an int')
    config['early_stop_threshold'] = smart_string_to_float(
        config['early_stop_threshold'],
        e='config.yaml: early_stop_threshold must be a float')
    config['early_stop_patience'] = smart_string_to_int(
        config['early_stop_patience'],
        e='config.yaml: early_stop_patience must be an int')
    config['mini_batch_size'] = smart_string_to_int(
        config['mini_batch_size'],
        e='config.yaml: mini_batch_size must be an int')
    config['min_lr'] = smart_string_to_float(
        config['min_lr'],
        e='config.yaml: min_lr must be a float')
    config['zeta'] = smart_string_to_float(
        config['zeta'],
        e='config.yaml: zeta must be a float')
    config['p'] = smart_string_to_int(
        config['p'],
        e='config.yaml: p must be an int')
    '''
        NOT WORKING WITH NUM_WORKERS in this version of search code
        config['num_workers'] = smart_string_to_int(
        config['num_workers'],
        e='config.yaml: num_works must be an int')
    '''
    if config['loss'] != 'cross_entropy':
        raise ValueError('config.yaml: loss must be cross_entropy')
    return config

def free_cuda_memory():

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def create_trial_data_file(kernel_data, conv_data, delta_info_kernel, delta_info, rank_final_data, rank_stable_data,
                           output_path_string_trials, output_path_string_graph_files, output_path_string_modelweights):
    # parameter_data.to_excel(output_path_string_trials+'\\'+'adapted_parameters.xlsx')
    if platform.system == 'Windows':
        slash = '\\'
    else:
        slash = '/'
    try:
        delta_info_kernel.to_excel(output_path_string_trials + slash + 'adapted_delta_info_kernel.xlsx')
        delta_info.to_excel(output_path_string_trials + slash + 'adapted_delta_info.xlsx')
        # kernel_data.to_excel(output_path_string_trials + slash + 'adapted_kernels.xlsx')
        conv_data.to_excel(output_path_string_trials + slash + 'adapted_architectures.xlsx')
        # rank_final_data.to_excel(output_path_string_trials + slash + 'adapted_rank_final.xlsx')
        # rank_stable_data.to_excel(output_path_string_trials + slash + 'adapted_rank_stable.xlsx')
        """
        create_graphs(GLOBALS.EXCEL_PATH, output_path_string_trials + slash + 'adapted_kernels.xlsx',
                      output_path_string_trials + slash + 'adapted_architectures.xlsx',
                      output_path_string_trials + slash + 'adapted_rank_final.xlsx',
                      output_path_string_trials + slash + 'adapted_rank_stable.xlsx', output_path_string_graph_files)
        """
    except Exception as ex:
        print('COULD NOT CREATE GRAPHS')
        print(ex)