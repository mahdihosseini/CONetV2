import pandas as pd
def getConvIndicies(model_instance):
    parameters = list(model_instance.parameters())
    conv_layer_indicies = list()
    conv_size = list()
    kernel_size = list()
    for layer_index in range(len(parameters)):
        layer_shape = parameters[layer_index].shape
        if len(layer_shape) == 4:
            conv_layer_indicies.append(layer_index)
            conv_size.append(layer_shape[0])
            kernel_size.append(layer_shape[-1])
    return conv_layer_indicies, conv_size, kernel_size


def initialize_dataframes_and_lists(model,config):
    conv_layer_indicies, conv_size_list,kernel_size_list = getConvIndicies(model_instance = model)
    conv_data = pd.DataFrame(columns=conv_layer_indicies)
    kernel_data = pd.DataFrame(columns=conv_layer_indicies)
    rank_final_data = pd.DataFrame(columns=conv_layer_indicies)
    rank_stable_data = pd.DataFrame(columns=conv_layer_indicies)

    conv_data.loc[0] = conv_size_list
    kernel_data.loc[0] = kernel_size_list
    delta_info = pd.DataFrame(columns=['in_{}'.format(config['sel_metric']), 'out_{}'.format(config['sel_metric']), 'avg_{}'.format(config['sel_metric'])])
    delta_info_kernel = pd.DataFrame(
        columns=['delta_percentage_kernel', 'factor_scale_kernel', 'last_operation_kernel'])
    return conv_data, kernel_data, rank_final_data, rank_stable_data, delta_info, delta_info_kernel, conv_size_list, kernel_size_list,conv_layer_indicies