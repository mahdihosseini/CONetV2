import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_metrics(metric_name, data):
    
    column_names = list()
    metric_values_epoch = list()
    columns = data.columns
    for column in columns:
        if column.startswith(metric_name):
            column_names.append(column)
            epoch = int(column.split('_')[-1])
            metric_values_epoch.append(data[column].to_numpy())


    return np.array(metric_values_epoch)
def plot_training(metric_values_epoch, save_path):
    plt.figure()
    plt.plot(range(0,metric_values_epoch.shape[0]),metric_values_epoch[:,0].reshape(250,),label = f"{training_metric_name}")
    plt.xlabel('Epochs')
    plt.ylabel(f"{training_metric_name}")
    plt.legend()
    plt.savefig(os.path.join(save_path,f"metric_{training_metric_name}_vs_epochs.png"))

def plot_metrics(metric_values_epoch,save_path):
    counter = 0
    for layer_index in range(0, metric_values_epoch.shape[1]):
        
        # print(counter,metric_values_epoch.shape[1] - 1)
        plt.plot(range(0,metric_values_epoch.shape[0]),metric_values_epoch[:,layer_index].reshape(25,),label = f"layer_{layer_index}")
        plt.xlabel('Epochs')
        # plt.ylabel(f"{metric_name}")
        plt.ylabel('Channel Size')
        if counter % 5 == 0 and counter != 0:
            plt.legend()
            plt.savefig(os.path.join(save_path,f"channel_change_vs_trials_{counter}.png"))
            plt.close()
            plt.figure()
        
        elif counter == (metric_values_epoch.shape[1] - 1):
            plt.legend()
            plt.savefig(os.path.join(save_path,f"channel_change_vs_trials_{counter}.png"))
            plt.close()
            plt.figure()
        counter +=1





def accumulate_epoch_metrics(epoch_metrics, method):
    if method == 'avg':
        return np.average(epoch_metrics,axis = 0)
    elif method == 'max':
        return np.amax(epoch_metrics,axis = 0)
def getTrialConvSizes(excel_path):
    data = pd.read_excel(excel_path)
    data_np = data.to_numpy()
    return data_np


if __name__ == '__main__':

    excel_path = '/home/helen/Documents/projects/AdaS-private/Channel_Search/adas_search/initEpoch=2_        epochpert=2_searching=greedy_        epoch_sel=avg_metric=MQC_        adaptnum=25_        gamma=0.8_1/Trials/adapted_architectures.xlsx'
    experiment_dir, _ = os.path.split(excel_path)
    conv_sizes_by_trial = getTrialConvSizes(excel_path)

    plot_metrics(conv_sizes_by_trial, os.path.join(experiment_dir, "ChannelEvolutionFolder"))

   




