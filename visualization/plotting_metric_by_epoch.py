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
        
        plt.plot(range(0,metric_values_epoch.shape[0]),metric_values_epoch[:,layer_index].reshape(250,),label = f"layer_{layer_index}")
        plt.xlabel('Epochs')
        plt.ylabel(f"{metric_name}")
        if counter % 5 == 0 and counter != 0:
            plt.legend()
            plt.savefig(os.path.join(save_path,f"metric_{metric_name}_vs_epochs_{counter}.png"))
            plt.close()
            plt.figure()
        
        elif counter == (metric_values_epoch.shape[1] - 1):
            plt.legend()
            plt.savefig(os.path.join(save_path,f"metric_{metric_name}_vs_epochs_{counter}.png"))
            plt.close()
            plt.figure()
        counter +=1





    

if __name__ == '__main__':
    
    path_to_spreadsheet = '/home/helen/Documents/projects/AdaS-private/Channel_Search/adas_search/initEpoch=2_        epochpert=2_searching=greedy_        epoch_sel=avg_metric=MQC_        adaptnum=25_        gamma=0.8_1/full_train/StepLR_adapt_trial=0_net=DASNet34_0.1_dataset=CIFAR100.xlsx'
    metric_names = ['in_newQC', 'out_newQC', 'in_QC','out_QC', 'in_condition','out_condition', 'in_rank', 'out_rank']
    training_metric_names = ['test_acc_','test_acc5', 'test_loss','train_acc_', 'train_acc5', 'train_loss']
    ex_data = pd.read_excel(path_to_spreadsheet)
    root_path, _ = os.path.split(path_to_spreadsheet)
    for metric_name in metric_names:
        metrics_array = extract_metrics(metric_name = metric_name,data = ex_data)
        plot_metrics(metrics_array,root_path)
    for training_metric_name in training_metric_names:
        metrics_array = extract_metrics(metric_name =training_metric_name,data = ex_data)
        plot_training(metrics_array, root_path)





