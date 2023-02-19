import sys
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")

import os
import datetime
import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
import torch
from sindy_utils import library_size
#from tf_training import train_network
import torch_training
from torch_autoencoder import SindyNet
#import tensorflow as tf
import pickle
import warnings
from data_utils import get_test_params,get_loader
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#data_path = '../examples/lorenz/'
#data_path = '../examples/pendulum/'
#data_path = '../examples/rd/'

#save_name = 'model1'
#save_name = 'model2'

#params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))
#params['save_name'] = data_path + save_name


#x = torch.rand(params['input_dim'])
#dx = torch.rand(params['input_dim'])


def BA_small_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = False
    l = len(training_data['x'])
    if torch.cuda.is_available():
        train_params = {'bag_epochs': 165, 'pretrain_epochs': 100, 'nbags': int((1.5 * l) // 75), 'bag_size': 75,
                    'subtrain_epochs': 30, 'bag_sub_epochs': 20, 'bag_learning_rate': .01, 'shuffle_threshold': 3}
        model_params['batch_size'] = l
    else:
        train_params = {'bag_epochs': 120, 'pretrain_epochs': 200, 'nbags': int((1.5 * l) // 7), 'bag_size': 7,
                        'subtrain_epochs': 30, 'bag_sub_epochs': 5, 'bag_learning_rate': .01, 'shuffle_threshold': 3}
        model_params['batch_size'] = 7
    model_params['threshold_frequency'] = 25
    net, Loss_dict = torch_training.train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def BA_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 88, 'pretrain_epochs': 200, 'nbags':  ls, 'bag_size': 250,
                    'subtrain_epochs': 80, 'bag_sub_epochs': 40, 'bag_learning_rate': .01, 'shuffle_threshold': 5}
    model_params['batch_size'] = 8000
    model_params['threshold_frequency'] = 25
    net, Loss_dict = torch_training.train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def A_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 7500, 'nbags': l // 6, 'bag_size': 100,
                    'subtrain_epochs': 60, 'bag_sub_epochs': 40, 'bag_learning_rate': .01, 'shuffle_threshold': 3}
    model_params['batch_size'] = 8000
    model_params['threshold_frequency'] = 25
    net, Loss_dict = torch_training.train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def A_small_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 5000, 'nbags': int(1.5 * l // 300), 'bag_size': 300,
                    'subtrain_epochs': 80, 'bag_sub_epochs': 4, 'bag_learning_rate': .01, 'shuffle_threshold': 5}
    if torch.cuda.is_available():
        model_params['batch_size'] = l
    else:
        model_params['batch_size'] = 7
    model_params['threshold_frequency'] = 25
    net, Loss_dict = torch_training.train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def Meta_test(runs = 5, small = False):
    Keys = {'decoder': [], 'sindy_x': [], 'reg': [], 'sindy_z': [], 'active_coeffs':[]}
    Meta_BA_dict = {}
    Meta_A_dict = {}
    for run_ix in range(runs):
        if small:
            model_params, training_data, validation_data = get_test_params(max_data=3000)
            BAnet, BALoss_dict = BA_small_test(model_params, training_data, validation_data)
            Anet, ALoss_dict = A_small_test(model_params, training_data, validation_data)
        else:
            model_params, training_data, validation_data = get_test_params(max_data=8000)
            BAnet, BALoss_dict = BA_test(model_params, training_data, validation_data)
            Anet, ALoss_dict = A_test(model_params, training_data, validation_data)

        for key,val in ALoss_dict.items():
            if key== 'epoch' and not run_ix:
                if not run_ix:
                    Meta_A_dict[f'{key}'] = val
            else:
                Meta_A_dict[f'{key}_{run_ix}'] = val

        for key,val in BALoss_dict.items():
            if key== 'epoch' and not run_ix:
                if not run_ix:
                    Meta_BA_dict[f'{key}'] = val
            else:
                Meta_BA_dict[f'{key}_{run_ix}'] = val

    for key in Keys:
        BAavg = np.zeros(len(Meta_BA_dict[f'{key}_{0}']))
        Aavg = np.zeros(len(Meta_A_dict[f'{key}_{0}']))
        for run_ix in range(runs):
            BAavg += np.asarray(Meta_BA_dict[f'{key}_{run_ix}'])
            Aavg += np.asarray(Meta_A_dict[f'{key}_{run_ix}'])
        Meta_A_dict[f'{key}_avg'] = (1/runs) * Aavg
        Meta_BA_dict[f'{key}_avg'] = (1 / runs) * BAavg

    Meta_A_df = pd.DataFrame.from_dict(Meta_A_dict, orient='columns')
    Meta_A_df.to_csv('Meta_A_df.csv')

    Meta_BA_df = pd.DataFrame.from_dict(Meta_BA_dict, orient='columns')
    Meta_BA_df.to_csv('Meta_BA_df.csv')

    return Meta_A_df, Meta_BA_df


def run():
    Meta_A_df_nn, Meta_BA_df_nn = Meta_test(runs=4, small=True)
    if torch.cuda.is_available():
        #Meta_A_df, Meta_BA_df = Meta_test(runs=6, small=False)
        Meta_A_df_nn, Meta_BA_df_nn = Meta_test(runs=4, small=True)
    else:
        #Meta_A_df, Meta_BA_df = Meta_test(runs=2, small=True)
        Meta_A_df = pd.read_csv('Meta_A_df.csv')
        Meta_BA_df = pd.read_csv('Meta_BA_df.csv')
    for i in [0,1,2,3]:
        plt.plot(Meta_A_df_nn['epoch'], Meta_A_df_nn[f'active_coeffs_{i}'], label = 'A_test')
        plt.plot(Meta_BA_df_nn['epoch'], Meta_BA_df_nn[f'active_coeffs_{i}'], label='BA_test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('# active_coeffs')
        plt.title(f'A v BA coeffcount run {i}')
        plt.savefig(f'plots/med_exp_ncum_nn{i}.png')

        torch_training.clear_plt()

        Meta_A_df[f'avg_loss_{i}'] = Meta_A_df_nn[f'decoder_{i}'] + Meta_A_df_nn[f'sindy_x_{i}']
        Meta_BA_df[f'avg_loss_{i}'] = Meta_BA_df_nn[f'decoder_{i}'] + Meta_BA_df_nn[f'sindy_x_{i}']

        plt.plot(Meta_A_df_nn['epoch'], Meta_A_df_nn[f'decoder_{i}'], label='A_test')
        plt.plot(Meta_BA_df_nn['epoch'], Meta_BA_df_nn[f'decoder_{i}'], label='BA_test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title(f'A v BA decoder loss run {i}')
        plt.savefig(f'plots/med_exp_decode_loss_nn{i}.png')

        torch_training.clear_plt()



if __name__=='__main__':
    run()
