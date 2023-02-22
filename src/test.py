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
import torch_training
from torch_training import parallell_train_sindy, train_sindy
from torch_autoencoder import SindyNet
import pickle
import warnings
from data_utils import get_test_params,get_loader
import matplotlib.pyplot as plt
from copy import deepcopy, copy

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
        train_params = {'bag_epochs': 165, 'pretrain_epochs': 50, 'nbags': int((1.5 * l) // 75), 'bag_size': 75,
                    'subtrain_epochs': 70, 'bag_sub_epochs': 15, 'bag_learning_rate': .01, 'shuffle_threshold': 3,
                        'refinement_epochs': 100}
        model_params['batch_size'] = l/10
    else:
        train_params = {'bag_epochs': 120, 'pretrain_epochs': 50, 'nbags': int((1.5 * l) // 7), 'bag_size': 7,
                        'subtrain_epochs': 30, 'bag_sub_epochs': 5, 'bag_learning_rate': .01, 'shuffle_threshold': 3,
                        'refinement_epochs': 100}
        model_params['batch_size'] = 7
    model_params['threshold_frequency'] = 25
    net, Loss_dict = torch_training.train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def BA_small_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 120, 'pretrain_epochs': 50, 'nbags': int((1.5 * l) // 7), 'bag_size': 7,
                    'subtrain_epochs': 30, 'bag_sub_epochs': 5, 'bag_learning_rate': .01,
                    'shuffle_threshold': 3, 'refinement_epochs': 100}
    model_params['batch_size'] = 7
    model_params['threshold_frequency'] = 25
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def BA_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 58, 'pretrain_epochs': 200, 'nbags':  int((1.5 * l) // 250), 'bag_size': 250,
                    'subtrain_epochs': 100, 'bag_sub_epochs': 20, 'bag_learning_rate': .01, 'shuffle_threshold': 5,
                    'refinement_epochs': 2000}
    model_params['batch_size'] = 5000
    model_params['threshold_frequency'] = 25
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def PA_small_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 1000, 'pretrain_epochs': 0, 'nbags': 3, 'bag_size': 7, 'refinement_epochs': 300}
    model_params['batch_size'] = 7
    model_params['threshold_frequency'] = 25
    net, Loss_dict = parallell_train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def PA_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 7000, 'nbags': 7, 'bag_size': int(l//7), 'refinement_epochs': 0}
    model_params['batch_size'] = int(l/2)
    model_params['threshold_frequency'] = 25
    model_params['crossval_freq'] = 25
    net, Loss_dict = parallell_train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def A_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 6000, 'nbags': l // 6, 'bag_size': 100,
                    'subtrain_epochs': 60, 'bag_sub_epochs': 40, 'bag_learning_rate': .01, 'shuffle_threshold': 3,
                    'refinement_epochs': 1000}
    model_params['batch_size'] = int(l/2)
    model_params['threshold_frequency'] = 25
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def A_small_test(model_params, training_data, validation_data):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 8000, 'nbags': int(1.5 * l // 300), 'bag_size': 300,
                    'subtrain_epochs': 80, 'bag_sub_epochs': 4, 'bag_learning_rate': .01, 'shuffle_threshold': 5,
                    'refinement_epochs': 0}
    if torch.cuda.is_available():
        model_params['batch_size'] = l
    else:
        model_params['batch_size'] = 7
    model_params['threshold_frequency'] = 25
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def Meta_test(runs = 5):
    Keys = {'decoder': [], 'sindy_x': [], 'reg': [], 'sindy_z': [], 'active_coeffs':[]}
    Meta_PA_dict = {}
    Meta_A_dict = {}
    for run_ix in range(runs):
        model_params, training_data, validation_data = get_test_params(max_data=8000)
        PAnet, PALoss_dict = PA_test(model_params, training_data, validation_data)
        Anet, ALoss_dict = A_test(model_params, training_data, validation_data)

        for key,val in ALoss_dict.items():
            if key=='epoch':
                if not run_ix:
                    Meta_A_dict[f'{key}'] = val
            else:
                Meta_A_dict[f'{key}_{run_ix}'] = val

        for key,val in PALoss_dict.items():
            if key == 'epoch':
                if not run_ix:
                    Meta_PA_dict[f'{key}'] = val
            else:
                Meta_PA_dict[f'{key}_{run_ix}'] = val

    for key in Keys:
        PAavg = np.zeros(len(Meta_PA_dict[f'{key}_{0}']))
        Aavg = np.zeros(len(Meta_A_dict[f'{key}_{0}']))
        for run_ix in range(runs):
            PAavg += np.asarray(Meta_PA_dict[f'{key}_{run_ix}'])
            Aavg += np.asarray(Meta_A_dict[f'{key}_{run_ix}'])
        Meta_A_dict[f'{key}_avg'] = Aavg * (1/runs)
        Meta_PA_dict[f'{key}_avg'] = PAavg * (1 / runs)

    PA_keys = copy(list(Meta_PA_dict.keys()))
    for key in PA_keys:
        if key.startswith('total'):
            Meta_PA_dict.pop(key,None)

    Meta_A_df = pd.DataFrame.from_dict(Meta_A_dict, orient='columns')
    Meta_A_df.to_csv('../Meta_A_df2.csv')

    Meta_BA_df = pd.DataFrame.from_dict(Meta_PA_dict, orient='columns')
    Meta_BA_df.to_csv('../Meta_PAS_df2.csv')

    return Meta_A_df, Meta_BA_df


def run():
    if torch.cuda.is_available():
        Meta_test(runs=5)
        #model_params, training_data, validation_data = get_test_params(max_data=5000)
        #A_test(model_params, training_data, validation_data)
        #PA_test(model_params, training_data, validation_data)

    else:
        Meta_A_df = pd.read_csv('../data/Meta_A_df.csv')
        Meta_PA_df = pd.read_csv('../data/Meta_PAS_df.csv')

        plt.plot(Meta_A_df['epoch'], Meta_A_df[f'active_coeffs_avg'], label='A_test')
        plt.plot(Meta_PA_df['epoch'], Meta_PA_df[f'active_coeffs_avg'], label='PA_test')
        plt.xlabel('epoch')
        plt.ylabel('# active_coeffs')
        plt.title(f'A v PA avg coeffcount')
        plt.savefig(f'../plots/PAS_exp_ncoeff_avg.png')
        plt.legend()
        torch_training.clear_plt()


        avg_loss_A = np.zeros(len(Meta_A_df[f'decoder_{0}']))
        avg_loss_BA = np.zeros(len(Meta_PA_df[f'decoder_{0}']))
        for i in [0,1,2,3,4]:
            plt.plot(Meta_A_df['epoch'], Meta_A_df[f'active_coeffs_{i}'], label = 'A_test')
            plt.plot(Meta_PA_df['epoch'], Meta_PA_df[f'active_coeffs_{i}'], label='PA_test')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('# active_coeffs')
            plt.title(f'A v PA coeffcount run {i}')
            plt.savefig(f'../plots/PAS_exp_ncoeff{i}.png')

            torch_training.clear_plt()

            Meta_A_df[f'avg_loss_{i}'] = Meta_A_df[f'decoder_{i}'] + Meta_A_df[f'sindy_x_{i}']
            Meta_PA_df[f'avg_loss_{i}'] = Meta_PA_df[f'decoder_{i}'] + Meta_PA_df[f'sindy_x_{i}']

            avg_loss_A += Meta_A_df[f'avg_loss_{i}']
            avg_loss_BA += Meta_PA_df[f'avg_loss_{i}']

            plt.plot(Meta_A_df['epoch'], np.log(Meta_A_df[f'avg_loss_{i}']), label='A_test')
            plt.plot(Meta_PA_df['epoch'], np.log(Meta_PA_df[f'avg_loss_{i}']), label='PA_test')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('Log loss')
            plt.title(f'A v PA loss run {i}')
            plt.savefig(f'../plots/PAS_exp_loss{i}.png')

            torch_training.clear_plt()

    avg_loss_A  *= .5
    avg_loss_BA *= .5
    plt.plot(Meta_A_df['epoch'], np.log(avg_loss_A), label='A_test')
    plt.plot(Meta_PA_df['epoch'], np.log(avg_loss_BA), label='PA_test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Log loss')
    plt.title(f'A v PA avg loss')
    plt.savefig(f'../plots/PAS_exp_avg_loss.png')
    torch_training.clear_plt()


if __name__=='__main__':
    run()
