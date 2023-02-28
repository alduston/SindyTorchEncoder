import sys
import os
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")
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


def PA_test_small(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 50, 'nbags': 12, 'bag_size': int(l//8), 'refinement_epochs': 0}
    model_params['batch_size'] = int(l/2)
    model_params['threshold_frequency'] = 25
    model_params['crossval_freq'] = 25
    model_params['run'] = run
    model_params['pretrain_epochs'] = 100
    net, Loss_dict = parallell_train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def PA_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 5000, 'nbags': 12, 'bag_size': int(l//8), 'refinement_epochs': 0}
    model_params['batch_size'] = int(l/2)
    model_params['threshold_frequency'] = 25
    model_params['crossval_freq'] = 25
    model_params['run'] = run
    model_params['pretrain_epochs'] = 100
    net, Loss_dict = parallell_train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def A_test(model_params, training_data, validation_data, run = 0):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 4500, 'nbags': l // 6, 'bag_size': 100,
                    'subtrain_epochs': 60, 'bag_sub_epochs': 40, 'bag_learning_rate': .01, 'shuffle_threshold': 3,
                    'refinement_epochs': 500}
    model_params['batch_size'] = int(l/2)
    model_params['threshold_frequency'] = 25
    model_params['run'] = run
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def A_test_small(model_params, training_data, validation_data, run = 0):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 45, 'nbags': l // 6, 'bag_size': 100,
                    'subtrain_epochs': 60, 'bag_sub_epochs': 40, 'bag_learning_rate': .01, 'shuffle_threshold': 3,
                    'refinement_epochs': 5}
    model_params['batch_size'] = int(l/2)
    model_params['threshold_frequency'] = 25
    model_params['run'] = run
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict



def Meta_test(runs = 5, small = False, exp_label = '', exp_size = (128,np.inf),
              param_updates = {}, PAparam_updates = {}, Aparam_updates = {}):
    Meta_PA_dict = {}
    Meta_A_dict = {}
    param_updates['exp_label'] = exp_label
    for run_ix in range(runs):
        model_params, training_data, validation_data = get_test_params(exp_size[0], max_data=exp_size[1])
        model_params.update(param_updates)

        pa_params = copy(model_params)
        pa_params.update(PAparam_updates)
        a_params = copy(model_params)
        a_params.update(Aparam_updates)

        if small:
            PAnet, PALoss_dict = PA_test_small(pa_params, training_data, validation_data, run = run_ix)
            Anet, ALoss_dict = A_test_small(a_params, training_data, validation_data, run = run_ix)
        else:
            PAnet, PALoss_dict = PA_test(pa_params, training_data, validation_data, run=run_ix)
            Anet, ALoss_dict = A_test(a_params, training_data, validation_data, run=run_ix)

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

    PA_keys = copy(list(Meta_PA_dict.keys()))
    l1 = len(Meta_PA_dict['epoch'])
    for key in PA_keys:
        if len(Meta_PA_dict[key])!=l1:
            Meta_PA_dict.pop(key, None)

    A_keys = copy(list(Meta_A_dict.keys()))
    l2 = len(Meta_A_dict['epoch'])
    for key in A_keys:
        if len(Meta_A_dict[key])!=l2:
            Meta_A_dict.pop(key, None)

    try:
        os.mkdir(f'../data/{exp_label}/')
    except OSError:
        pass

    Meta_A_df = pd.DataFrame.from_dict(Meta_A_dict, orient='columns')
    Meta_A_df.to_csv(f'../data/{exp_label}/Meta_A.csv')

    Meta_PA_df = pd.DataFrame.from_dict(Meta_PA_dict, orient='columns')
    Meta_PA_df.to_csv(f'../data/{exp_label}/Meta_PA.csv')

    return Meta_A_df, Meta_PA_df


def trajectory_plot(Meta_A_df, Meta_PA_df, exp_label, plot_key, runix):
    if plot_key in ["sindy_x_","decoder_", "sindy_z"]:
        plt.plot(Meta_A_df['epoch'], np.log(Meta_A_df[f'{plot_key}{runix}']), label='A_test')
        plt.plot(Meta_PA_df['epoch'], np.log(Meta_PA_df[f'{plot_key}{runix}']), label='PA_test')
        plt.ylabel(f'Log {plot_key}')
    else:
        plt.plot(Meta_A_df['epoch'], Meta_A_df[f'{plot_key}{runix}'], label='A_test')
        plt.plot(Meta_PA_df['epoch'], Meta_PA_df[f'{plot_key}{runix}'], label='PA_test')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'A v PA {plot_key} run {runix}')
    plt.savefig(f'../data/{exp_label}/{exp_label}_exp_{plot_key}_{runix}.png')
    torch_training.clear_plt()
    return True


def avg_trajectory_plot(Meta_A_df, Meta_PA_df, A_avg, PA_avg, exp_label, plot_key):
    if plot_key in ["sindy_x_","decoder_","sindy_z_"]:
        plt.plot(Meta_A_df['epoch'], np.log(A_avg), label='A_test')
        plt.plot(Meta_PA_df['epoch'], np.log(PA_avg), label='PA_test')
        plt.ylabel(f'Log {plot_key}')
    else:
        plt.plot(Meta_A_df['epoch'], A_avg, label='A_test')
        plt.plot(Meta_PA_df['epoch'], PA_avg, label='PA_test')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'A v PA {plot_key} avg')
    plt.savefig(f'../data/{exp_label}/{exp_label}_exp_{plot_key}_avg.png')
    torch_training.clear_plt()
    return True



def get_plots(Meta_A_df, Meta_PA_df, n_runs, exp_label, plot_keys = ["sindy_x_","decoder_", "active_coeffs_"]):
    try:
        os.mkdir(f'../plots/{exp_label}')
    except OSError:
        pass

    for key in plot_keys:
        avg_A = np.zeros(len(Meta_A_df[f'epoch']))
        avg_PA = np.zeros(len(Meta_PA_df[f'epoch']))

        for i in range(n_runs):
            avg_A += Meta_A_df[f'{key}{i}']
            avg_PA += Meta_PA_df[f'{key}{i}']
            trajectory_plot(Meta_A_df, Meta_PA_df, exp_label, key, i)

        avg_A *= (1/n_runs)
        avg_PA *= (1 / n_runs)
        avg_trajectory_plot(Meta_A_df, Meta_PA_df, avg_A, avg_PA, exp_label, key)
    return True


def run():
    exp_label='coeff_loss'
    n_runs = 1
    PAparam_updates = {'coefficient_initialization': 'xavier'}
    Meta_A_df, Meta_PA_df = Meta_test(runs=n_runs, exp_label=exp_label,
                                      exp_size=(100, 10000), PAparam_updates=PAparam_updates)
    if torch.cuda.is_available():
        Meta_A_df, Meta_PA_df = Meta_test(runs=n_runs, exp_label=exp_label,
                                          exp_size=(1024, np.inf), PAparam_updates = PAparam_updates)
    else:
        try:
            os.mkdir(f'../plots/{exp_label}')
        except OSError:
            pass
        Meta_A_df = pd.read_csv(f'../data/{exp_label}/Meta_A.csv')
        Meta_PA_df = pd.read_csv(f'../data/{exp_label}/Meta_PA.csv')

    plot_keys = ["sindy_x_", "decoder_", "active_coeffs_", "sindy_z_","coeff_"]
    get_plots(Meta_A_df, Meta_PA_df, n_runs, exp_label, plot_keys=plot_keys)



if __name__=='__main__':
    run()
