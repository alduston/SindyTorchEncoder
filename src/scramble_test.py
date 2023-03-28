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
from torch_training import parallell_train_sindy, train_sindy, scramble_train_sindy
from torch_autoencoder import SindyNet
import pickle
import warnings
from data_utils import get_test_params, get_loader
import matplotlib.pyplot as plt
from copy import deepcopy, copy
warnings.filterwarnings("ignore")


def pa_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 5000, 'nbags': 10, 'bag_size': int(l//10), 'refinement_epochs': 0}
    model_params['batch_size'] = int(l//2)
    model_params['run'] = run
    model_params['pretrain_epochs'] = 100
    net, Loss_dict = parallell_train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def pas_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    model_params['add_noise'] = False
    l = len(training_data['x'])

    if model_params['nbags'] == 1:
        model_params['nbags'] = 50
    train_params = {'bag_epochs': 5000, 'nbags': model_params['nbags'],
                    'bag_size': int(l//2), 'refinement_epochs': 0}

    model_params['batch_size'] = int(l//8)
    model_params['crossval_freq'] = 50
    model_params['run'] = run
    model_params['pretrain_epochs'] = 100
    net, Loss_dict = scramble_train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def pas_sub_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    model_params['add_noise'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 3001, 'nbags': 3, 'bag_size': int(l//3), 'refinement_epochs': 0}
    model_params['batch_size'] = int(l)
    model_params['crossval_freq'] = 40
    model_params['run'] = run
    model_params['pretrain_epochs'] = 50
    model_params['test_freq'] = 50
    net, Loss_dict, Sub_Loss_dict = scramble_train_sindy(model_params, train_params, training_data,
                                                       validation_data,  printout = True, sub_dicts=True)
    return net, Loss_dict,Sub_Loss_dict


def a_test(model_params, training_data, validation_data, run = 0):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 4500, 'nbags': 1, 'bag_size': 100,
                    'subtrain_epochs': 60, 'bag_sub_epochs': 40, 'bag_learning_rate': .01, 'shuffle_threshold': 3,
                    'refinement_epochs': 500}
    model_params['batch_size'] = int(l//8)
    model_params['threshold_frequency'] = 50
    model_params['run'] = run
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def PAS_test(runs = 5, exp_label = '', exp_size = (256,np.inf),
                  param_updates = {}, PAparam_updates = {}, sub = False):
    Meta_PA_dict = {}
    param_updates['exp_label'] = exp_label
    for run_ix in range(runs):
        model_params, training_data, validation_data = get_test_params(exp_size[0], max_data=exp_size[1])
        model_params.update(param_updates)

        pa_params = copy(model_params)
        pa_params.update(PAparam_updates)

        if sub:
            PAnet, PALoss_dict, PASub_Loss_dict = pas_sub_test(pa_params, training_data,
                                                               validation_data, run=run_ix)
            PALoss_dict.update(PASub_Loss_dict)
        else:
            PAnet, PALoss_dict = pas_test(pa_params, training_data,  validation_data, run=run_ix)

        for key, val in PALoss_dict.items():
            if key == ('epoch'):
                if not run_ix:
                    Meta_PA_dict[f'{key}'] = val
            else:
                Meta_PA_dict[f'{key}_{run_ix}'] = val

    l1 = min([len(val) for val in Meta_PA_dict.values()])
    for key,val in Meta_PA_dict.items():
        Meta_PA_dict[key] = val[:l1]
    try:
        os.mkdir(f'../data/{exp_label}/')
    except OSError:
        pass

    Meta_PA_df = pd.DataFrame.from_dict(Meta_PA_dict, orient='columns')
    Meta_PA_df.to_csv(f'../data/{exp_label}/Meta_PA.csv')

    return Meta_PA_df



def Meta_test(runs = 5, exp_label = '', exp_size = (128,np.inf),
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

        PAnet, PALoss_dict = pas_test(pa_params, training_data, validation_data, run=run_ix)
        Anet, ALoss_dict = a_test(a_params, training_data, validation_data, run=run_ix)

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


def sub_trajectory_plot(dfs, exp_label, plot_key, runix, df_labels = ['A','PA'],
                    test_label= 'A v PA', sub_label = ''):
    if plot_key in ["sindy_x_","decoder_", "sindy_z"]:
        for df, df_label in zip(dfs,df_labels):
            plt.plot(df['epoch'], np.log(df[f'{sub_label}_{plot_key}{runix}']), label=df_label)
        plt.ylabel(f'Log {plot_key}')
    else:
        for df, df_label in zip(dfs, df_labels):
            plt.plot(df['epoch'], df[f'{sub_label}_{plot_key}{runix}'], label='df_label')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'{test_label} {plot_key} run {runix}')
    if sub_label:
        plt.savefig(f'../data/{exp_label}/{sub_label}/{plot_key}_{runix}.png')
    else:
        plt.savefig(f'../data/{exp_label}/{plot_key}_{runix}.png')
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


def avg_sub_trajectory_plot(Meta_A_df, Meta_PA_df, A_avg, PA_avg, exp_label, sub_label, plot_key):
    if plot_key in ["sindy_x_"]:
        plt.plot(Meta_A_df['epoch'], np.log(A_avg), label='A_test')
        plt.plot(Meta_PA_df['epoch'], np.log(PA_avg), label='PA_test')
        plt.ylabel(f'Log {plot_key}')
    else:
        plt.plot(Meta_A_df['epoch'], A_avg, label='A_test')
        plt.plot(Meta_PA_df['epoch'], PA_avg, label='PA_test')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'A v PA {plot_key} avg')
    plt.savefig(f'../data/{exp_label}/{sub_label}/{exp_label}_exp_{plot_key}_avg.png')
    torch_training.clear_plt()
    return True


def get_plots(Meta_A_df, Meta_PA_df, n_runs, exp_label,
              plot_keys = ["sindy_x_","decoder_", "active_coeffs_", "coeff_"]):
    try:
        os.mkdir(f'../plots/{exp_label}/')
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
        avg_PA *= (1/n_runs)
        avg_trajectory_plot(Meta_A_df, Meta_PA_df, avg_A, avg_PA, exp_label, key)
    return True


def get_sub_plots(Meta_PA_df, n_runs, exp_label, nbags,
              plot_keys = ["sindy_x_"]):
    try:
        os.mkdir(f'../data/{exp_label}')
    except OSError:
        pass

    for i in range(nbags):
        sub_label = f'bag{i}'
        try:
            os.mkdir(f'../data/{exp_label}/{sub_label}')
        except OSError:
            pass
        sub_df = {key: val for (key,val) in Meta_PA_df.items() if key.startswith(f'bag{i}')}
        sub_df['epoch'] = Meta_PA_df['epoch']
        for key in plot_keys:
            avg_PA = np.zeros(len(Meta_PA_df[f'epoch']))
            for i in range(n_runs):
                avg_PA += sub_df[f'{sub_label}_{key}{i}']
                sub_trajectory_plot([sub_df], exp_label, key, i,
                            df_labels = ['PA'], test_label='PA', sub_label=sub_label)

            avg_PA *= (1/n_runs)
            avg_sub_trajectory_plot(sub_df, sub_df, avg_PA, avg_PA, exp_label, sub_label, key)
    return True


def run():
    PAparam_updates = {'coefficient_initialization': 'xavier', 'replacement': True}
    param_updates = {'loss_weight_decoder': .1, 'nbags': 50, 'bagn_factor': 1}
    n_runs = 10
    exp_label = 'lit_informed'

    if torch.cuda.is_available():
        Meta_A_df, Meta_PA_df  = Meta_test(runs=n_runs, exp_label=exp_label, param_updates=param_updates,
                                      exp_size=(128, np.inf), PAparam_updates=PAparam_updates)
    else:
        try:
            os.mkdir(f'../plots/{exp_label}')
        except OSError:
            pass
        #Meta_A_df = pd.read_csv(f'../data/{exp_label}/Meta_A.csv')
        #Meta_PA_df = pd.read_csv(f'../data/{exp_label}/Meta_PA.csv')

    get_plots(Meta_A_df, Meta_PA_df, n_runs, exp_label)


if __name__=='__main__':
    run()
