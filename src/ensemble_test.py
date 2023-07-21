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
from torch_training import train_sindy, clear_plt
from ensemble_training import train_ea_sindy
import pickle
import warnings
from data_utils import get_test_params, get_loader
import matplotlib.pyplot as plt
from copy import deepcopy, copy
warnings.filterwarnings("ignore")


def ea_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    model_params['add_noise'] = False

    l = len(training_data['x'])
    train_params = {'bag_epochs': model_params['max_epochs'], 'nbags': model_params['nbags'],
                    'bag_size': l, 'refinement_epochs': 0}

    model_params['batch_size'] = l
    model_params['crossval_freq'] = 50
    model_params['run'] = run
    model_params['pretrain_epochs'] = 100
    model_params['test_freq'] = model_params['test_freq']
    net, Loss_dict = train_ea_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def a_test(model_params, training_data, validation_data, run = 0):
    model_params['sequential_thresholding'] = True
    l = len(training_data['x'])
    train_params = {'bag_epochs': 0, 'pretrain_epochs':  model_params['max_epochs'], 'nbags': 1,
                    'refinement_epochs': model_params['refinement_epochs']}

    model_params['batch_size'] = l//2
    model_params['print_freq'] = model_params['test_freq']
    model_params['run'] = run
    net, Loss_dict = train_sindy(model_params, train_params, training_data, validation_data, printout = True)
    return net, Loss_dict


def drop_index(df):
    for col in df.columns:
        if col.startswith('Unnamed'):
            df = df.drop(col, axis = 1)
    return df


def update_loss_df(model_dict, loss_df, exp_dir):
    csv_name = f'{model_dict["label"]}.csv'
    if csv_name in os.listdir(exp_dir):
        current_loss_df =  drop_index(pd.read_csv(f'{exp_dir}{csv_name}'))

        k = (len(current_loss_df.columns) - 1)//(len(loss_df.columns) - 1)
        for col in loss_df.columns:
            if col not in ['epoch']:
                if not col.startswith('Unnamed'):
                    current_loss_df[f'{col}_{k}'] = loss_df[col]
    else:
        current_loss_df = pd.DataFrame()
        current_loss_df['epoch'] = loss_df['epoch']
        for col in loss_df.columns:
            if col not in ['epoch']:
                current_loss_df[f'{col}_0'] = loss_df[col]
    drop_index(current_loss_df).to_csv(f'{exp_dir}{csv_name}')
    return current_loss_df


def comparison_test(models, exp_label = '', exp_size = (128,np.inf), noise = 1e-6):
    exp_dir = f'../data/{exp_label}/'
    try:
        os.mkdir(exp_dir)
    except OSError:
        pass

    base_params, training_data, validation_data = get_test_params(exp_size[0], max_data=exp_size[1], noise=noise)
    base_params['exp_label'] = exp_label
    for model, model_dict in models.items():
        model_params = deepcopy(base_params)
        model_params.update(model_dict['params_updates'])
        run_function = model_dict['run_function']
        net, loss_dict = run_function(model_params, training_data, validation_data)

        l = len(loss_dict['epoch'])
        loss_dict = {key:val[:l] for key,val in loss_dict.items()}
        loss_df = pd.DataFrame.from_dict(loss_dict, orient='columns')
        update_loss_df(model_dict, loss_df, exp_dir)
    return True


def trajectory_plot(model1_df, model2_df, exp_label, plot_key, runix, model_labels = ['A', 'EA']):
    label1, label2 = model_labels
    if plot_key in ["sindy_x_","decoder_", "sindy_z"]:
        plt.plot(model1_df['epoch'], np.log(model1_df[f'{plot_key}{runix}']), label=label1)
        plt.plot(model2_df['epoch'], np.log(model2_df[f'{plot_key}{runix}']), label=label2)
        plt.ylabel(f'Log {plot_key}')
    else:
        plt.plot(model1_df['epoch'], model1_df[f'{plot_key}{runix}'], label=label1)
        plt.plot(model2_df['epoch'], model2_df[f'{plot_key}{runix}'], label=label2)
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'{label1} v {label2} {plot_key} run {runix}')
    plt.savefig(f'../data/{exp_label}/{exp_label}_exp_{plot_key}_{runix}.png')
    clear_plt()
    return True


def sub_trajectory_plot(dfs, exp_label, plot_key, runix, df_labels = ['A','EA'],
                    test_label= 'A v EA', sub_label = ''):
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
    clear_plt()
    return True


def avg_trajectory_plot(model1_df, model2_df, avg1, avg2, exp_label, plot_key, model_labels = ['A', 'EA']):
    label1,label2 = model_labels
    if plot_key in ["sindy_x_","decoder_","sindy_z_"]:
        plt.plot(model1_df['epoch'], np.log(avg1), label=label1)
        plt.plot(model2_df['epoch'], np.log(avg2), label=label2)
        plt.ylabel(f'Log {plot_key}')
    else:
        plt.plot(model1_df['epoch'], avg1,  label=label1)
        plt.plot(model2_df['epoch'], avg2, label=label2)
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'{label1} v {label2} {plot_key} avg')
    plt.savefig(f'../data/{exp_label}/{exp_label}_exp_{plot_key}_avg.png')
    clear_plt()
    return True


def avg_sub_trajectory_plot(Meta_A_df, Meta_PA_df, A_avg, PA_avg, exp_label, sub_label, plot_key):
    if plot_key in ["sindy_x_"]:
        plt.plot(Meta_A_df['epoch'], np.log(A_avg), label='A_test')
        plt.plot(Meta_PA_df['epoch'], np.log(PA_avg), label='EA_test')
        plt.ylabel(f'Log {plot_key}')
    else:
        plt.plot(Meta_A_df['epoch'], A_avg, label='A_test')
        plt.plot(Meta_PA_df['epoch'], PA_avg, label='EA_test')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(f'A v EA {plot_key} avg')
    plt.savefig(f'../data/{exp_label}/{sub_label}/{exp_label}_exp_{plot_key}_avg.png')
    clear_plt()
    return True


def get_key_med(df,key, nruns):
    l = len(df[f'epoch'])
    key_df = df[[f'{key}{i}' for i in range(nruns)]]
    med = key_df.median(axis = 1)
    return np.asarray(med)

def list_in(str, str_list):
    for sub_str in str_list:
        if sub_str in str:
            return True
    return False


def get_plots(model1_df, model2_df, exp_label, plot_keys = ["sindy_x_","decoder_", "active_coeffs_", "coeff_"],
              model_labels = ['A', 'EA'], nruns = None, factor = 1):
    try:
        os.mkdir(f'../plots/{exp_label}/')
    except OSError:
        pass

    str_list = ['decoder', 'sindy_x', 'sindy_z']
    for col in model1_df.columns:
        if list_in(col, str_list):

            model1_df[col] = model1_df[col] / factor

    for col in model2_df.columns:
        if list_in(col, str_list):
            model2_df[col] = model2_df[col] / factor


    if not nruns:
        runs_1 = max([int(key[-1]) for key in model1_df.columns if key.startswith('coeff')])
        runs_2 = max([int(key[-1]) for key in model2_df.columns if key.startswith('coeff')])
        nruns = min(runs_1, runs_2) + 1

    for key in plot_keys:
        avg_1 = np.zeros(len(model1_df[f'epoch']))
        avg_2 = np.zeros(len(model2_df[f'epoch']))

        for i in range(nruns):
            avg_1 += model1_df[f'{key}{i}']
            avg_2 += model2_df[f'{key}{i}']
            trajectory_plot(model1_df, model2_df, exp_label, key, i, model_labels = model_labels)

        avg_1 *= (1/nruns)
        avg_2 *= (1/nruns)
        if key in ['sindy_x_','decoder_']:
            avg_1 = get_key_med(model1_df, key, nruns)
            avg_2 = get_key_med(model2_df, key, nruns)
        avg_trajectory_plot(model1_df, model2_df, avg_1, avg_2, exp_label, key, model_labels = model_labels)
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
                            df_labels = ['EA'], test_label='EA', sub_label=sub_label)

            avg_PA *= (1/n_runs)
            avg_sub_trajectory_plot(sub_df, sub_df, avg_PA, avg_PA, exp_label, sub_label, key)
    return True


def update_df_cols(df, update_num):
    rename_dict = {}
    for col in df.columns:
        try:
            try:
                col_num = int(col[-2:])
                col_num += update_num
                rename_dict[col] = f'{col[:-2]}{col_num}'
            except BaseException:
                col_num = int(col[-1:])
                col_num += update_num
                rename_dict[col] = f'{col[:-1]}{col_num}'
        except BaseException:
            pass
    return df.rename(columns=rename_dict)


def test(size = 40, epochs = 1000, nbags = 10, exp_name = 'exp'):

    params_1 = {'nbags': nbags, 'replacement': True, 'criterion': 'stability',
                'coefficient_initialization': 'xavier', 'max_epochs': epochs, 'test_freq': 50, 'exp_name': exp_name}

    params_2 = {'nbags': 1, 'replacement': False, 'criterion': 'avg', 'coefficient_initialization': 'constant',
                'max_epochs': epochs, 'test_freq': 50, 'exp_name': exp_name}

    model_1 = {'params_updates': params_1, 'run_function': ea_test, 'label': 'Meta_EA'}
    model_2 = {'params_updates': params_2, 'run_function': ea_test, 'label': 'Meta_A'}
    models_dict = {'Meta_EA': model_1, 'Meta_A': model_2}

    comparison_test(models_dict, exp_name, exp_size=(size, np.inf))

    label1 = 'Meta_EA'
    label2 = 'Meta_A'

    Meta_df_1 = pd.read_csv(f'../data/{exp_name}/{label1}.csv')
    Meta_df_2 = pd.read_csv(f'../data/{exp_name}/{label2}.csv')

    get_plots(Meta_df_1, Meta_df_2, exp_name, model_labels=['Meta_EA', 'Meta_A'], factor=1)

    return True

#scp -r ald6fd@klone.hyak.uw.edu:/mmfs1/gscratch/dynamicsai/ald6fd/alt/SindyTorchEncoder/data/stuff /Users/aloisduston/Desktop/Math/Research/Kutz/SindyTorchEncoder/data/

def run():
    test(size = 50, epochs = 3000, nbags = 30, exp_name='basic_new')

if __name__=='__main__':
    run()
