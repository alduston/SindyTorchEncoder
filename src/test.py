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
from torch_training import train_sindy, scramble_train_sindy
from torch_autoencoder import SindyNet
import pickle
import warnings
from data_utils import get_test_params, get_loader
import matplotlib.pyplot as plt
from copy import deepcopy, copy
warnings.filterwarnings("ignore")


def pas_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    model_params['add_noise'] = False

    l = len(training_data['x'])

    train_params = {'bag_epochs': model_params['max_epochs'], 'nbags': model_params['nbags'],
                    'bag_size': l//2, 'refinement_epochs': 0}

    model_params['batch_size'] = l//2
    model_params['crossval_freq'] = 40
    model_params['run'] = run
    model_params['pretrain_epochs'] = 51
    net, Loss_dict = scramble_train_sindy(model_params, train_params, training_data, validation_data,  printout = True)
    return net, Loss_dict


def pas_recursive(model_params, training_data, validation_data, run  = 0, k = 5):
    Loss_dict = {}
    for i in range(k):
        net,loss_dict = pas_test(model_params, training_data, validation_data)
        model_params['coefficient_mask'] = net.coefficient_mask.detach().numpy()
        loss_dict['epoch'] = [val +  model_params['max_epochs'] for val in loss_dict['epoch']]
        if len(Loss_dict.keys()):
            Loss_dict = {key: val.append(loss_dict[key]) for key,val in Loss_dict.items()}
        else:
            Loss_dict = loss_dict
    return net, Loss_dict


def pas_sub_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    model_params['add_noise'] = False
    l = len(training_data['x'])
    train_params = {'bag_epochs': 1001, 'nbags': 3, 'bag_size': int(l//3), 'refinement_epochs': 0}
    model_params['batch_size'] = int(l//2)
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
    train_params = {'bag_epochs': 0, 'pretrain_epochs': 1800, 'nbags': 1, 'bag_size': 100,
                    'subtrain_epochs': 60, 'bag_sub_epochs': 40, 'bag_learning_rate': .01, 'shuffle_threshold': 3,
                    'refinement_epochs': 200}
    model_params['batch_size'] = l//8
    model_params['threshold_frequency'] = 50
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


def trajectory_plot(model1_df, model2_df, exp_label, plot_key, runix, model_labels = ['A', 'PA']):
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


def avg_trajectory_plot(model1_df, model2_df, avg1, avg2, exp_label, plot_key, model_labels = ['A', 'PA']):
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


def get_plots(model1_df, model2_df, exp_label,
              plot_keys = ["sindy_x_","decoder_", "active_coeffs_", "coeff_"], model_labels = ['A', 'EA']):
    try:
        os.mkdir(f'../plots/{exp_label}/')
    except OSError:
        pass

    runs_1 = max([int(key[-1]) for key in model1_df.columns if key.startswith('coeff')])
    runs_2 = max([int(key[-1]) for key in model2_df.columns if key.startswith('coeff')])
    n_runs = min(runs_1, runs_2) + 1

    for key in plot_keys:
        avg_1 = np.zeros(len(model1_df[f'epoch']))
        avg_2 = np.zeros(len(model2_df[f'epoch']))

        for i in range(n_runs):
            avg_1 += model1_df[f'{key}{i}']
            avg_2 += model2_df[f'{key}{i}']
            trajectory_plot(model1_df, model2_df, exp_label, key, i, model_labels = model_labels)

        avg_1 *= (1/n_runs)
        avg_2 *= (1/n_runs)
        avg_trajectory_plot(model1_df, model1_df, avg_1, avg_2, exp_label, key, model_labels = model_labels)
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
    exp_label = 'L1_v_Recursive'

    params_1 = {'coefficient_initialization': 'xavier',
                'replacement': True, 'avg_crossval': False, 'c_loss': False,
                'loss_weight_decoder': .1, 'nbags': 30, 'bagn_factor': 1, 'max_epochs': 6000}

    params_2 = {'loss_weight_decoder': .1, 'nbags': 1, 'bagn_factor': 1,
                'expand_sample': False}

    params_3 = {'coefficient_initialization': 'xavier',
                 'replacement': True, 'avg_crossval': False, 'c_loss': True,
                 'loss_weight_decoder': .1, 'nbags': 30, 'bagn_factor': 1}

    params_4 = {'coefficient_initialization': 'xavier',
                'replacement': True, 'avg_crossval': False, 'c_loss': False,
                'hybrid_reg': True, 'loss_weight_decoder': .1, 'nbags': 30,
                'bagn_factor': 1,'max_epochs': 1200}

    model_1 = {'params_updates': params_1, 'run_function': pas_test, 'label': 'EA_L1'}
    model_2 = {'params_updates': params_4, 'run_function': pas_recursive, 'label': 'EA_recursive'}

    models_dict = {'EA_L1': model_1, 'EA_recursive': model_2}


    if torch.cuda.is_available():
        comparison_test(models_dict, exp_label, exp_size=(100, np.inf))
    else:
        exp = 'AA_comparison_long'
        try:
            os.mkdir(f'../plots/{exp}')
        except OSError:
            pass
        label1 = 'EA_results'
        label2 = 'EAalt_results'

        os.rename(f'../data/{exp}/{label1}.csv', f'../data/{exp}/{label1}_local.csv')
        os.rename(f'../data/{exp}/{label2}.csv', f'../data/{exp}/{label2}_local.csv')

        Meta_df_1 = pd.read_csv(f'../data/{exp}/{label1}_local.csv')
        Meta_df_2 = pd.read_csv(f'../data/{exp}/{label2}_local.csv')


        get_plots(Meta_df_1, Meta_df_2, exp, model_labels = ['EA', 'EAalt'])

if __name__=='__main__':
    run()

'Important commit: 9ee60e3 '
