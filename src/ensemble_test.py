import sys
import os
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")
import pandas as pd
import numpy as np
import torch
from ensemble_training import train_eas,train_step2, train_eas_1
import warnings
from data_utils import get_lorenz_params
import matplotlib.pyplot as plt
from copy import deepcopy
from compressor_model import SindyNetCompEnsemble
from translator_model import SindyNetTCompEnsemble
import dill
import pickle

warnings.filterwarnings("ignore")

def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True

def ea_s1_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    model_params['add_noise'] = False

    l = len(training_data['x'])
    train_params = {'s1_epochs': model_params['s1_epochs'],'bag_size': l, 'refinement_epochs': 0,
                    'n_encoders': model_params['n_encoders'],'n_decoders': model_params['n_decoders']}
    train_params['nbags'] = 1 #max(train_params['n_encoders'], train_params['n_decoders'])

    model_params['batch_size'] = l
    model_params['run'] = run
    model_params['test_freq'] = model_params['test_freq']
    net, Loss_dict, bag_loader, test_loader = train_eas(model_params, train_params, training_data, validation_data, two_stage=False)
    return net, Loss_dict, bag_loader, test_loader


def ea_test(model_params, training_data, validation_data, run  = 0):
    model_params['sequential_thresholding'] = False
    model_params['use_activation_mask'] = False
    model_params['add_noise'] = False

    l = len(training_data['x'])
    train_params = {'s1_epochs': model_params['s1_epochs'],'bag_size': l, 'refinement_epochs': 0,
                    'n_encoders': model_params['n_encoders'],'n_decoders': model_params['n_decoders']}
    train_params['nbags'] = train_params['n_encoders']


    model_params['batch_size'] = l//2
    model_params['run'] = run
    model_params['test_freq'] = model_params['test_freq']
    net, Loss_dict, bag_loader, test_loader = train_eas(model_params, train_params, training_data, validation_data)
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

    base_params, training_data, validation_data = get_lorenz_params(exp_size[0], max_data=exp_size[1], noise=noise)
    base_params['exp_label'] = exp_label
    for model, model_dict in models.items():
        model_params = deepcopy(base_params)
        model_params.update(model_dict['params_updates'])
        run_function = model_dict['run_function']
        net, loss_dict = run_function(model_params, training_data, validation_data)[:2]

        l = len(loss_dict['epoch'])
        loss_dict = {key:val[:l] for key,val in loss_dict.items()}
        loss_df = pd.DataFrame.from_dict(loss_dict, orient='columns')
        update_loss_df(model_dict, loss_df, exp_dir)
    return True


def trajectory_plot(model1_df, model2_df, exp_label, plot_key, runix, model_labels = ['EA', 'A']):
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


def sub_trajectory_plot(dfs, exp_label, plot_key, runix, df_labels = ['EA','A'],
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


def avg_trajectory_plot(model1_df, model2_df, avg1, avg2, exp_label, plot_key, model_labels = ['EA', 'A']):
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


def agg_comparison_plots(model, exp_label = 'exp', loss_keys = ['decoder', 'sindy_x', 'sindy_z']):
    save_dir = f'../data/{exp_label}/'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass
    marker_freq = max(1, len(list(model.item_loss_dict.values())[0])//40)
    item_loss_dict = model.item_loss_dict
    for loss_key in loss_keys:
        for (key, losses) in item_loss_dict.items():
            if key.startswith(loss_key):
                if key.endswith('agg'):
                    plt.plot(np.log(np.asarray(losses)), label='agg', marker='x', markevery=marker_freq)
                else:
                    plt.plot(np.log(np.asarray(losses)))
        plt.ylabel(loss_key)
        plt.xlabel('epoch')
        plt.legend()
        plt.title(f'Agg v Submodel {loss_key} log loss')
        plt.savefig(f'{save_dir}{loss_key}_agg_v_sub.png')
        clear_plt()
    return True


def get_plots(model1_df, model2_df, exp_label, plot_keys = ["sindy_x_","decoder_"],
              model_labels = ['EA', 'A'], nruns = None, factor = 1):
    save_dir = f'../data/{exp_label}/'
    try:
        os.mkdir(save_dir)
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
        try:
            runs_1 = max([int(key[-1]) for key in model1_df.columns if key.startswith('coeff')])
        except ValueError:
            runs_1 = 0
        try:
            runs_2 = max([int(key[-1]) for key in model2_df.columns if key.startswith('coeff')])
        except ValueError:
            runs_2 = 0
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


def save_model(model, bag_loader, test_loader, save_dir = 'model'):
    try:
        os.mkdir(f'../data/models/{save_dir}')
    except OSError:
        pass
    torch.save(model, f'../data/models/{save_dir}/model.pkl')
    torch.save(bag_loader, f'../data/models/{save_dir}/bag_loader.pkl')
    torch.save(test_loader, f'../data/models/{save_dir}/test_loader.pkl')
    return True


def load_model(save_dir = 'model'):
    model = torch.load(f'../data/models/{save_dir}/model.pkl')
    bag_loader =  torch.load(f'../data/models/{save_dir}/bag_loader.pkl')
    test_loader = torch.load(f'../data/models/{save_dir}/test_loader.pkl')
    return model,bag_loader, test_loader


#scp -r ald6fd@klone.hyak.uw.edu:/mmfs1/gscratch/dynamicsai/ald6fd/SindyTorchEncoder/data/stuff /Users/aloisduston/Desktop/Math/Research/Kutz/SindyEnsemble/data/

def basic_test(exp_label = 'indep_model_train_medium', model_save_name = 'model0'):
    try:
        os.mkdir(f'../data/{exp_label}')
    except OSError:
        pass

    params, training_data, validation_data = get_lorenz_params(train_size=10, test_size=5)
    #params_update = {'replacement': True, 'coefficient_initialization': 'constant', 'pretrain_epochs': 200,
                     #'n_encoders': 5, 'n_decoders': 5, 'criterion': 'avg', 's1_epochs':5000,
                     #'test_freq': 100, 'exp_label': 'two_step', 's2_epochs': 0, 'crossval_freq': 100}

    params, training_data, validation_data = get_lorenz_params(train_size=30, test_size=15)
    params_update = {'replacement': True, 'coefficient_initialization': 'constant', 'pretrain_epochs': 200,
                     'n_encoders': 10, 'n_decoders': 10, 'criterion': 'avg', 's1_epochs': 10000,
                      'test_freq': 100, 'exp_label': 'two_step', 's2_epochs': 0, 'crossval_freq': 100}

    params.update(params_update)
    model1, Loss_dict, bag_loader, test_loader = ea_s1_test(params, training_data, validation_data)
    save_model(model1, bag_loader, test_loader, save_dir = model_save_name)
    agg_comparison_plots(model1,exp_label )


def run():
    indep_model, bag_loader, test_loader = load_model('model0')
    train_eas_1(indep_model, bag_loader, test_loader, model_params = {'s1_epochs': 10})
    indep_model, bag_loader, test_loader = load_model('model0')
    print(' ')

    indep_model.params['coefficient_initialization'] = 'constant'
    compressor_model = SindyNetTCompEnsemble(indep_model)
    model_params = compressor_model.params
    model_params['s2_epochs'] = 15000
    train_step2(compressor_model, bag_loader, test_loader, compressor_model.params)


if __name__=='__main__':
    run()