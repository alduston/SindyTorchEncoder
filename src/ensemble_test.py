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
#from translator_model import SindyNetTCompEnsemble
from one_way_translator_model import SindyNetTCompEnsemble
import shutil

warnings.filterwarnings("ignore")


def log_script_state(dir):
    os.system(f'cp ./ensemble_test.py {dir}/exp_script.py')
    return True

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
    train_params = {'s1_epochs': model_params['s1_epochs'],'bag_size': min(l, 8000),
                    'refinement_epochs': model_params['s1_epochs']//10,
                    'n_encoders': model_params['n_encoders'],'n_decoders': model_params['n_decoders']}
    train_params['nbags'] = max(1, l // train_params['bag_size'])
    model_params['batch_size'] = train_params['bag_size']
    model_params['run'] = run
    model_params['test_freq'] = model_params['test_freq']
    net, Loss_dict, bag_loader, test_loader = train_eas(model_params, train_params, training_data, validation_data)
    return net, Loss_dict, bag_loader, test_loader


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


def agg_comparison_plots(model, exp_label = 'exp', loss_keys = ['decoder', 'sindy_x', 'active_coeffs', 'coeff']):
    save_dir = f'../data/{exp_label}/'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass
    log_keys = ['decoder', 'sindy_x']
    marker_freq = max(1, len(list(model.item_loss_dict.values())[0])//40)
    item_loss_dict = model.item_loss_dict
    item_loss_dict = {key:val for key,val in item_loss_dict.items() if len(val) > 0}
    epochs = np.linspace(0, int(model.epoch.detach()), len(list(item_loss_dict.values())[0])).astype(int)
    for loss_key in loss_keys:
        for (key, losses) in item_loss_dict.items():

            if key.startswith(loss_key):
                if loss_key in log_keys:
                    plot_losses = np.log(np.asarray(losses))
                else:
                    plot_losses = np.asarray(losses)
                if key.endswith('agg'):
                    plt.plot(epochs, plot_losses, label='ensemble', linestyle = 'dashed',
                             marker='x', markevery=marker_freq)
                else:
                    plt.plot(epochs, plot_losses)
        if key in log_keys:
            plt.ylabel(f'Log {loss_key} loss')
            plt.title(f'Ensemble v Submodel {loss_key} Log Loss')
        else:
            plt.ylabel(loss_key)
            plt.title(f'Ensemble v Submodel')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(f'{save_dir}{loss_key}_ensemble_v_sub.png')
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


def step_2_plots(E_loss_dicts2, E_loss_dict1, Indep_loss_dict1, exp_label = 'exp'):
    try:
        os.mkdir(f'../data/{exp_label}')
    except OSError:
        pass
    save_dir = f'../data/{exp_label}'
    keys = ['E_agr_Decoder', 'E_agr_Sindy_x','active_coeffs', 'coeff']
    x = E_loss_dicts2[0]['Epoch']

    plot_labels = {'E_agr_Decoder': 'Log decode loss',
                   'E_agr_Sindy_x': 'Log sindy_x loss',
                   'active_coeffs': 'active_coeffs',
                   'coeff': 'false elim rate'}
    for key in keys:

        step_1_eloss = E_loss_dict1[key]
        step_1_eloss_vec = [step_1_eloss for i in range(len(x))]

        step_1_loss = Indep_loss_dict1[key]
        step_1_loss_vec = [step_1_loss for i in range(len(x))]
        eps =  np.abs(np.asarray(1e-18,dtype=np.float64))
        if key in ['E_agr_Decoder', 'E_agr_Sindy_x']:
            step_1_eloss_vec = np.log(eps + np.abs(np.asarray(step_1_eloss_vec,dtype=np.float64)))
            step_1_loss_vec = np.log(eps + np.abs(np.asarray(step_1_loss_vec, dtype=np.float64)))
            plt.plot(x, step_1_eloss_vec, linestyle='dashed', label='step 1 essemble error')

        plt.plot(x, step_1_loss_vec, linestyle='dashed', label='step 1 median error')

        for dict in E_loss_dicts2:
            plot_vec = dict[key][:len(x)]
            if key in ['E_agr_Decoder', 'E_agr_Sindy_x']:
                plot_vec = np.log(eps + np.abs(np.asarray(plot_vec),dtype=np.float64))
            plt.plot(x,plot_vec)
        plt.xlabel('Epoch')
        plt.ylabel(plot_labels[key])
        plt.xlim(0, x[-1])
        plt.legend()
        plt.savefig(f'{save_dir}/{key}_plot.png')
        clear_plt()
    return True



def save_model(model, bag_loader, test_loader, save_dir = 'model'):
    try:
        os.mkdir(f'../data/models/{save_dir}')
    except OSError:
        shutil.rmtree(f'../data/models/{save_dir}')
        os.mkdir(f'../data/models/{save_dir}')

    torch.save(model, f'../data/models/{save_dir}/model.pkl')
    torch.save(bag_loader, f'../data/models/{save_dir}/bag_loader.pkl')
    torch.save(test_loader, f'../data/models/{save_dir}/test_loader.pkl')
    param_str = f'encoders = {model.params["n_encoders"]}, ' \
                f'epochs = {float(model.epoch.detach())}, ' \
                f'data size = {sum([int(max(bag["x_bag"].shape)) for bag in bag_loader])}'
    os.system(f'echo {param_str} > ../data/models/{save_dir}/model_params.txt')
    return True


def load_model(save_dir = 'model'):
    model = torch.load(f'../data/models/{save_dir}/model.pkl')
    bag_loader =  torch.load(f'../data/models/{save_dir}/bag_loader.pkl')
    test_loader = torch.load(f'../data/models/{save_dir}/test_loader.pkl')
    return model,bag_loader, test_loader

#scp -r ald6fd@klone.hyak.uw.edu:/mmfs1/gscratch/dynamicsai/ald6fd/SindyTorchEncoder/data/stuff /Users/aloisduston/Desktop/Math/Research/Kutz/SindyEnsemble/data/

def basic_test(exp_label = 'exp', model_save_name = 'model0', small = False,
               replace = True, s1_epochs  = 10001):
    try:
        os.mkdir(f'../data/{exp_label}')
    except OSError:
        pass

    if small:
        params, training_data, validation_data = get_lorenz_params(train_size=20, test_size=20)
        params_update = {'replacement': replace, 'coefficient_initialization': 'constant', 'pretrain_epochs': 100,
                         'n_encoders': 20, 'n_decoders': 20, 'criterion': 'avg', 's1_epochs': s1_epochs,
                         'test_freq': 10, 'exp_label': exp_label, 's2_epochs': 0, 'crossval_freq': 500}
    else:
        params, training_data, validation_data = get_lorenz_params(train_size=64, test_size=20)
        params_update = {'replacement': replace, 'coefficient_initialization': 'constant', 'pretrain_epochs': 100,
                         'n_encoders': 10, 'n_decoders': 10, 'criterion': 'avg', 's1_epochs': s1_epochs,
                         'test_freq': 100, 'exp_label': exp_label, 's2_epochs': 0, 'crossval_freq': 500}

    params.update(params_update)
    model1, Loss_dict, bag_loader, test_loader = ea_s1_test(params, training_data, validation_data)
    agg_comparison_plots(model1, exp_label)
    save_model(model1, bag_loader, test_loader, save_dir = model_save_name)


def get_step1_med_losses(item_loss_dict):
    loss_keys = ['decoder', 'sindy_x']
    med_loss_dict = {'decoder': [], 'sindy_x': []}

    item_loss_dict = {key:val for key,val in item_loss_dict.items() if len(val)}

    for loss_key in loss_keys:
        med_losses = []
        for epoch in range(len(list(item_loss_dict.values())[0])):
            loss_vals = [item_loss_dict[key][epoch] for key in item_loss_dict.keys() if key.startswith(loss_key)]
            med_losses.append(float(np.median(np.asfarray(loss_vals))))
        med_loss_dict[loss_key] = med_losses
    return {'E_agr_Decoder': med_loss_dict['decoder'], 'E_agr_Sindy_x':  med_loss_dict['sindy_x']}


def plot_masks(ensemble_model, exp_name = 'exp'):
    try:
        os.system(f'mkdir ../data/{exp_name}/masks')
    except BaseException:
        pass
    for i,(coeffs, mask) in enumerate(zip(ensemble_model.Sindy_coeffs, ensemble_model.coefficient_masks)):
        mask_str = (coeffs * mask).detach().cpu().numpy().round(2)
        os.system(f'echo "{mask_str}" > ../data/{exp_name}/masks/mask{i}.txt')
    return True


def plot_coeffs(ensemble_model, exp_name = 'exp', trial_n = 0):
    try:
        os.system(f'mkdir ../data/{exp_name}/trial_{trial_n}')
    except BaseException:
        pass

    (x, y) = ensemble_model.coefficient_mask.shape
    for ix in range(x):
        for iy in range(y):
            coeff_vec = ensemble_model.sindy_coeffs[:, ix, iy]
            coeff_list = list(coeff_vec.detach().cpu().numpy())
            plt.hist(coeff_list)
            plt.savefig(f'../data/{exp_name}/trial_{trial_n}/coeff_plot_{ix}_{iy}.png')
            clear_plt()
    return True


def run():
    exp_label = 'one_way_exp'
    model_name = 'subsample_model'
    s1_epochs = 4001
    s2_epochs = 6001
    do_base = True
    do_corr_ensemble = True
    n_trials = 1

    if do_base:
        basic_test(exp_label= exp_label, model_save_name = model_name, small =  True,
                   replace = True, s1_epochs=  s1_epochs)

    indep_model, bag_loader, test_loader = load_model(model_name)
    net, Loss_dict,  E_loss_dict0 = train_eas_1(indep_model, bag_loader, test_loader, model_params = {'s1_epochs': 1})

    item_loss_dict = net.item_loss_dict
    med_losses = get_step1_med_losses(item_loss_dict)
    s_1_losses = {'E_agr_Decoder': med_losses['E_agr_Decoder'][-1],'E_agr_Sindy_x': med_losses['E_agr_Sindy_x'][-1],
                  'active_coeffs': Loss_dict['active_coeffs'][-1], 'coeff': Loss_dict['coeff'][-1]}

    indep_model, bag_loader, test_loader = load_model(model_name)
    plot_masks(indep_model, exp_name= exp_label)

    if do_corr_ensemble:
        indep_model.params['coefficient_initialization'] = 'xavier'
        indep_model.params['criterion'] = 'stability'
        indep_model.params['accept_threshold'] = .77
        indep_model.params['exp_label'] = exp_label

        E_loss_dicts = []
        for i in range(n_trials):
            compressor_model = SindyNetTCompEnsemble(indep_model)
            model_params = compressor_model.params
            model_params['s2_epochs'] = s2_epochs

            net, Loss_dict, E_loss_dict1, bag_loader, test_loader = train_step2(compressor_model, bag_loader,
                                                                           test_loader, compressor_model.params)
            E_loss_dicts.append(E_loss_dict1)

            step_2_plots(deepcopy(E_loss_dicts),deepcopy(E_loss_dict0), s_1_losses, exp_label=exp_label)
            plot_coeffs(compressor_model, exp_name = exp_label, trial_n = i)



if __name__=='__main__':
    sindy_x_range = [-6.57, -8.71]
    sindy_x_markers = [-6.75, -7, -7.25, -7.5, -7.75, -8.0, -8.25, -8.5]
    decoder_range = [-1.083, -7.76]
    decoder_markets = [-2, -3, -4, -5, -6, -7]
    coeff_range = [13.5, 62.1]
    coeff_markers = [20, 30, 40, 50, 60]
    run()
