import numpy as np
import pickle
import torch
from ensemble_model import SindyNetEnsemble, binarize, clear_plt
from data_utils import get_loader, get_bag_loader
from copy import deepcopy, copy
from itertools import permutations
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime as dt

def update_list_dict(list_dict, update_dict):
    for key in list_dict.keys():
        list_dict[key].append(update_dict[key])
    return list_dict


def format(n, n_digits = 6):
    try:
        n = float(n)
        if n > 1e-4:
            return round(n,n_digits)
        a = '%E' % n
        str =  a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
        scale = str[-5:]
        digits = str[:-5]
        return digits[:min(len(digits),n_digits)] + scale
    except IndexError:
        return float(n)


def col_permutations(M):
    M_permutes = []
    columns = list(range(M.shape[1]))
    for perm in permutations(columns):
        M_permuted = np.stack([M[:, idx] for idx in perm])
        M_permutes.append(np.transpose(M_permuted))
    return M_permutes


def coeff_pattern_loss(pred_coeffs, true_coeffs, binary = True):
    pred_coeffs = deepcopy(pred_coeffs).detach().cpu().numpy()
    true_coeffs = deepcopy(true_coeffs).detach().cpu().numpy()
    if binary:
        pred_coeffs[np.where(np.abs(pred_coeffs) <.1)] = 0
        pred_coeffs[np.where(np.abs(pred_coeffs) >= .1)] = 1
        pred_coeffs = np.asarray((np.abs(pred_coeffs) ** 0.000000000001),int)
        true_coeffs = np.asarray(np.abs(true_coeffs) ** 0.00000000001,int)

    losses = []
    for permutation in col_permutations(true_coeffs):
        leq_M = pred_coeffs[np.where(pred_coeffs + .1 < permutation)]
        L_minus = len(leq_M)
        losses.append(L_minus)
    return min(losses)/np.sum(true_coeffs > 0)


def train_step(model, data, optimizer):
    optimizer.zero_grad()
    loss, losses = model.Loss(x=data['x_bag'].to(model.device), dx=data['dx_bag'].to(model.device))

    loss.backward()
    optimizer.step()
    return loss, losses


def train_epoch(model, bag_loader, optimizer):
    model.train()
    epoch_loss = 0
    model.epoch += 1
    for bag_idx, bag in enumerate(bag_loader):
        loss, losses = train_step(model, bag, optimizer)
        epoch_loss += loss
    return epoch_loss


def validate_step(model, data):
    loss, losses = model.Loss(x=data['x'].to(model.device), dx=data['dx'].to(model.device))
    return loss, losses


def get_coeff_loss(model,true_coeffs):
    try:
        coeff_loss_val = coeff_pattern_loss(model.coefficient_mask, true_coeffs)
    except BaseException:
        coeff_loss_val = 0
        for mask in model.coefficient_masks:
            pred_coeffs = mask
            coeff_loss_val += coeff_pattern_loss(pred_coeffs, true_coeffs)
        coeff_loss_val /= len(model.coefficient_masks)
    return coeff_loss_val


def validate_epoch(model, data_loader, Loss_dict,  true_coeffs = None):
    model.params['eval'] = True
    model.eval()
    total_loss = 0
    total_loss_dict = {}
    l = len(data_loader)

    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss, losses = validate_step(model, data)
            total_loss += loss/l

            if len(total_loss_dict.keys()):
                for key in total_loss_dict.keys():
                    total_loss_dict[key] += losses[key]/l
            else:
                for key, val in losses.items():
                    if key == 'active_coeffs':
                        pass
                    else:
                        total_loss_dict[key] = val/l

    Loss_dict['epoch'].append(int(model.epoch))
    Loss_dict['active_coeffs'].append(int(model.num_active_coeffs()))

    if true_coeffs != None:
        Loss_dict['coeff'].append(get_coeff_loss(model,true_coeffs))

    for key, val in total_loss_dict.items():
        Loss_dict[key].append(float(val.detach().cpu()))
    model.params['eval'] = False
    return model, Loss_dict


def process_bag_coeffs(bag_coeffs, model, plot = False):
    new_mask =  np.zeros(bag_coeffs.shape[1:])
    x,y = new_mask.shape
    agr_coeffs =  model.aggregate(bag_coeffs)
    coeff_criterion = model.criterion_f

    for ix in range(x):
        for iy in range(y):
                coeffs_vec = bag_coeffs[:,ix,iy]
                new_mask[ix, iy] = float(coeff_criterion(coeffs_vec.cpu()))
    new_mask = torch.tensor(new_mask, dtype = torch.float32, device = model.params['device'])
    return new_mask, agr_coeffs


def process_coeffs(bag_coeffs, model):
    new_mask =  np.zeros(bag_coeffs.shape)
    x,y = new_mask.shape
    coeff_criterion = model.criterion_f

    for ix in range(x):
        for iy in range(y):
                coeff = bag_coeffs[ix,iy]
                new_mask[ix, iy] = float(coeff_criterion(coeff.cpu()))
    new_mask = torch.tensor(new_mask, dtype = torch.float32, device = model.params['device'])
    return new_mask


def indep_crossval(model):
    for idx in range(model.params['n_encoders']):
        model = sub_crossval(model, idx)
    return model


def agr_crossval(model, plot_dir = False):
    coeff_shape = [model.params['n_encoders'] * model.params['n_decoders']] + list(model.Sindy_coeffs.shape[-2:])
    Bag_coeffs = deepcopy(model.Sindy_coeffs.detach()).reshape(coeff_shape)
    new_mask, agr_coeffs = process_bag_coeffs(Bag_coeffs, model, plot = plot_dir)
    model.coefficient_mask = model.coefficient_mask * new_mask
    model.num_active_coeffs = int(torch.sum(copy(model.coefficient_mask)).cpu().detach())
    return model


def sub_crossval(model, idx):
    Bag_coeffs = deepcopy(model.Sindy_coeffs.detach())[idx]
    Bag_coeffs = Bag_coeffs.reshape([1]+list(Bag_coeffs.shape))
    new_mask, agr_coeffs = process_bag_coeffs(Bag_coeffs, model)
    model.coefficient_masks[idx] = model.coefficient_masks[idx] * new_mask
    return model


def cross_val(model):
    bag_coeffs = deepcopy(model.sindy_coeffs.detach())
    if len(bag_coeffs.shape) > 2:
        new_mask = process_bag_coeffs(bag_coeffs, model)[0]
    else:
        new_mask = process_coeffs(bag_coeffs, model)
    model.coefficient_mask *=  new_mask
    return model


def str_list_sum(str_list, clip = True):
    sum_str = ''
    for str in str_list:
        sum_str += str
    if clip:
        sum_str = sum_str[:-2]
    return sum_str


def print_keyval(key,val_list):
    if len(val_list):
        return f"{key.capitalize()}: {format(val_list[-1])}, "
    return ''


def print_val_losses1(net):
    val_dict = net.val_dict
    epoch = net.epoch
    E_Decoder = format(np.mean(np.asarray(val_dict['E_Decoder'])))
    E_Sindy_x = format(np.mean(np.asarray(val_dict['E_Sindy_x'])))

    print(f'TEST: Epoch: {epoch}, E_Decoder: {E_Decoder}, E_Sindy_x: {E_Sindy_x}')
    net.refresh_val_dict = True

    return {'Epoch': epoch, 'E_agr_Decoder': E_Decoder, 'E_agr_Sindy_x': E_Sindy_x,
            'active_coeffs': net.num_active_coeffs(), 'coeff': get_coeff_loss(net, net.true_coeffs)}


def train_eas_1(net, bag_loader, test_loader, model_params):
    Loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'sindy_z': [], 'reg': [],
                 'active_coeffs': [], 'coeff': [0.0]}
    test_freq = net.params['test_freq']
    cross_val_freq =  net.params['crossval_freq']
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])
    pretrain_epocs = net.params['pretrain_epochs']
    true_coeffs = net.true_coeffs
    E_loss_dict = {'Epoch': [], 'E_agr_Decoder': [], 'E_agr_Sindy_x': [], 'active_coeffs': [], 'coeff': []}
    for epoch in range(model_params['s1_epochs']):
        if (not epoch % test_freq):
            net.params['cp_batch'] = True
            net, Loss_dict = validate_epoch(net, test_loader, Loss_dict, true_coeffs)
            print(f'{str_list_sum(["TEST: "] + [print_keyval(key, val) for key, val in Loss_dict.items()])}')
            e_loss_dict = print_val_losses1(net)
            E_loss_dict = update_list_dict(E_loss_dict, e_loss_dict)

        train_epoch(net, bag_loader, optimizer)

        if epoch > pretrain_epocs and not (epoch % cross_val_freq):
            net = indep_crossval(net)

    net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict, E_loss_dict


def print_val_losses2(net):
    val_dict = net.val_dict
    epoch = net.epoch
    E_agr_Decoder = format(np.mean(np.asarray(val_dict['E_agr_Decoder'])))
    E_agr_Sindy_x =  format(np.mean(np.asarray(val_dict['E_agr_Sindy_x'])))
    print_str = f'TEST: Epoch: {epoch}, E_agr_Decoder: {E_agr_Decoder}, E_agr_Sindy_x: {E_agr_Sindy_x}'
    print(print_str)
    os.system(f'echo {print_str} >> ./job_outputs/job0.out')

    net.refresh_val_dict = True

    return {'Epoch': epoch, 'E_agr_Decoder':  E_agr_Decoder, 'E_agr_Sindy_x': E_agr_Sindy_x,
            'active_coeffs': net.num_active_coeffs().detach().cpu(),
            'coeff': get_coeff_loss(net, net.true_coeffs)}


def train_step2(net, bag_loader, test_loader, model_params):
    net.params['stage'] = 2
    test_freq = net.params['test_freq']
    Loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'sindy_z': [], 'reg': [],
                 'active_coeffs': [],'coeff': [0.0]}
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])
    pretrain_epocs = net.params['pretrain_epochs']
    cross_val_freq = net.params['crossval_freq']
    net.to(net.device)
    true_coeffs = net.true_coeffs
    E_loss_dict = {'Epoch': [], 'E_agr_Decoder': [], 'E_agr_Sindy_x': [], 'active_coeffs': [], 'coeff': []}

    for epoch in range(model_params['s2_epochs']):
        if  (not epoch % test_freq):
            net, Loss_dict = validate_epoch(net, test_loader, Loss_dict, true_coeffs)
            print_str = str_list_sum(["TEST: "] + [print_keyval(key, val) for key, val in Loss_dict.items()])
            print(print_str)
            os.system(f'echo {print_str} >> ./job_outputs/job0.out')
            e_loss_dict = print_val_losses2(net)
            E_loss_dict = update_list_dict(E_loss_dict, e_loss_dict)

        train_epoch(net, bag_loader, optimizer)
        if epoch > pretrain_epocs and not (epoch % cross_val_freq):
            net = cross_val(net)

    net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict, E_loss_dict, bag_loader, test_loader


def train_eas(model_params, train_params, training_data, validation_data):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    bag_loader = get_bag_loader(training_data, train_params, model_params, device=device,
                                augment=True, replacement=model_params['replacement'])
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNetEnsemble(model_params).to(device)
    net, Loss_dict, E_loss_dict = train_eas_1(net, bag_loader, test_loader, model_params)
    return net, Loss_dict, bag_loader, test_loader

