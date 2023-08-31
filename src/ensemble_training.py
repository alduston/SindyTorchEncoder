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


def validate_epoch(model, data_loader, Loss_dict):
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
                    total_loss_dict[key] = val/l

    Loss_dict['epoch'].append(int(model.epoch))

    #Loss_dict['active_coeffs'].append(int(model.num_active_coeffs))
    #coeff_loss_val = coeff_pattern_loss(model.coefficient_mask, model.true_coeffs)
    #Loss_dict['coeff'].append(coeff_loss_val)

    for key, val in total_loss_dict.items():
        Loss_dict[key].append(float(val.detach().cpu()))
    model.params['eval'] = False
    return model, Loss_dict


def process_bag_coeffs(Bag_coeffs, model, plot = False):
    bag_coeffs = Bag_coeffs
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


def indep_crossval(model):
    for idx in range(model.params['n_encoders']):
        model = sub_crossval(model, idx)
    return model


def crossval(model, plot_dir = False):
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


def str_list_sum(str_list, clip = True):
    sum_str = ''
    for str in str_list:
        sum_str += str
    if clip:
        sum_str = sum_str[:-2]
    return sum_str


def print_keyval(key,val_list):
    if len(val_list):
        return f"{key.capitalize()}: {round(val_list[-1],9)}, "
    return ''


def train_eas_1(net, bag_loader, test_loader, model_params):
    Loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'sindy_z': [], 'reg': []}
    test_freq = net.params['test_freq']
    cross_val_freq =  net.params['crossval_freq']
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])
    pretrain_epocs = net.params['pretrain_epochs']

    for epoch in range(model_params['s1_epochs']):
        if (not epoch % test_freq):
            net.params['cp_batch'] = True
            net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
            print(f'{str_list_sum(["TEST: "] + [print_keyval(key, val) for key, val in Loss_dict.items()])}')
        train_epoch(net, bag_loader, optimizer)
        if epoch > pretrain_epocs and not (epoch % cross_val_freq):
            net = indep_crossval(net)

    net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict


def train_eas_2(net, bag_loader, test_loader, model_params):
    net.params['stage'] = 2
    test_freq = net.params['test_freq']
    Loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'sindy_z': [], 'reg': [], 'latent': []}
    net.s2_param_update()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=net.params['learning_rate'])
    for epoch in range(model_params['s2_epochs']):
        if  (not epoch % test_freq):
            net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
            print(f'{str_list_sum(["TEST: "] + [print_keyval(key, val) for key, val in Loss_dict.items()])}')
        train_epoch(net, bag_loader, optimizer)

    net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict, bag_loader, test_loader


def train_step2(net, bag_loader, test_loader, model_params):
    net.params['stage'] = 2
    test_freq = net.params['test_freq']
    Loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'sindy_z': [], 'reg': [], 'latent': []}
    #print(len([paramter for paramter in net.parameters()]))
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])
    net.to(net.device)
    for epoch in range(model_params['s2_epochs']):
        if  (not epoch % test_freq):
            net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
            print(f'{str_list_sum(["TEST: "] + [print_keyval(key, val) for key, val in Loss_dict.items()])}')
        train_epoch(net, bag_loader, optimizer)

    net, Loss_dict = validate_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict, bag_loader, test_loader



def train_eas(model_params, train_params, training_data, validation_data, two_stage = True):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    bag_loader = get_bag_loader(training_data, train_params, model_params, device=device,
                                augment=True, replacement=model_params['replacement'])
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNetEnsemble(model_params).to(device)
    net, Loss_dict = train_eas_1(net, bag_loader, test_loader, model_params)[:2]
    if two_stage:
        net, Loss_dict2 = train_eas_2(net, bag_loader, test_loader, model_params)[:2]
        return net, Loss_dict, Loss_dict2, bag_loader, test_loader
    return net, Loss_dict, bag_loader, test_loader

