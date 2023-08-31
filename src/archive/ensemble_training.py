import numpy as np
#import tensorflow as tf
import torch
from archive.ensemble_model import SindyNetEnsemble, binarize, clear_plt
from data_utils import get_loader, get_bag_loader
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from itertools import permutations
import random
import os
from scipy.stats import anderson, normaltest
import pandas as pd



def crit_test(coeffs_vec, model, ix, iy):
    mean = float(np.mean(coeffs_vec.numpy()))
    astat = float(anderson(coeffs_vec.numpy()).statistic)
    pstat = float(normaltest(coeffs_vec.numpy())[1])
    prop = float(sum([abs(val) > .1 for val in coeffs_vec]) / len(coeffs_vec))
    try:
        model.params['criteria_info'][f'mean_{ix}_{iy}'].append(mean)
        model.params['criteria_info'][f'astat_{ix}_{iy}'].append(astat)
        model.params['criteria_info'][f'pstat_{ix}_{iy}'].append(pstat)
        model.params['criteria_info'][f'prop_{ix}_{iy}'].append(prop)

    except KeyError:
        model.params['criteria_info'][f'mean_{ix}_{iy}'] = [mean]
        model.params['criteria_info'][f'astat_{ix}_{iy}'] = [astat]
        model.params['criteria_info'][f'pstat_{ix}_{iy}'] = [pstat]
        model.params['criteria_info'][f'prop_{ix}_{iy}'] = [prop]
    return True



def process_bag_coeffs(Bag_coeffs, model):
    bag_coeffs = Bag_coeffs
    new_mask =  np.zeros(bag_coeffs.shape[1:])
    x,y = new_mask.shape
    agr_coeffs =  model.aggregate(bag_coeffs)
    coeff_criterion = model.criterion_f
    criteria_test = model.params['criteria_test']

    for ix in range(x):
        for iy in range(y):
            if model.coefficient_mask[ix, iy]:
                coeffs_vec = bag_coeffs[:,ix,iy]
                if criteria_test:
                    crit_test(coeffs_vec, model, ix, iy)
                    new_mask[ix, iy] = 1
                else:
                    new_mask[ix, iy] = float(coeff_criterion(coeffs_vec.cpu()))

    new_mask = torch.tensor(new_mask, dtype = torch.float32, device = model.params['device'])
    return new_mask, agr_coeffs


def train_one_step(model, data, optimizer):
    optimizer.zero_grad()
    if model.params['model_order'] == 1:
        loss, loss_refinement, losses = model.auto_Loss(x = data['x'], dx = data['dx'])

    else:
        loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'], dxx = data['dxx'])
    loss.backward()
    optimizer.step()
    return loss, loss_refinement, losses


def train_one_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    total_loss_dict = {}

    for batch_index, data in enumerate(data_loader):
        loss, loss_refinement, losses = train_one_step(model, data, optimizer)
        total_loss += loss
        if len(total_loss_dict.keys()):
            for key in total_loss_dict.keys():
                total_loss_dict[key] += losses[key]
        else:
            for key,val in losses.items():
                total_loss_dict[key] = val
    model.epoch += 1
    return total_loss, total_loss_dict


def validate_one_step(model, data):
    if model.params['model_order'] == 1:
            print('here')
            loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'])
    else:
        loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'], dxx=data['dxx'])
    return loss, loss_refinement, losses


def train_ensemble_step(model, data, optimizer):
    optimizer.zero_grad()
    #if model.params['translate']:
        #loss, loss_refinement, losses = model.Loss(x=data['x_bag'], dx=data['dx_bag'])
    loss, loss_refinement, losses = model.Loss(x=data['x_bag'], dx=data['dx_bag'])
    loss.backward()
    optimizer.step()
    return loss, loss_refinement, losses


def train_ensemble_epoch(model, bag_loader, optimizer):
    model.train()
    epoch_loss = 0
    model.epoch += 1
    for bag_idx, bag in enumerate(bag_loader):
        loss, loss_refinement, losses = train_ensemble_step(model, bag, optimizer)
        epoch_loss += loss
    return epoch_loss


def crossval(model):
    Bag_coeffs = deepcopy(model.sub_model_coeffs.detach())
    new_mask, agr_coeffs = process_bag_coeffs(Bag_coeffs, model)
    model.coefficient_mask = model.coefficient_mask * new_mask
    model.num_active_coeffs = int(torch.sum(copy(model.coefficient_mask)).cpu().detach())
    return model


def col_permutations(M):
    M_permutes = []
    columns = list(range(M.shape[1]))
    for perm in permutations(columns):
        M_permuted = np.stack([M[:, idx] for idx in perm])
        M_permutes.append(np.transpose(M_permuted))
    return M_permutes


def coinflip():
    return random.choice([True, False])


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


def validate_ensemble_epoch(model, data_loader, Loss_dict, true_coeffs = None):
    model.eval()
    model.params['eval'] = True
    total_loss = 0

    total_loss_dict = {}
    val_model = copy(model)
    val_model.sindy_coeffs = model.get_sindy_coefficients()
    l = len(data_loader)

    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss, loss_refinement, losses = validate_one_step(val_model, data)
            total_loss += loss/l
            if len(total_loss_dict.keys()):
                for key in total_loss_dict.keys():
                    total_loss_dict[key] += losses[key]/l
            else:
                for key, val in losses.items():
                    total_loss_dict[key] = val/l
    Loss_dict['epoch'].append(int(model.epoch))
    Loss_dict['active_coeffs'].append(int(torch.sum(val_model.coefficient_mask).cpu().detach()))
    Loss_dict['total'].append(float(total_loss.cpu().detach()))

    if true_coeffs != None:
        pred_coeffs = val_model.coefficient_mask
        coeff_loss_val = coeff_pattern_loss(pred_coeffs, true_coeffs)
        Loss_dict['coeff'].append(coeff_loss_val)

    for key, val in total_loss_dict.items():
        Loss_dict[key].append(float(val.detach().cpu()))
    model.params['eval'] = False
    return val_model, Loss_dict


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


def expand_tensor(tensor, r):
    l,L = tensor.shape
    expanded_tensor = torch.zeros(l*r, L)
    for i in range(l):
        for k in range(r):
            row_idx = i*r +k
            expanded_tensor[row_idx, :] += tensor[i,:]
    return expanded_tensor


def plot_mask(mask, savedir = 'mask.png'):
    transpose = False
    if mask.shape[0] > mask.shape[1]:
        transpose = True
        mask = mask.T
    L = max(mask.shape)
    l = min(mask.shape)
    r = int(L / l)
    print_mask = expand_tensor(mask,r)
    if transpose:
        print_mask = print_mask.T

    plt.imshow(binarize(print_mask))
    plt.savefig(savedir)
    clear_plt()
    return True


def get_output_masks(net):
    batch_len = net.params['bag_size']
    mask_shape = (batch_len, net.params['input_dim'])
    l = batch_len // net.params['nbags']
    masks = []
    device = net.device
    for i in range(net.params['nbags']-1):
        mask = torch.zeros(mask_shape, device=device)
        mask[i * l:(i + 1) * l, :] += 1.0
        masks.append(mask)
    i = net.params['nbags']-1
    final_mask = torch.zeros(mask_shape, device=device)
    final_mask[i * l:, :] += 1.0
    masks.append(final_mask)
    return torch.stack(masks)


def error_plot(net):
    for i in range(net.params['nbags']):
        errors = np.log(np.asarray(net.params['dx_error_lists'][i]))
        plt.plot(errors)
    avg_errors = np.log(np.asarray(net.params['dx_error_lists'][-1]))
    plt.plot(avg_errors, label=f'avg', marker='x')
    plt.legend()
    l = len(os.listdir('../../data/misc/dx_plots/')) + 1
    plt.savefig(f'../data/misc/dx_plots/dx_errors_{l}.png')
    clear_plt()
    return True


def train_two_stage_ea_sindy(model_params, train_params, training_data, validation_data, printout = False):
    Loss_dict = {'epoch': [], 'total': [], 'decoder': [], 'sindy_x': [], 'latent': [],
                 'reg': [], 'sindy_z': [], 'active_coeffs': [], 'coeff': [0.0]}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    train_bag_loader = get_bag_loader(training_data, train_params, model_params, device=device,
                                      augment = True, replacement = model_params['replacement'])
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNetEnsemble(model_params).to(device)
    net.params['dx_error_lists'] = [[] for i in range(net.params['nbags']+1)]
    net.params['dx_errors'] = [0 for i in range(net.params['nbags']+1)]

    for idx, bag in enumerate(train_bag_loader):
        if idx == 0:
            net.params['bag_size'] = len(bag['x_bag'])

    output_masks = get_output_masks(net)
    for i,submodel in enumerate(net.submodels):
        submodel['output_mask'] = output_masks[i]

    crossval_freq = net.params['crossval_freq']
    test_freq = net.params['test_freq']
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])
    true_coeffs = net.true_coeffs

    for epoch in range(train_params['bag_epochs']):
        if not epoch % test_freq:
            val_model, Loss_dict = validate_ensemble_epoch(net, test_loader, Loss_dict, true_coeffs)

            for i in range(net.params['nbags'] + 1):
                net.params['dx_error_lists'][i].append(net.params['dx_errors'][i])
            net.params['dx_errors'] = [0 for i in range(net.params['nbags'] + 1)]
            if printout:
                print(f'{str_list_sum(["TEST: "] + [print_keyval(key,val) for key,val in Loss_dict.items()])}')

        if not epoch % crossval_freq and epoch >= net.params['pretrain_epochs']:
            net = crossval(net)
        net.params['dx_errors'] = [0 for i in range(net.params['nbags']+1)]
        train_ensemble_epoch(net, train_bag_loader, optimizer)

    net, Loss_dict = validate_ensemble_epoch(net, test_loader, Loss_dict)
    error_plot(net)
    clear_plt()

    if net.params['criteria_test']:
        try:
            os.mkdir(f'../data/criteria_test/')
        except:
            pass
        l = len(os.listdir(f'../data/criteria_test/'))
        criterion_df = pd.DataFrame.from_dict(net.params['criteria_info'])
        criterion_df.to_csv(f'../data/criteria_test/criterion_df{i}.csv')
    return net, Loss_dict


def train_ea_sindy(model_params, train_params, training_data, validation_data, printout = False):
    Loss_dict = {'epoch': [], 'total': [], 'decoder': [], 'sindy_x': [], 'latent': [],
                 'reg': [], 'sindy_z': [], 'active_coeffs': [], 'coeff': [0.0]}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    train_bag_loader = get_bag_loader(training_data, train_params, model_params, device=device,
                                      augment = True, replacement = model_params['replacement'])
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNetEnsemble(model_params).to(device)
    net.params['dx_error_lists'] = [[] for i in range(net.params['nbags']+1)]
    net.params['dx_errors'] = [0 for i in range(net.params['nbags']+1)]

    for idx, bag in enumerate(train_bag_loader):
        if idx == 0:
            net.params['bag_size'] = len(bag['x_bag'])

    output_masks = get_output_masks(net)
    for i,submodel in enumerate(net.submodels):
        submodel['output_mask'] = output_masks[i]

    crossval_freq = net.params['crossval_freq']
    test_freq = net.params['test_freq']
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])
    true_coeffs = net.true_coeffs

    for epoch in range(train_params['bag_epochs']):
        if not epoch % test_freq:
            val_model, Loss_dict = validate_ensemble_epoch(net, test_loader, Loss_dict, true_coeffs)

            for i in range(net.params['nbags'] + 1):
                net.params['dx_error_lists'][i].append(net.params['dx_errors'][i])
            net.params['dx_errors'] = [0 for i in range(net.params['nbags'] + 1)]
            if printout:
                print(f'{str_list_sum(["TEST: "] + [print_keyval(key,val) for key,val in Loss_dict.items()])}')

        if not epoch % crossval_freq and epoch >= net.params['pretrain_epochs']:
            net = crossval(net)
        net.params['dx_errors'] = [0 for i in range(net.params['nbags']+1)]
        train_ensemble_epoch(net, train_bag_loader, optimizer)

    net, Loss_dict = validate_ensemble_epoch(net, test_loader, Loss_dict)
    error_plot(net)
    clear_plt()

    if net.params['criteria_test']:
        try:
            os.mkdir(f'../data/criteria_test/')
        except:
            pass
        l = len(os.listdir(f'../data/criteria_test/'))
        criterion_df = pd.DataFrame.from_dict(net.params['criteria_info'])
        criterion_df.to_csv(f'../data/criteria_test/criterion_df{i}.csv')
    return net, Loss_dict

