import numpy as np
#import tensorflow as tf
import pickle
import torch
from torch_autoencoder import SindyNet
from data_utils import get_test_params,get_loader, get_bag_loader
from sindy_utils import get_initialized_weights
from math import inf, isinf
from copy import deepcopy, copy
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from scipy import stats as st
import random
import os


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


def train_one_step(model, data, optimizer, mode = None):
    optimizer.zero_grad()
    if model.params['model_order'] == 1:
        loss, loss_refinement, losses = model.auto_Loss(x = data['x'], dx = data['dx'])
    else:
        loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'], dxx = data['dxx'])
    loss.backward()
    optimizer.step()
    return loss, loss_refinement, losses


def train_one_bagstep(model, data, optimizer):
    optimizer.zero_grad()
    if model.params['model_order'] == 1:
        try:
            loss = model.bag_loss(x = data['x_bag'], dx = data['dx_bag'])
        except KeyError:
            loss = model.bag_loss(x=data['x'], dx=data['dx'])
    else:
        pass
        #loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'], dxx = data['dxx'])
    loss.backward()
    optimizer.step()
    return loss


def get_bag_coeffs(bag_model, bag_data, params, train_params):
    epochs = train_params['bag_sub_epochs']
    bag_model.train()
    optimizer = torch.optim.Adam([bag_model.sindy_coeffs], lr=train_params['bag_learning_rate'])
    for epoch in range(epochs):
        loss = train_one_bagstep(bag_model, bag_data, optimizer)
    return bag_model.active_coeffs()


def frac_round(vec,val):
    vec *=(1/val)
    vec = np.asarray(np.asarray(vec, int), float)
    vec *= .05
    return vec


def f_check(tensor, ix, iy):
    return torch.abs(tensor[ix, iy]) < .1


def process_bag_coeffs(Bag_coeffs, model):
    new_mask =  np.zeros(Bag_coeffs.shape[1:])
    damping_mask = model.damping_mask
    x,y = new_mask.shape

    n_samples = Bag_coeffs.shape[0]
    avg_coeffs = (1/n_samples) * torch.sum(Bag_coeffs, dim = 0)

    ip_thresh = 2/3
    for ix in range(x):
        for iy in range(y):
            coeffs_vec = Bag_coeffs[:,ix,iy]
            ip = sum([abs(val) > .1 for val in coeffs_vec])/len(coeffs_vec)
            if ip > ip_thresh:
                new_mask[ix, iy] = 1
            else:
                damping_mask[ix, iy] = min(damping_mask[ix, iy], .7)
    new_mask = torch.tensor(new_mask, dtype = torch.float32, device = model.params['device'])
    return new_mask, damping_mask, avg_coeffs


def get_choice_tensor(shape, prob, device):
    num_vals = torch.exp(torch.sum(torch.log(torch.tensor(shape)))).detach().cpu()
    num_vals = int(num_vals.detach().cpu().numpy())
    vals = torch.tensor(random.choices([1.0, 0], weights=[prob, 1 - prob], k=num_vals),
                        dtype=torch.float32, device = device).reshape(shape)
    return vals


def train_bag_epochs(model, bag_loader, params, train_params):
    Bag_coeffs = []
    for batch_index, bag_data in enumerate(bag_loader):
        bag_model = deepcopy(model)
        perturbation = .005 * torch.randn(bag_model.sindy_coeffs.shape, device = params['device'])
        bag_model.sindy_coeffs = torch.nn.Parameter(bag_model.sindy_coeffs + perturbation, requires_grad = True)
        bag_coeffs = get_bag_coeffs(bag_model, bag_data, params, train_params)
        Bag_coeffs.append(bag_coeffs)
    Bag_coeffs = torch.stack(Bag_coeffs)
    new_mask, damping_mask, avg_coeffs = process_bag_coeffs(Bag_coeffs, model)
    coefficient_mask = new_mask * model.coefficient_mask
    model.damping_mask = damping_mask
    model.sindy_coeffs = torch.nn.Parameter(coefficient_mask * avg_coeffs, requires_grad=True)

    model.coefficient_mask = coefficient_mask
    model.num_active_coeffs = torch.sum(model.coefficient_mask).cpu().detach().numpy()
    return model



def train_one_epoch(model, data_loader, optimizer, scheduler = None):
    model.train()
    total_loss = 0
    total_loss_dict = {}
    for batch_index, data in enumerate(data_loader):
        loss, loss_refinement, losses = train_one_step(model, data, optimizer)
        if scheduler:
            scheduler.step()
        total_loss += loss
        if len(total_loss_dict.keys()):
            for key in total_loss_dict.keys():
                total_loss_dict[key] += losses[key]
        else:
            for key,val in losses.items():
                total_loss_dict[key] = val
    model.epoch += 1
    return total_loss, total_loss_dict


def validate_one_step(model, data, corr = False):
    if model.params['model_order'] == 1:
            loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'])
    else:
        loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'], dxx=data['dxx'])
    return loss, loss_refinement, losses


def validate_one_epoch(model, data_loader):
    model.eval()
    total_loss = 0
    total_loss_dict = {}
    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss, loss_refinement, losses = validate_one_step(model, data)
            total_loss += loss
            if len(total_loss_dict.keys()):
                for key in total_loss_dict.keys():
                    total_loss_dict[key] += losses[key]
            else:
                for key, val in losses.items():
                    total_loss_dict[key] = val
    return  total_loss, total_loss_dict


def subtrain_sindy(net, train_loader, model_params, train_params, mode, print_freq = inf,
                   test_loader = [], printout = False, Loss_dict = {}):

    loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'reg': [], 'sindy_z': [],  'active_coeffs': []}
    pretrain_epochs = train_params[f'{mode}_epochs']
    optimizer = torch.optim.Adam(net.parameters(), lr= model_params['learning_rate'])
    for epoch in range(pretrain_epochs):
        total_loss, total_loss_dict = train_one_epoch(net, train_loader, optimizer)
        if not isinf(print_freq):
            if not epoch % print_freq:
                #if printout:
                    #pass
                    #print(f'TRAIN Epoch {net.epoch}: Active coeffs: {net.num_active_coeffs}, {[f"{key}: {val.cpu().detach().numpy()}" for (key, val) in total_loss_dict.items()]}')
                if len(test_loader):
                    test_loss, test_loss_dict = validate_one_epoch(net, test_loader)
                    for key,val in test_loss_dict.items():
                        loss_dict[key].append(float(val.detach().cpu()))
                    loss_dict['epoch'].append(float(net.epoch.detach().cpu()))
                    loss_dict['active_coeffs'].append(int(net.num_active_coeffs))
                    if printout:
                       print(f'{str_list_sum(["TEST: "] + [print_keyval(key,val) for key,val in loss_dict.items()])}')

    if len(Loss_dict.keys()):
        for key, val in loss_dict.items():
            Loss_dict[key] += val
        return net, loss_dict, Loss_dict
    return net, loss_dict


def train_sindy(model_params, train_params, training_data, validation_data, printout = False):
    Loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'reg': [], 'sindy_z': [], 'active_coeffs': []}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    train_loader = get_loader(training_data, model_params, device=device)
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNet(model_params).to(device)
    net, loss_dict, Loss_dict = subtrain_sindy(net, train_loader, model_params, train_params,
                         mode = 'pretrain', print_freq = 50, test_loader = test_loader,
                         printout= printout, Loss_dict = Loss_dict)

    if train_params['bag_epochs']:
        bag_loader = get_bag_loader(training_data, train_params, model_params, device=device)
        shuffle_threshold = train_params['shuffle_threshold']
        for epoch in range(train_params['bag_epochs']):
            if epoch and not epoch%shuffle_threshold:
                bag_loader = get_bag_loader(training_data, train_params, model_params, device=device)
            net  = train_bag_epochs(net, bag_loader, model_params, train_params)
            net, loss_dict ,Loss_dict = subtrain_sindy(net, train_loader, model_params,train_params,
                                            mode='subtrain', print_freq = 50, test_loader = test_loader,
                                            printout= printout,  Loss_dict = Loss_dict)
    if train_params['refinement_epochs']:
        net.params['sequential_thresholding'] = False
        net, loss_dict,Loss_dict = subtrain_sindy(net, train_loader, model_params, train_params,
                                    mode='refinement', print_freq=50, test_loader=test_loader,
                                    printout=printout, Loss_dict = Loss_dict)
    return net, Loss_dict



def train_parallel_step(model, data, optimizer, idx, spooky = True):
    optimizer.zero_grad()
    if model.params['model_order'] == 1:
        loss, loss_refinement, losses = model.auto_Loss(x = data['x_bag'], dx = data['dx_bag'],  idx = idx, spooky = spooky)
    else:
        loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'], dxx = data['dxx'], idx = idx)
    loss.backward()
    optimizer.step()
    return loss, loss_refinement, losses


def train_paralell_epoch(model, bag_loader, optimizer):
    printfreq = model.params['train_print_freq']
    model.eval()
    epoch_loss = 0

    sub_model_losses_dict = model.sub_model_losses_dict
    update = bool(not (model.epoch + 1) % (model.params['update_freq']))
    model.epoch += 1
    model.activation_mask *= model.damping_mask

    print_bool = model.epoch and not model.epoch % printfreq
    if print_bool:
        print_dict = {key: 0 for key in sub_model_losses_dict['0']}
        print_dict['epoch'] = model.epoch
        print_dict['active_coeffs'] = model.num_active_coeffs

    for idx, bag in enumerate(bag_loader):
        loss, loss_refinement, losses = train_parallel_step(model, bag, optimizer, idx)
        epoch_loss += loss

        if update:
            sub_losses_dict = sub_model_losses_dict[f'{idx}']
            sub_losses_dict['epoch'].append(model.epoch)
            sub_losses_dict['active_coeffs'] = model.num_active_coeffs

            for key, val in losses.items():
                sub_losses_dict[key].append(val)
                if print_bool:
                    print_dict[key] += val
            sub_model_losses_dict[f'{idx}'] = sub_losses_dict
    if print_bool:
        print(f'{str_list_sum(["TRAIN: "] + [f"{key.capitalize()}: {val}, " for key, val in print_dict.items()])}')
    return model


def crossval(model):
    Bag_coeffs = model.sub_model_coeffs
    new_mask, damping_mask, avg_coeffs = process_bag_coeffs(Bag_coeffs, model)

    model.coefficient_mask = model.coefficient_mask * new_mask
    model.damping_mask = damping_mask
    model.num_active_coeffs = int(torch.sum(model.coefficient_mask).cpu().detach())
    model.sindy_coeffs = torch.nn.Parameter(model.coefficient_mask * avg_coeffs, requires_grad=True)
    return model


def validate_paralell_epoch(model, data_loader, Loss_dict):
    model.eval()
    total_loss = 0
    total_loss_dict = {}
    Bag_coeffs = copy(model.sub_model_coeffs)
    n_bags = Bag_coeffs.shape[0]
    avg_coeffs = (1 / n_bags) * torch.sum(Bag_coeffs, dim=0)
    val_model = copy(model)
    val_model.sindy_coeffs = torch.nn.Parameter(avg_coeffs, requires_grad=True)
    plt.imshow(val_model.sindy_coeffs.detach().cpu().numpy(), vmin=0, vmax=1)

    plt.colorbar()
    plt.savefig(f'../plots/coeff_hmaps/{val_model.epoch}_hmmap.png')
    clear_plt()

    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss, loss_refinement, losses = validate_one_step(val_model, data, corr = False)
            total_loss += loss
            if len(total_loss_dict.keys()):
                for key in total_loss_dict.keys():
                    total_loss_dict[key] += losses[key]
            else:
                for key, val in losses.items():
                    total_loss_dict[key] = val

    Loss_dict['epoch'].append(int(model.epoch))
    Loss_dict['active_coeffs'].append(int(torch.sum(val_model.coefficient_mask).cpu().detach()))
    Loss_dict['total'].append(float(total_loss.cpu().detach()))
    for key, val in total_loss_dict.items():
        Loss_dict[key].append(float(val.detach().cpu()))
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


def parallell_train_sindy(model_params, train_params, training_data, validation_data, printout = False):
    Loss_dict = {'epoch': [], 'total': [], 'decoder': [], 'sindy_x': [], 'reg': [],
                 'spooky':[], 'sindy_z': [], 'active_coeffs': []}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    train_bag_loader = get_bag_loader(training_data, train_params, model_params, device=device)
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNet(model_params).to(device)
    sub_model_coeffs = []
    sub_model_losses_dict = {}

    for idx, bag in enumerate(train_bag_loader):
        library_dim = net.params['library_dim']
        latent_dim = net.params['latent_dim']

        initializer, init_param = net.initializer()
        sub_model_coeffs.append(get_initialized_weights([library_dim, latent_dim], initializer,
                                       init_param = init_param, device = net.device))
        sub_model_losses_dict[f'{idx}'] = deepcopy(Loss_dict)
    sub_model_test_losses_dict = deepcopy(sub_model_losses_dict)

    sub_model_coeffs = torch.stack(sub_model_coeffs)
    net.sub_model_coeffs = torch.nn.Parameter(sub_model_coeffs, requires_grad=True)
    net.sub_model_losses_dict = sub_model_losses_dict
    net.sub_model_test_losses_dict = sub_model_test_losses_dict

    crossval_freq = net.params['crossval_freq']
    test_freq = net.params['test_freq']
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])
    for epoch in range(train_params['bag_epochs']):
        train_paralell_epoch(net, train_bag_loader,optimizer)
        if not epoch % crossval_freq and epoch:
            net = crossval(net)
        if not epoch % test_freq:
            validate_paralell_epoch(net, test_loader, Loss_dict)
            if printout:
                print(f'{str_list_sum(["TEST: "] + [print_keyval(key,val) for key,val in Loss_dict.items()])}')

    net, Loss_dict = validate_paralell_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict

    #train_loader = get_loader(training_data, model_params, device=device)
    #net, loss_dict, Loss_dict = subtrain_sindy(net, train_loader, model_params, train_params, mode='refinement',
                                               #print_freq=50, test_loader=test_loader, printout=printout, Loss_dict=Loss_dict














