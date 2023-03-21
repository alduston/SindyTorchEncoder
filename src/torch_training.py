import numpy as np
#import tensorflow as tf
import pickle
import torch
from torch_autoencoder import SindyNet
from data_utils import get_test_params,get_loader, get_bag_loader
from sindy_utils import get_initialized_weights
from math import inf, isinf
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from itertools import permutations
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
    vec *= (1/val)
    vec = np.asarray(np.asarray(vec, int), float)
    vec *= .05
    return vec


def f_check(tensor, ix, iy):
    return torch.abs(tensor[ix, iy]) < .1


def process_bag_coeffs(Bag_coeffs, model):
    new_mask =  np.zeros(Bag_coeffs.shape[1:])
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
    new_mask = torch.tensor(new_mask, dtype = torch.float32, device = model.params['device'])
    return new_mask, avg_coeffs


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
    new_mask, avg_coeffs = process_bag_coeffs(Bag_coeffs, model)
    coefficient_mask = new_mask * model.coefficient_mask
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


def validate_one_step(model, data):
    if model.params['model_order'] == 1:
            loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'])
    else:
        loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'], dxx=data['dxx'])
    return loss, loss_refinement, losses


def validate_one_epoch(model, data_loader, true_coeffs = None):
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
    if true_coeffs!= None:
        pred_coeffs = copy(model.sindy_coeffs) * copy(model.coefficient_mask)
        coeff_loss_val = coeff_pattern_loss(pred_coeffs, true_coeffs)
        total_loss_dict['coeff'] = coeff_loss_val
    return  total_loss, total_loss_dict


def subtrain_sindy(net, train_loader, model_params, train_params, mode, print_freq = np.inf,
                   test_loader = [], printout = False, Loss_dict = {}):
    loss_dict = {key:[] for key in Loss_dict.keys()}
    train_epochs = train_params[f'{mode}_epochs']
    optimizer = torch.optim.Adam(net.parameters(), lr= model_params['learning_rate'])
    true_coeffs = net.true_coeffs
    for epoch in range(train_epochs):
        if not epoch % print_freq:
            test_loss, test_loss_dict = validate_one_epoch(net, test_loader, true_coeffs)
            loss_dict['epoch'].append(float(net.epoch.detach().cpu()))
            loss_dict['active_coeffs'].append(int(net.num_active_coeffs))
            for key,val in test_loss_dict.items():
                try:
                    loss_dict[key].append(float(copy(val.detach()).cpu()))
                except AttributeError:
                    loss_dict[key].append(float(val))
            if printout:
                print(f'{str_list_sum(["TEST: "] + [print_keyval(key,val) for key,val in loss_dict.items()])}')

        total_loss, total_loss_dict = train_one_epoch(net, train_loader, optimizer)
        if not (net.epoch % 1000)-1:
            pass
            #plt.imshow(net.sindy_coeffs.detach().cpu().numpy(), vmin=0, vmax=1)
            #run = net.params['run']
            #plt.colorbar()
            #plt.savefig(f'../plots/coeff_hmaps/A_run{run}_{net.epoch}_hmap.png')
            #clear_plt()

    if len(Loss_dict.keys()):
        for key, val in loss_dict.items():
            Loss_dict[key] += val
        return net, loss_dict, Loss_dict
    return net, loss_dict


def train_sindy(model_params, train_params, training_data, validation_data, printout = False):
    Loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [],
                 'reg': [], 'sindy_z': [], 'active_coeffs': [], 'coeff': []}
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

    if train_params['refinement_epochs']:
        net.params['sequential_thresholding'] = False
        net, loss_dict,Loss_dict = subtrain_sindy(net, train_loader, model_params, train_params,
                                    mode='refinement', print_freq=50, test_loader=test_loader,
                                    printout=printout, Loss_dict = Loss_dict)
    return net, Loss_dict



def train_parallel_step(model, data, optimizer, idx, scramble = False):
    optimizer.zero_grad()
    if model.params['model_order'] == 1:
        if scramble:
            loss, loss_refinement, losses = model.scramble_Loss(x = data['x_bag'], dx = data['dx_bag'], idx=idx,
                                                                penalize_self=False)
        else:
            loss, loss_refinement, losses = model.auto_Loss(x=data['x_bag'], dx=data['dx_bag'],
                                                            idx=idx, penalize_self=False)
    else:
        loss, loss_refinement, losses = model.auto_Loss(x=data['x'], dx=data['dx'],
                                                        dxx = data['dxx'], idx = idx)
    loss.backward()
    optimizer.step()
    return loss, loss_refinement, losses


def train_paralell_epoch(model, bag_loader, optimizer, scramble = False):
    printfreq = model.params['train_print_freq']
    model.train()
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
        loss, loss_refinement, losses = train_parallel_step(model, bag, optimizer, idx, scramble)
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
    new_mask, avg_coeffs = process_bag_coeffs(Bag_coeffs, model)
    model.coefficient_mask = model.coefficient_mask * new_mask
    model.num_active_coeffs = int(torch.sum(copy(model.coefficient_mask)).cpu().detach())
    model.sindy_coeffs = torch.nn.Parameter(model.coefficient_mask * avg_coeffs, requires_grad=True)
    return model


def col_permutations(M):
    M_permutes = []
    columns = list(range(M.shape[1]))
    for perm in permutations(columns):
        M_permuted = np.stack([M[:, idx] for idx in perm])
        M_permutes.append(np.transpose(M_permuted))
    return M_permutes


def coeff_pattern_loss(pred_coeffs, true_coeffs, binary = True):
    pred_coeffs = copy(pred_coeffs).detach().cpu().numpy()
    true_coeffs = copy(true_coeffs).detach().cpu().numpy()
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
    return min(losses)


def validate_paralell_epoch(model, data_loader, Loss_dict, true_coeffs = None):
    model.eval()
    total_loss = 0
    total_loss_dict = {}
    Bag_coeffs = copy(model.sub_model_coeffs)
    n_bags = Bag_coeffs.shape[0]
    avg_coeffs = (1 / n_bags) * torch.sum(Bag_coeffs, dim=0)
    val_model = copy(model)
    val_model.sindy_coeffs = torch.nn.Parameter(avg_coeffs, requires_grad=True)

    if not (model.epoch % 1000)-1:
        pass
        #run =  model.params['run']
        #plt.imshow(val_model.sindy_coeffs.detach().cpu().numpy(), vmin=0, vmax=1)
        #plt.colorbar()
        #plt.savefig(f'../plots/coeff_hmaps/PA_run{run}_{model.epoch}_hmap.png')
        #clear_plt()

    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss, loss_refinement, losses = validate_one_step(val_model, data)
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

    if true_coeffs != None:
        pred_coeffs = val_model.sindy_coeffs * val_model.coefficient_mask
        coeff_loss_val = coeff_pattern_loss(pred_coeffs, true_coeffs)
        Loss_dict['coeff'].append(coeff_loss_val)

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
    Loss_dict = {'epoch': [], 'total': [], 'decoder': [], 'sindy_x': [],
                 'reg': [], 'sindy_z': [], 'active_coeffs': [], 'coeff': []}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    train_bag_loader = get_bag_loader(training_data, train_params, model_params, device=device, augment = True)
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNet(model_params).to(device)
    net.params['nbags'] = len(train_bag_loader)
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
    true_coeffs = net.true_coeffs
    for epoch in range(train_params['bag_epochs']):
        if not epoch % test_freq:
            validate_paralell_epoch(net, test_loader, Loss_dict, true_coeffs)
            if printout:
                print(f'{str_list_sum(["TEST: "] + [print_keyval(key,val) for key,val in Loss_dict.items()])}')
        if (not epoch % crossval_freq) and (epoch >= net.params['pretrain_epochs']):
            net = crossval(net)
        train_paralell_epoch(net, train_bag_loader, optimizer)

    net, Loss_dict = validate_paralell_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict


def get_masks(net):
    batch_len = net.params['bag_size']
    mask_shape = (batch_len, net.params['latent_dim'])
    l = batch_len // net.params['nbags']
    masks = []
    device = net.device
    for i in range(net.params['nbags']):
        mask = torch.zeros(mask_shape, device = device)
        mask[:, i*l:(i+1)*l] += 1.0
        masks.append(mask)
    return torch.stack(masks)



def scramble_train_sindy(model_params, train_params, training_data, validation_data, printout = False):
    Loss_dict = {'epoch': [], 'total': [], 'decoder': [], 'sindy_x': [],
                 'reg': [], 'sindy_z': [], 'active_coeffs': [], 'coeff': []}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    train_bag_loader = get_bag_loader(training_data, train_params, model_params, device=device, augment = True)
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNet(model_params).to(device)
    net.params['nbags'] = len(train_bag_loader)
    sub_model_coeffs = []
    sub_model_losses_dict = {}

    for idx, bag in enumerate(train_bag_loader):
        library_dim = net.params['library_dim']
        latent_dim = net.params['latent_dim']
        initializer, init_param = net.initializer()
        sub_model_coeffs.append(get_initialized_weights([library_dim, latent_dim], initializer,
                                       init_param = init_param, device = net.device))
        sub_model_losses_dict[f'{idx}'] = deepcopy(Loss_dict)
        if idx == 0:
            net.params['bag_size'] = len(bag['x_bag'])

    net.params['coeff_masks'] = get_masks(net)
    print(net.params['coeff_masks'].shape)
    sub_model_test_losses_dict = deepcopy(sub_model_losses_dict)

    sub_model_coeffs = torch.stack(sub_model_coeffs)
    net.sub_model_coeffs = torch.nn.Parameter(sub_model_coeffs, requires_grad=True)
    net.sub_model_losses_dict = sub_model_losses_dict
    net.sub_model_test_losses_dict = sub_model_test_losses_dict

    crossval_freq = net.params['crossval_freq']
    test_freq = net.params['test_freq']
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'], capturable=True)
    true_coeffs = net.true_coeffs
    for epoch in range(train_params['bag_epochs']):
        if not epoch % test_freq:
            validate_paralell_epoch(net, test_loader, Loss_dict, true_coeffs)
            if printout:
                print(f'{str_list_sum(["TEST: "] + [print_keyval(key,val) for key,val in Loss_dict.items()])}')
        if not epoch % crossval_freq and epoch >= net.params['pretrain_epochs']:
            net = crossval(net)
        train_paralell_epoch(net, train_bag_loader, optimizer, scramble=True)

    net, Loss_dict = validate_paralell_epoch(net, test_loader, Loss_dict)
    return net, Loss_dict











