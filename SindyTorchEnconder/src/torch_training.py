import numpy as np
#import tensorflow as tf
import pickle
import torch
from torch_autoencoder import SindyNet
from data_utils import get_test_params,get_loader, get_bag_loader
from math import inf, isinf
from copy import deepcopy
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


def process_bag_coeffs(Bag_coeffs, params, model, minboost = False):
    new_mask = np.zeros(Bag_coeffs.shape[1:])
    x,y = new_mask.shape

    n_samples = Bag_coeffs.shape[0]
    avg_coeffs = (1/n_samples) * torch.sum(Bag_coeffs, dim = 0)
    min_ix = []
    min_ip = 1

    for ix in range(x):
        for iy in range(y):
            coeffs_vec = Bag_coeffs[:,ix,iy]
            ip = sum([abs(val) > .1 for val in coeffs_vec])/len(coeffs_vec)

            if 0 < ip < min_ip:
                min_ix = [ix,iy]
            new_mask[ix, iy] = 1 if ip > .5 else 0

    new_mask = torch.tensor(new_mask, dtype = torch.float32, device = params['device'])
    if minboost:
        avg_coeffs[min_ix] *= .6
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
    new_mask, avg_coeffs = process_bag_coeffs(Bag_coeffs, params, model)
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
                   test_loader = [], printout = False, Loss_dict ={}):

    loss_dict = {'epoch': [], 'decoder': [], 'sindy_x': [], 'reg': [], 'sindy_z': [], 'active_coeffs': []}
    pretrain_epochs = train_params[f'{mode}_epochs']
    optimizer = torch.optim.Adam(net.parameters(), lr= model_params['learning_rate'])

    for epoch in range(pretrain_epochs):
        total_loss, total_loss_dict = train_one_epoch(net, train_loader, optimizer)

        if not isinf(print_freq):
            if not epoch % print_freq:
                if printout:
                    pass
                    print(f'TRAIN Epoch {net.epoch}: Active coeffs: {net.num_active_coeffs}, {[f"{key}: {val.cpu().detach().numpy()}" for (key, val) in total_loss_dict.items()]}')

                if len(test_loader):
                    test_loss, test_loss_dict = validate_one_epoch(net, test_loader)
                    for key,val in test_loss_dict.items():
                        loss_dict[key].append(float(val.detach().cpu()))
                    loss_dict['epoch'].append(float(net.epoch.detach().cpu()))
                    loss_dict['active_coeffs'].append(net.num_active_coeffs)

                    if printout:
                        pass
                        print(f'TEST Epoch {net.epoch}: Active coeffs: {net.num_active_coeffs}, {[f"test_{key}: {val.cpu().detach().numpy()}" for (key, val) in test_loss_dict.items()]}')

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
    x, y = net.sindy_coeffs.shape
    coeff_trajectories = {}
    for ix in range(x):
        for iy in range(y):
            coeff_trajectories[f'{ix}_{iy}_trajectory'] = []
    net.coeff_trajectories = coeff_trajectories
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
            net, loss_dict, Loss_dict = subtrain_sindy(net, train_loader, model_params,train_params,
                                            mode='subtrain', print_freq = 50, test_loader = test_loader,
                                            printout= printout, Loss_dict = Loss_dict)
    if train_params['refinement_epochs']:
        net, loss_dict, Loss_dict = subtrain_sindy(net, train_loader, model_params, train_params,
                                    mode='refinement', print_freq=50, test_loader=test_loader,
                                    printout=printout, Loss_dict = Loss_dict)
    return net, Loss_dict