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
        loss = model.bag_loss(x = data['x_bag'], dx = data['dx_bag'])
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
        train_one_bagstep(bag_model, bag_data, optimizer)
    return bag_model.active_coeffs()


def frac_round(vec,val):
    vec *=(1/val)
    vec = np.asarray(np.asarray(vec, int), float)
    vec *= .05
    return vec


def process_bag_coeffs(Bag_coeffs, params, model):
    new_mask = np.zeros(Bag_coeffs.shape[1:])
    x,y = new_mask.shape

    n_samples = Bag_coeffs.shape[0]
    avg_coeffs = (1/n_samples) * torch.sum(Bag_coeffs, dim = 0)
    for ix in range(x):
        for iy in range(y):
             new_mask[ix, iy] = 1 if avg_coeffs[ix, iy] > .1 else 0
    new_mask = torch.tensor(new_mask, dtype = torch.float32, device = params['device'])
    #avg_coeffs = torch.tensor(avg_coeffs, dtype = torch.float32, device = params['device'])
    return new_mask#, avg_coeffs


def train_bag_epochs(model, bag_loader, params, train_params):
    Bag_coeffs = []
    for batch_index, bag_data in enumerate(bag_loader):
        bag_model = deepcopy(model)
        perturbation = .01 * torch.randn(bag_model.sindy_coeffs.shape)
        bag_model.sindy_coeffs = torch.nn.Parameter(perturbation + bag_model.sindy_coeffs, requires_grad = True)
        bag_coeffs = get_bag_coeffs(bag_model, bag_data, params, train_params)
        Bag_coeffs.append(bag_coeffs)
    Bag_coeffs = torch.stack(Bag_coeffs)
    new_mask = process_bag_coeffs(Bag_coeffs, params, model)
    coefficient_mask = new_mask * model.coefficient_mask
    #model.sindy_coeffs = torch.nn.Parameter(avg_coeffs, requires_grad = True)
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
    return model.active_coeffs(), total_loss, total_loss_dict



#def bagtrain_sindy(net, bag_loader, model_params, train_params, print_freq = inf):
    #optimizer = torch.optim.Adam(net.bag_parameters(), lr=model_params['learning_rate'])
    #bag_params = 0


def subtrain_sindy(net, train_loader, model_params, train_params, mode, print_freq = inf):
    pretrain_epochs = train_params[f'{mode}_epochs']
    optimizer = torch.optim.Adam(net.parameters(), lr= model_params['learning_rate'])
    for epoch in range(pretrain_epochs):
        total_loss, total_loss_dict = train_one_epoch(net, train_loader, optimizer)
        if not isinf(print_freq):
            if not epoch % print_freq:
                print(f'Epoch: {net.epoch}, Active coeffs: {net.num_active_coeffs}, {[f"{key}: {val.cpu().detach().numpy()}" for (key, val) in total_loss_dict.items()]}')
    return net


def train_sindy(model_params, train_params, training_data, validation_data):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    train_loader = get_loader(training_data, model_params, device=device)
    test_loader = get_loader(validation_data, model_params, device=device)

    net = SindyNet(model_params).to(device)
    net = subtrain_sindy(net, train_loader, model_params, train_params, mode = 'pretrain', print_freq = 99)
    if train_params['bag_epochs']:
        bag_loader = get_bag_loader(training_data, train_params, model_params, device=device)
        for epoch in range(train_params['bag_epochs']):
            net  = train_bag_epochs(net, bag_loader, model_params, train_params)
            net = subtrain_sindy(net, train_loader, model_params, train_params, mode='subtrain', print_freq = 25)
        else:
            return net



def print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm):
    """
    Print loss function values to keep track of the training progress.

    Arguments:
        sess - the tensorflow session
        i - the training iteration
        loss - tensorflow object representing the total loss function used in training
        losses - tuple of the individual losses that make up the total loss
        train_dict - feed dictionary of training data
        validation_dict - feed dictionary of validation data
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.
        Can be first or second order time derivatives depending on the model order.

    Returns:
        Tuple of losses calculated on the validation set.
    """
    training_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=train_dict)
    validation_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=validation_dict)
    print("Epoch %d" % i)
    print("   training loss {0}, {1}".format(training_loss_vals[0],
                                             training_loss_vals[1:]))
    print("   validation loss {0}, {1}".format(validation_loss_vals[0],
                                               validation_loss_vals[1:]))
    decoder_losses = sess.run((losses['decoder'], losses['sindy_x']), feed_dict=validation_dict)
    loss_ratios = (decoder_losses[0]/x_norm, decoder_losses[1]/sindy_predict_norm)
    print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)
    return validation_loss_vals


def create_feed_dictionary(data, params, idxs=None):
    """
    Create the feed dictionary for passing into tensorflow.

    Arguments:
        data - Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        params - Dictionary object containing model and training parameters. The relevant
        parameters are model_order (which determines whether the SINDy model predicts first or
        second order time derivatives), sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), coefficient_mask (optional if sequential
        thresholding is performed; 0/1 mask that selects the relevant coefficients in the SINDy
        model), and learning rate (float that determines the learning rate).
        idxs - Optional array of indices that selects which examples from the dataset are passed
        in to tensorflow. If None, all examples are used.

    Returns:
        feed_dict - Dictionary object containing the relevant data to pass to tensorflow.
    """
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    feed_dict = {}
    feed_dict['x:0'] = data['x'][idxs]
    feed_dict['dx:0'] = data['dx'][idxs]
    if params['model_order'] == 2:
        feed_dict['ddx:0'] = data['ddx'][idxs]
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
    feed_dict['learning_rate:0'] = params['learning_rate']
    return feed_dict
