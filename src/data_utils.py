import sys
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")

import os
import numpy as np
from example_lorenz import get_lorenz_data
import torch
import pickle
import warnings
import random
from sindy_utils import library_size
from torch.utils.data import Dataset, DataLoader
from copy import copy
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def augment_sample(sample):
    n_bags = len(sample)
    shuffled_samples = []
    l = len(sample[0])//n_bags
    for i in range(n_bags):
        shuffled_sample = [bag[i*l:(i+1)*(l)] for bag in sample]
        shuffle_shape = [n_bags*l] + list(sample[0].shape)[1:]
        shuffled_sample = torch.stack(shuffled_sample).reshape(*shuffle_shape)
        shuffled_samples.append(shuffled_sample)
    return shuffled_samples


def make_samples(tensors, n_samples, sample_size, device, replacement = True, augment = False):
    samples = [[] for tensor in tensors]
    indexes = list(range(0,tensors[0].shape[0]))
    #sample_size = tensors[0].shape[0]
    for i in range(n_samples):
        if replacement:
            sub_indexes = random.choices(indexes, k = sample_size)
        else:
            sub_indexes = random.sample(indexes, k = sample_size)
        for i,tensor in enumerate(tensors):
            sample = torch.index_select(tensor, 0, torch.tensor(sub_indexes, device = device))
            samples[i].append(sample)

    for i,Sample in enumerate(samples):
        if augment:
            Sample = augment_sample(Sample)
        shape = [n_samples * sample_size] + list(tensors[i].shape[1:])
        try:
            samples[i] = torch.stack(Sample).reshape(shape)
        except BaseException:
            sample_stack = torch.stack(Sample)
            l = int(sample_stack.shape[0] * sample_stack.shape[1])
            sample_stack = sample_stack.reshape(l, shape[-1])
            paddings = (0,0, 0, shape[0]- sample_stack.shape[0])
            padded_stack = torch.nn.functional.pad(sample_stack,paddings, 'constant')
            samples[i] = padded_stack
    return samples

def expand_tensor(tensor, expansion_factor):
    tensor_shape = tensor.shape
    expanded_tensor = torch.stack([tensor for i in range(expansion_factor)]).reshape(tensor_shape[0] * expansion_factor, *tensor_shape[1:])
    return expanded_tensor

class model_data(Dataset):
    def __init__(self, data={}, params = {}, device = None, bag_params ={}, expand_factor = None):
        super().__init__()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        self.data_dict = data
        self.x = torch.tensor(self.data_dict['x'], dtype=torch.float32, device=self.device)
        self.dx = torch.tensor(self.data_dict['dx'], dtype=torch.float32, device=self.device)
        self.n_samples = self.x.shape[0]
        if bag_params:
            x_bags,dx_bags = make_samples([self.x,self.dx], n_samples = bag_params['nbags'],
                                          augment = bag_params['augment'], sample_size = bag_params['bag_size'],
                                          replacement = bag_params['replacement'], device = self.device)

            self.x_bags = x_bags
            self.dx_bags = dx_bags
            self.n_samples = self.x_bags.shape[0]
        if expand_factor:
            self.x = expand_tensor(self.x, expand_factor)
            self.dx = expand_tensor(self.dx, expand_factor)
            self.n_samples = self.x.shape[0]
        self.params = params
        self.bag_params = bag_params
        if self.params['model_order'] == 2:
            self.dxx = torch.tensor(self.data_dict['dxx'], dtype=torch.float32, device = self.device)

    def __getitem__(self, index):
        if self.bag_params:
            return {'x_bag': self.x_bags[index], 'dx_bag': self.dx_bags[index]}
        else:
            if self.params['model_order'] == 2:
                return {'x': self.x[index], 'dx': self.dx[index], 'dxx': self.dxx[index]}
            else:
                return {'x': self.x[index], 'dx': self.dx[index]}


    def __len__(self):
        return self.n_samples


def get_test_params(train_size = 100, max_data = 100000, noise = 1e-6):
    noise_strength = 1e-4
    training_data = get_lorenz_data(train_size, noise_strength=noise)
    validation_data = get_lorenz_data(20, noise_strength=noise)

    training_data = {key: vec[:min(max_data, len(training_data['x']))] for key,vec in training_data.items()}
    validation_data = {key: vec[:min(max_data, len(validation_data['x']))] for key, vec in validation_data.items()}
    params = {}

    params['input_dim'] = 128
    params['latent_dim'] = 3
    params['model_order'] = 1
    params['poly_order'] = 3
    params['include_sine'] = False
    params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

    # sequential thresholding parameters
    params['sequential_thresholding'] = False
    params['coefficient_threshold'] = 0.1
    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
    params['coefficient_initialization'] = 'constant'

    # loss function weighting
    params['loss_weight_decoder'] = 1.0
    params['loss_weight_sindy_z'] = 0.00
    params['loss_weight_sindy_x'] = 1e-4
    params['loss_weight_sindy_regularization'] = 1e-5
    params['loss_weight_consistency'] = 1e-2

    params['activation'] = 'sigmoid'
    params['widths'] = [64,32]

    # training parameters
    params['epoch_size'] = training_data['x'].shape[0]
    params['batch_size'] = min([params['epoch_size']//8, train_size])
    params['nbags'] = 1
    params['threshold_frequency'] = 5
    params['learning_rate'] = 1e-3

    params['data_path'] = os.getcwd() + '/'
    params['print_progress'] = True
    params['print_frequency'] = 1
    params['print_freq'] = 50

    # training time cutoffs
    params['max_epochs'] = 5000
    params['refinement_epochs'] = 2000
    params['crossval_freq'] = 200
    params['test_freq'] = 50
    params['train_print_freq'] = np.inf
    params['update_freq'] = 50
    params['use_activation_mask'] = False
    params['use_median'] = False
    params['avg_crossval'] = False
    params['c_loss'] = False
    params['scramble'] = False
    params['eval'] = False
    params['expand_sample'] = True
    params['hybrid_reg'] = False
    params['bagn_factor'] = 1
    params['true_coeffs'] = training_data['sindy_coefficients']

    return params,training_data, validation_data


def get_loader(data, params, workers = 0, device = 'cpu', expand_factor = None):
    data_class = model_data(data, params, device, expand_factor = expand_factor)
    return DataLoader(data_class, batch_size=params['batch_size'], num_workers=workers, shuffle=False)


def get_bag_loader(data, train_params, model_params,  workers = 0,
                   device = 'cpu', augment = False, replacement = True):
    train_params['augment'] = augment
    train_params['replacement'] = replacement
    data_class = model_data(data, model_params, device, bag_params = train_params)

    return DataLoader(data_class, batch_size=train_params['bag_size'], num_workers=workers, shuffle=False)


def run():
    model_params, training_data, validation_data = get_test_params()
    train_params = {'bag_epochs': 1, 'pretrain_epochs': 40, 'nbags': 5000, 'bag_size': 7}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    bag_loader = get_bag_loader(training_data, train_params, model_params, device=device)


if __name__=='__main__':
    run()