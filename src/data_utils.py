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
warnings.filterwarnings("ignore")


def augment_sample(sample, n_samples, indexes, device):
    L = len(sample)
    indexes = random.choices(indexes, k = L)
    augmented_sample = []
    for i,tensor in enumerate(sample):
        index_tensor = torch.tensor([indexes[i]], device = device, dtype = torch.float32)
        augmented_tensor = torch.cat((tensor, index_tensor))
        augmented_sample.append(augmented_tensor)
    return torch.stack(augmented_sample)


def make_samples(tensors, n_samples, sample_size, device, augment = False):
    samples = [[] for tensor in tensors]
    indexes = list(range(0,tensors[0].shape[0]))
    for i in range(n_samples):
        sub_indexes = random.choices(indexes, k = sample_size)
        for i,tensor in enumerate(tensors):
            sample = torch.index_select(tensor, 0, torch.tensor(sub_indexes, device = device))
            if augment:
                sample = augment_sample(sample, n_samples, indexes, device)
            samples[i].append(sample)

    for i,Sample in enumerate(samples):
        shape = [n_samples * sample_size] + list(tensors[i].shape[1:])
        if augment:
            shape[1] += 1
        samples[i] = torch.stack(Sample).reshape(shape)
    return samples


class model_data(Dataset):
    def __init__(self, data={}, params = {}, device = None, bag_params ={}):
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
            x_bags,dx_bags = make_samples([self.x,self.dx], n_samples = bag_params['nbags'], augment = bag_params['augment'],
                                          sample_size = bag_params['bag_size'], device = self.device)
            self.x_bags = x_bags
            self.dx_bags = dx_bags
            self.n_samples = self.x_bags.shape[0]
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


def get_test_params(train_size = 100, max_data = 100000):
    noise_strength = 1e-6
    training_data = get_lorenz_data(train_size, noise_strength=noise_strength)
    validation_data = get_lorenz_data(20, noise_strength=noise_strength)

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
    params['loss_weight_mystery'] = 0

    params['activation'] = 'sigmoid'
    params['widths'] = [64,32]

    # training parameters
    params['epoch_size'] = training_data['x'].shape[0]
    params['batch_size'] = min([params['epoch_size']//8, train_size])
    params['threshold_frequency'] = 5
    params['learning_rate'] = 1e-3

    params['data_path'] = os.getcwd() + '/'
    params['print_progress'] = True
    params['print_frequency'] = 1

    # training time cutoffs
    params['max_epochs'] = 5000
    params['refinement_epochs'] = 2000
    params['crossval_freq'] = 200
    params['test_freq'] = 50
    params['train_print_freq'] = np.inf
    params['update_freq'] = 50
    params['use_activation_mask'] = False
    params['true_coeffs'] = training_data['sindy_coefficients']

    return params,training_data, validation_data


def get_loader(data, params, workers = 0, device = 'cpu'):
    data_class = model_data(data, params, device)
    return DataLoader(data_class, batch_size=params['batch_size'], num_workers=workers)


def get_bag_loader(data, train_params, model_params,  workers = 0, device = 'cpu', augment = False):
    train_params['augment'] = augment
    data_class = model_data(data, model_params, device, bag_params = train_params)
    return DataLoader(data_class, batch_size=train_params['bag_size'], num_workers=workers)


def run():
    model_params, training_data, validation_data = get_test_params()
    train_params = {'bag_epochs': 1, 'pretrain_epochs': 40, 'nbags': 5000, 'bag_size': 7}
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    bag_loader = get_bag_loader(training_data, train_params, model_params, device=device)

    #training_data = model_data(training_data, params)
    #validation_data = model_data(validation_data, params)

    #train_loader = DataLoader(training_data, batch_size=params['batch_size'], num_workers=0)
    #validate_loader = DataLoader(validation_data, batch_size=params['batch_size'], num_workers=4)
    i = 0


if __name__=='__main__':
    run()