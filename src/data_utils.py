import sys
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")

import os
#import datetime
#import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
import torch
import pickle
import warnings
from sindy_utils import library_size
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")




class model_data(Dataset):

    def __init__(self, data={}, params = {}):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.data_dict = data
        self.x = torch.tensor(self.data_dict['x'], dtype=torch.float32)
        self.dx = torch.tensor(self.data_dict['dx'], dtype=torch.float32)
        self.params = params
        if self.params['model_order'] == 2:
            self.dxx = torch.tensor(self.data_dict['dxx'], dtype=torch.float32)
        self.n_samples = self.x.shape[1]


    def __getitem__(self, index):
        if self.params['model_order'] == 2:
            return {'x': self.x[index], 'dx': self.dx[index], 'dxx': self.dxx[index]}
        else:
            return {'x': self.x[index], 'dx': self.dx[index]}


    def __len__(self):
        return self.n_samples


def get_test_params():
    noise_strength = 1e-6
    training_data = get_lorenz_data(100, noise_strength=noise_strength)
    validation_data = get_lorenz_data(20, noise_strength=noise_strength)
    params = {}

    params['input_dim'] = 128
    params['latent_dim'] = 3
    params['model_order'] = 1
    params['poly_order'] = 3
    params['include_sine'] = False
    params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

    # sequential thresholding parameters
    params['sequential_thresholding'] = True
    params['coefficient_threshold'] = 0.1
    params['threshold_frequency'] = 500
    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
    params['coefficient_initialization'] = 'constant'

    # loss function weighting
    params['loss_weight_decoder'] = 1.0
    params['loss_weight_sindy_z'] = 0.0
    params['loss_weight_sindy_x'] = 1e-4
    params['loss_weight_sindy_regularization'] = 1e-5

    params['activation'] = 'sigmoid'
    params['widths'] = [64,32]

    # training parameters
    params['epoch_size'] = training_data['x'].shape[0]
    params['batch_size'] = 5
    params['learning_rate'] = 1e-3

    params['data_path'] = os.getcwd() + '/'
    params['print_progress'] = True
    params['print_frequency'] = 1

    # training time cutoffs
    params['max_epochs'] = 1000
    params['refinement_epochs'] = 200
    return params,training_data, validation_data


def get_loader(data, params, workers = 4):
    data_class = model_data(data, params)
    return DataLoader(data_class, batch_size=params['batch_size'], num_workers=workers)



def run():
    params, training_data, validation_data = get_test_params()

    training_data = model_data(training_data, params)
    validation_data = model_data(validation_data, params)

    train_loader = DataLoader(training_data, batch_size=params['batch_size'], num_workers=4)
    validate_loader = DataLoader(validation_data, batch_size=params['batch_size'], num_workers=4)

    dataiter = iter(train_loader)
    data = dataiter.next()
    print([item.shape for item in data])


if __name__=='__main__':
    run()