import sys
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")

import os
import datetime
#import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
import torch
from sindy_utils import library_size
#from tf_training import train_network
import torch_training
from torch_autoencoder import SindyNet
#import tensorflow as tf
import pickle
import warnings
from data_utils import get_test_params,get_loader

warnings.filterwarnings("ignore")

#data_path = '../examples/lorenz/'
#data_path = '../examples/pendulum/'
#data_path = '../examples/rd/'

#save_name = 'model1'
#save_name = 'model2'

#params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))
#params['save_name'] = data_path + save_name


#x = torch.rand(params['input_dim'])
#dx = torch.rand(params['input_dim'])


def run():
    params,training_data, validation_data = get_test_params()
    params['batch_size'] = 100
    train_loader = get_loader(training_data, params)
    test_loader = get_loader(validation_data, params)

    net = SindyNet(params)
    optimizer = torch.optim.Adam(net.parameters(), lr = params['learning_rate'])
    for epoch in range(params['max_epochs']):
        total_loss, total_loss_dict = torch_training.train_one_epoch(net, train_loader, optimizer)
        print([f'{key}: {val.detach().numpy()} \n' for (key,val) in total_loss_dict.items()])

if __name__=='__main__':
    run()
