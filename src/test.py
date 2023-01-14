import sys
sys.path.append("../src")
sys.path.append("../tf_model/src")
sys.path.append("../examples/lorenz")

import os
import datetime
import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
import torch
from sindy_utils import library_size
from tf_training import train_network
from torch_autoencoder import Sindy_net
import tensorflow as tf
import pickle
import warnings


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





noise_strength = 1e-6
training_data = get_lorenz_data(100, noise_strength=noise_strength)

validation_data = get_lorenz_data(10, noise_strength=noise_strength)

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

batch_size = 7

Net = Sindy_net(params)
for i in range(len(training_data['x'])//batch_size  - 1):
    x = torch.tensor(training_data['x'][batch_size *i:batch_size *(i+1)], dtype=torch.float32)
    dx = torch.tensor(training_data['dx'][batch_size *i:batch_size *(i+1)], dtype=torch.float32)

    x_p, z = Net.forward(x)
    loss, loss_refinement, losses = Net.Loss(x, x_p, z, dx)

    loss.backward()
    print(losses)


'''
num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = 'lorenz_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()

    results_dict = train_network(training_data, validation_data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

#df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')


#autoencoder_network = full_network_torch(test_data['x'].shape[0], params)
#loss, losses, loss_refinement = define_loss(autoencoder_network, params)

#create_feed_dictionary(params)

#x:0 has shape (100000, 128)
#dx:0 has shape (100000, 128)
#coefficient_mask:0 has shape (20, 3)
#learning_rate:0 is 0.001


#train_network(test_data, val_data, params)

#autoencoder_network = full_network_torch(params)
#tf_autoencoder_network = full_network(params)
'''